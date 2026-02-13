# python
# File: maskllm/adapter_from_notebook.py
import os
import copy
import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from timm import utils as timm_utils
from timm.models import create_model
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2

from maskllm.maskllm import MaskedLinear, MaskedLinearFrozen
from maskllm.sparsegpt import prune_sparsegpt  # available in repo; not used unless cfg requests it
from maskllm.utils import replace_linear_with_
from utils.analysis_isomorphism import ViTIsomorphicAnalyzer
from data.loaders import get_cifar10_loaders_pbench


class MaskLLMRunner:
    """
    Implements MaskLLM flow based on the attached notebook.
    Usage:
      runner = MaskLLMRunner()
      saved_path = runner.run(cfg)
    cfg is a dict with keys similar to notebook (model, sparse_checkpoint, mask_only, epochs, batch_size, ...)
    """

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    def _load_sparse_checkpoint_into(self, model: nn.Module, path: str):
        if not path:
            return
        if not os.path.exists(path):
            raise FileNotFoundError(f"sparse checkpoint not found: {path}")
        ck = torch.load(path, map_location="cpu")
        # notebook saved either raw state_dict or dict with 'model'
        state = ck.get("model", ck) if isinstance(ck, dict) else ck
        model.load_state_dict(state, strict=False)

    def _apply_priors_and_freeze(self, model: nn.Module, cfg: Dict[str, Any]):
        prior_strength = cfg.get("prior_strength")
        # load mask priors if module supports it
        if prior_strength is not None:
            for m in model.modules():
                if hasattr(m, "load_mask_prior"):
                    try:
                        m.load_mask_prior(prior_strength)
                    except Exception:
                        pass

        # mask-only training (freeze all except gates)
        if cfg.get("mask_only", True):
            for name, p in model.named_parameters():
                if ".gate" in name:
                    p.requires_grad = True
                else:
                    p.requires_grad = False

    def _build_optimizer_and_scheduler(self, model: nn.Module, cfg: Dict[str, Any]):
        optim_cfg = {
            "opt": cfg.get("opt", "adamw"),
            "lr": cfg.get("lr", 1e-3),
            "weight_decay": cfg.get("weight_decay", 0.01),
        }
        optimizer = create_optimizer_v2(model, **optim_cfg)

        sched_cfg = {
            "sched": cfg.get("sched", "cosine"),
            "num_epochs": cfg.get("epochs", 1),
            "warmup_epochs": cfg.get("warmup_epochs", 0),
            "min_lr": cfg.get("min_lr", 1e-4),
        }
        scheduler, num_epochs = create_scheduler_v2(optimizer, **sched_cfg)
        return optimizer, scheduler, num_epochs

    def _train_one_epoch(self, epoch, model, loader, optimizer, loss_fn, cfg, amp_autocast, loss_scaler):
        model.train()
        for batch_idx, (input, target) in enumerate(loader):
            input, target = input.to(self.device), target.to(self.device)

            def _forward():
                with amp_autocast():
                    output = model(input)
                    loss = loss_fn(output, target)
                    # optional sparse weight regularization
                    if cfg.get("sparse_weight_reg", 0) > 0:
                        for m in model.modules():
                            if isinstance(m, MaskedLinear) and hasattr(m, "sparse_weight_reg"):
                                loss = loss - cfg.get("sparse_weight_reg", 0) * m.sparse_weight_reg()
                return loss

            if loss_scaler is not None:
                loss_scaler(_forward(), optimizer, clip_grad=cfg.get("clip_grad", None), parameters=model.parameters())
            else:
                loss = _forward()
                loss.backward()
                if cfg.get("clip_grad") is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("clip_grad"))
                optimizer.step()
            optimizer.zero_grad()

    def _validate(self, model, loader, loss_fn, cfg, amp_autocast):
        model.eval()
        correct1 = 0
        correct5 = 0
        total = 0
        loss_sum = 0.0
        with torch.no_grad():
            for input, target in loader:
                input, target = input.to(self.device), target.to(self.device)
                with amp_autocast():
                    out = model(input)
                    loss = loss_fn(out, target)
                loss_sum += loss.item() * input.size(0)
                total += input.size(0)
                _, pred = out.topk(5, 1, True, True)
                pred = pred.t()
                correct = pred.eq(target.view(1, -1).expand_as(pred))
                correct1 += correct[:1].reshape(-1).float().sum(0, keepdim=True).item()
                correct5 += correct[:5].reshape(-1).float().sum(0, keepdim=True).item()
        if total == 0:
            return {"loss": None, "top1": 0.0, "top5": 0.0}
        return {"loss": loss_sum / total, "top1": (correct1 / total) * 100.0, "top5": (correct5 / total) * 100.0}

    def run(self, cfg: Dict[str, Any]) -> str:
        """
        Run the MaskLLM workflow using the notebook logic.
        Returns saved checkpoint path (model_final.pth).
        """
        model_name = cfg.get("model", "deit_tiny_patch16_224.fb_in1k")
        num_classes = cfg.get("num_classes", 10)

        # 1) create base model (pretrained=False as notebook does)
        model = create_model(model_name, pretrained=False, num_classes=num_classes)

        # 2) build ViT groups (if ViT analyzer works for model)
        try:
            analyser = ViTIsomorphicAnalyzer(model)
            groups = analyser.create_isomorphic_groups(
                target_macs=cfg.get("target_macs", None),
                baseline_macs=cfg.get("baseline_macs", None),
            )
        except Exception:
            groups = None

        # 3) replace linear layers with MaskedLinear for training
        # Use MaskedLinear (trainable) unless user explicitly asks for frozen variant
        replace_class = MaskedLinear if not cfg.get("use_frozen_masked", False) else MaskedLinearFrozen
        # try to figure out classifier to exclude
        exclude = []
        try:
            clf = getattr(model, "get_classifier", None)
            if callable(clf):
                exclude = [model.get_classifier()]
        except Exception:
            exclude = []

        replace_linear_with_(model, replace_class, exclude=exclude, groups=groups)

        # 4) load sparse checkpoint if provided
        sparse_ckpt = cfg.get("sparse_checkpoint")
        if sparse_ckpt:
            self._load_sparse_checkpoint_into(model, sparse_ckpt)
        else:
            # optional: prune with SparseGPT if no checkpoint provided
            if cfg.get("use_sparsegpt", False):
                target_sparsity = cfg.get("target_sparsity", 0.75)
                prune_sparsegpt(model, target_sparsity=target_sparsity, device=self.device)

        # 5) set priors + freeze masks if mask-only
        self._apply_priors_and_freeze(model, cfg)

        model.to(self.device)

        # 6) prepare data loaders
        batch_size = cfg.get("batch_size", 8)
        train_loader, val_loader = get_cifar10_loaders_pbench(batch_size, num_workers=cfg.get("num_workers", 4))

        # 7) optimizer and scheduler
        optimizer, scheduler, num_epochs = self._build_optimizer_and_scheduler(model, cfg)

        # 8) loss, amp, scaler
        amp = cfg.get("amp", True)
        amp_autocast = torch.cuda.amp.autocast if amp and torch.cuda.is_available() else torch.cpu.amp.autocast
        loss_scaler = timm_utils.NativeScaler() if amp and torch.cuda.is_available() else None
        train_loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.get("smoothing", 0.0)).to(self.device)
        validate_loss_fn = nn.CrossEntropyLoss().to(self.device)

        # 9) training loop (update tau/scaling each epoch)
        for epoch in range(num_epochs):
            # update tau/scaling per epoch
            tau0, tau1 = cfg.get("tau_range", [4.0, 0.05])
            s0, s1 = cfg.get("scaling_range", [1e1, 1e2])
            tau = tau0 + (tau1 - tau0) * epoch / max(num_epochs - 1, 1)
            scaling = s0 + (s1 - s0) * epoch / max(num_epochs - 1, 1)
            for m in model.modules():
                if isinstance(m, MaskedLinear):
                    setattr(m, "tau", tau)
                    setattr(m, "scaling", scaling)

            # train + validate
            self._train_one_epoch(epoch, model, train_loader, optimizer, train_loss_fn, cfg, amp_autocast, loss_scaler)
            metrics = self._validate(model, val_loader, validate_loss_fn, cfg, amp_autocast)
            # scheduler step (notebook uses lr_scheduler.step(epoch + 1))
            try:
                scheduler.step(epoch + 1)
            except Exception:
                pass

        # 10) save final model
        output_root = cfg.get("output", "output/maskllm_simplified")
        experiment = cfg.get("experiment", "MaskLLM-Simplified")
        outdir = timm_utils.get_outdir(output_root, experiment)
        os.makedirs(outdir, exist_ok=True)
        save_path = os.path.join(outdir, "model_final.pth")
        torch.save({"model": model.state_dict(), "config": cfg}, save_path)

        return save_path


def run_maskllm_from_strategy(cfg: Dict[str, Any]) -> str:
    """
    Convenience entrypoint for other agents. Blocking call.
    Returns path to saved model_final.pth.
    """
    runner = MaskLLMRunner()
    return runner.run(cfg)

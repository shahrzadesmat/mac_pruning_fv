#!/usr/bin/env python3  
"""Simplified MaskLLM training script for sparse Vision Transformers"""  
  
import os  
import torch  
import torch.nn as nn  
from timm import utils  
from timm.data import create_dataset, create_loader, resolve_data_config  
from timm.models import create_model, safe_model_name  
from timm.optim import create_optimizer_v2  
from timm.scheduler import create_scheduler_v2  
import sparsity  
  
def main():  
    # Fixed configuration for MaskLLM training  
    config = {  
        'model': 'vit_base_patch16_224',  
        'data_dir': 'data/imagenet',  
        'sparse_checkpoint': 'output/pruned/vit_base_patch16_224.augreg_in1k.sparsegpt24.pt',  
        'sparsity_mode': 'maskllm',  
        'mask_only': True,  
        'batch_size': 128,  
        'epochs': 20,  
        'lr': 1e-3,  
        'weight_decay': 0.01,  
        'opt': 'adamw',  
        'sched': 'cosine',  
        'warmup_epochs': 0,  
        'min_lr': 1e-4,  
        'tau_range': [4, 0.05],  
        'scaling_range': [1e1, 1e2],  
        'prior_strength': 3,  
        'sparse_weight_reg': 1e-5,  
        'clip_grad': 2.0,  
        'mixup': 0.8,  
        'cutmix': 1.0,  
        'smoothing': 0.1,  
        'drop_path': 0.1,  
        'aa': 'rand-m8-inc1-mstd101',  
        'reprob': 0.3,  
        'remode': 'pixel',  
        'amp': True,  
        'output': 'output/maskllm_simplified',  
        'experiment': 'MaskLLM-Simplified'  
    }  
      
    # Setup distributed training  
    device = utils.init_distributed_device(type(config).__dict__)  
    utils.random_seed(42, utils.get_rank())  
      
    # Create model with MaskLLM layers  
    model = create_model(config['model'], pretrained=False)  
      
    # Replace linear layers with MaskedLinear for learnable masks  
    model = sparsity.utils.replace_linear_with_(  
        model,   
        sparsity.maskllm.MaskedLinear,   
        exclude=[model.get_classifier()],  
        N=2, M=4, hard=False  
    )  
      
    # Load sparse checkpoint and initialize mask priors  
    print(f'Loading sparse checkpoint from {config["sparse_checkpoint"]}')  
    checkpoint = torch.load(config['sparse_checkpoint'], map_location='cpu')  
    model.load_state_dict(checkpoint, strict=False)  
      
    # Initialize gate parameters from prior masks  
    for name, module in model.named_modules():  
        if isinstance(module, sparsity.maskllm.MaskedLinear):  
            module.load_mask_prior(prior_strength=config['prior_strength'])  
      
    # Freeze all parameters except gates  
    if config['mask_only']:  
        print('Mask only mode - freezing all parameters except gates...')  
        for name, param in model.named_parameters():  
            if '.gate' not in name:  
                param.requires_grad = False  
      
    model.to(device=device)  
      
    # Setup data  
    data_config = resolve_data_config(config, model=model)  
    dataset_train = create_dataset('', root=config['data_dir'], split='train', is_training=True)  
    dataset_eval = create_dataset('', root=config['data_dir'], split='validation', is_training=False)  
      
    loader_train = create_loader(  
        dataset_train, input_size=data_config['input_size'],  
        batch_size=config['batch_size'], is_training=True,  
        re_prob=config['reprob'], re_mode=config['remode'],  
        auto_augment=config['aa'], mean=data_config['mean'],  
        std=data_config['std'], distributed=device.type != 'cpu'  
    )  
      
    loader_eval = create_loader(  
        dataset_eval, input_size=data_config['input_size'],  
        batch_size=config['batch_size'], is_training=False,  
        mean=data_config['mean'], std=data_config['std']  
    )  
      
    # Setup optimizer and scheduler  
    optimizer = create_optimizer_v2(model, **config)  
    lr_scheduler, num_epochs = create_scheduler_v2(optimizer, **config)  
      
    # Setup loss functions  
    train_loss_fn = nn.CrossEntropyLoss(label_smoothing=config['smoothing']).to(device)  
    validate_loss_fn = nn.CrossEntropyLoss().to(device)  
      
    # Setup AMP  
    amp_autocast = torch.cuda.amp.autocast if config['amp'] else suppress  
    loss_scaler = utils.NativeScaler() if config['amp'] else None  
      
    # Training loop  
    for epoch in range(num_epochs):  
        # Update tau and scaling for MaskLLM  
        tau = config['tau_range'][0] + (config['tau_range'][1] - config['tau_range'][0]) * epoch / max(num_epochs - 1, 1)  
        scaling = config['scaling_range'][0] + (config['scaling_range'][1] - config['scaling_range'][0]) * epoch / max(num_epochs - 1, 1)  
          
        for m in model.modules():  
            if isinstance(m, sparsity.maskllm.MaskedLinear):  
                m.tau = tau  
                m.scaling = scaling  
          
        print(f'Epoch {epoch}: tau={tau}, scaling={scaling}')  
          
        # Train one epoch  
        train_one_epoch(epoch, model, loader_train, optimizer, train_loss_fn,   
                       config, amp_autocast=amp_autocast, loss_scaler=loss_scaler)  
          
        # Validate  
        eval_metrics = validate(model, loader_eval, validate_loss_fn, config,   
                              device=device, amp_autocast=amp_autocast)  
          
        print(f'Epoch {epoch}: Top-1: {eval_metrics["top1"]:.2f}%, Top-5: {eval_metrics["top5"]:.2f}%')  
          
        # Step scheduler  
        lr_scheduler.step(epoch + 1)  
      
    # Save final model  
    output_dir = utils.get_outdir(config['output'], config['experiment'])  
    torch.save({  
        'model': model.state_dict(),  
        'config': config,  
    }, os.path.join(output_dir, 'model_final.pth'))  
  
def train_one_epoch(epoch, model, loader, optimizer, loss_fn, config,   
                   amp_autocast=suppress, loss_scaler=None):  
    model.train()  
    for batch_idx, (input, target) in enumerate(loader):  
        input, target = input.to(model.device), target.to(model.device)  
          
        def _forward():  
            with amp_autocast():  
                output = model(input)  
                loss = loss_fn(output, target)  
                # Add sparse weight regularization  
                if config['sparse_weight_reg'] > 0:  
                    for m in model.modules():  
                        if isinstance(m, sparsity.maskllm.MaskedLinear):  
                            loss += -config['sparse_weight_reg'] * m.sparse_weight_reg()  
            return loss  
          
        if loss_scaler is not None:  
            loss_scaler(_forward(), optimizer, clip_grad=config['clip_grad'])  
        else:  
            loss = _forward()  
            loss.backward()  
            if config['clip_grad'] is not None:  
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_grad'])  
            optimizer.step()  
          
        optimizer.zero_grad()  
  
def validate(model, loader, loss_fn, config, device, amp_autocast=suppress):  
    model.eval()  
    correct1 = correct5 = total = 0  
    loss_sum = 0  
      
    with torch.no_grad():  
        for input, target in loader:  
            input, target = input.to(device), target.to(device)  
            with amp_autocast():  
                output = model(input)  
                loss = loss_fn(output, target)  
              
            loss_sum += loss.item() * input.size(0)  
            total += input.size(0)  
              
            # Calculate accuracy  
            _, pred = output.topk(5, 1, True, True)  
            pred = pred.t()  
            correct = pred.eq(target.view(1, -1).expand_as(pred))  
            correct1 += correct[:1].reshape(-1).float().sum(0, keepdim=True)  
            correct5 += correct[:5].reshape(-1).float().sum(0, keepdim=True)  
      
    return {  
        'loss': loss_sum / total,  
        'top1': (correct1 / total).item() * 100,  
        'top5': (correct5 / total).item() * 100  
    }  
  
if __name__ == '__main__':  
    main()
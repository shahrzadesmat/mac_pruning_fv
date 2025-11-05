import os
import re
import copy
from datetime import datetime
from typing import Dict
import torch
from torch import nn
import torch_pruning as tp
import timm
import pbench
from utils.timing import time, time_it, time_it_async
from utils.analysis_structures import PruningState
from utils.analysis_isomorphism import ViTIsomorphicAnalyzer
from utils.pruning_safety import PruningSafetyValidator
from utils.pruning_math import (
    calculate_mac_targets,
    extract_pruning_ratio,
    extract_mac_target,
)
from utils.json_utils import parse_llm_json_response, deep_merge
import pbench  # Your custom pruning benchmarks
from data.loaders import (
    get_cifar10_loaders_pbench, 
    get_imagenet_folder_loaders_pbench
)

from utils.analysis_structures import (
    PruningState,
    MLPCouple,
    AttentionCouple,
)
from utils.analysis_isomorphism import IsomorphicGroup
from utils.logging_wandb import log_to_wandb
from utils.model_factory import get_model


MIN_HEAD_DIM = 6       # Minimum 32 for stable attention
MAX_SAFE_ATTN_PRUNE = 0.9  # Max 85% pruning on attention
MIN_MLP_DIM = 48       # Can go lower - MLP is less critical

class PruningAgent:
    def __init__(self):
        pass

    def get_device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def _prepare_model(self, model_name: str, device: torch.device, dataset: str = "cifar10", num_classes: int = 10):
        """Enhanced model preparation with support for new architectures"""
        
        # Handle different model name formats
        model_name_normalized = self._normalize_model_name(model_name)
        
        if dataset.lower() == 'imagenet':
            try:
                model = get_model(model_name_normalized, num_classes, pretrained=True)
                # print(f"[🔧] Created ImageNet model: {model_name_normalized} with pretrained weights")
            except Exception as e:
                # print(f"[⚠️] Failed to create {model_name_normalized} with pretrained weights: {e}")
                # Try without pretrained
                model = get_model(model_name_normalized, num_classes, pretrained=False)
                # print(f"[🔧] Created ImageNet model: {model_name_normalized} without pretrained weights")
        else:
            model = get_model(model_name_normalized, num_classes, pretrained=False)
            # print(f"[🔧] Created {dataset} model: {model_name_normalized}")
        
        for param in model.parameters():
            param.requires_grad = True
        return model.to(device)
    
    def _normalize_model_name(self, model_name: str) -> str:
        """Normalize model names to match timm conventions"""
        name_mapping = {
            'convnext-s': 'convnext_small',
            'convnext_s': 'convnext_small',
            'nvit-s': 'vit_small_patch16_224',  # Map to closest equivalent
            'nvit_s': 'vit_small_patch16_224',
            'upop': 'vit_base_patch16_224',     # Map to base ViT if Upop not available
            'vit-slim': 'vit_small_patch16_224', # Map to small ViT variant
            'vit_slim': 'vit_small_patch16_224',
            'resnet-101': 'resnet101',
            'resnet-50': 'resnet50',
            'resnet_101': 'resnet101',
            'resnet_50': 'resnet50',
        }
        
        model_name_lower = model_name.lower().replace('-', '_')
        return name_mapping.get(model_name_lower, model_name)

    def _calculate_importance(self, model, loader, criterion, device, importance_type="taylor"):
        """Calculate importance based on the specified importance criterion"""
        if importance_type == "taylor":
            # For taylor importance, we need gradients
            model.train()
            model.zero_grad()
            
            # print(f"[🧮] Calculating Taylor importance using gradient accumulation...")
            for batch_idx, batch in enumerate(loader):
                if batch_idx >= 100:  # Limit batches for efficiency
                    break
                
                # Handle different data formats (CIFAR-10 vs ImageNet arrow format)
                if isinstance(batch, dict):
                    # ImageNet arrow format
                    inputs = batch['pixel_values'].to(device)
                    targets = batch['label'].to(device)
                else:
                    # Standard CIFAR-10 format
                    inputs, targets = batch
                    inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                
            # print(f"[🧮] Taylor importance calculation complete")
            return tp.importance.TaylorImportance()
            
        elif importance_type == "l1norm":
            # print(f"[🧮] Using L1 norm importance (no gradient computation needed)")
            return tp.importance.MagnitudeImportance(p=1)
        elif importance_type == "l2norm":
            # print(f"[🧮] Using L2 norm importance (no gradient computation needed)")
            return tp.importance.MagnitudeImportance(p=2)
        else:
            # print(f"[⚠️] Unknown importance type: {importance_type}, falling back to taylor")
            return tp.importance.TaylorImportance()

    def _setup_dataset_loader(self, dataset: str, data_path: str, batch_size: int = 64):
        """Setup dataset-appropriate data loader with caching handled by loaders.py"""
        
        if dataset.lower() == 'imagenet':
            # Get subset fraction from state
            imagenet_subset = getattr(self, '_current_state', {}).get('imagenet_subset', 1.0)
            train_loader, val_loader = get_imagenet_folder_loaders_pbench(
                data_path, batch_size, num_workers=16, subset_fraction=imagenet_subset
            )
            return train_loader, val_loader
        else:
            train_loader, val_loader = get_cifar10_loaders_pbench(batch_size, num_workers=16)
            return train_loader, val_loader

    def _setup_ignored_layers(self, model, num_classes: int, dataset: str):
        """Enhanced ignored layers setup for new architectures"""
        ignored_layers = []
        num_heads = {}
        
        for name, m in model.named_modules():
            # Preserve final classifiers
            if (('fc' in name) or ('classifier' in name) or ('head' in name)) and isinstance(m, nn.Linear):
                if m.out_features == num_classes:
                    ignored_layers.append(m)
                    # print(f"[🔒] Preserving final classifier: {name} ({num_classes} classes)")
            
            # ConvNext-specific layer preservation
            if 'convnext' in self.model_name.lower():
                # Preserve layer normalization layers
                if isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                    ignored_layers.append(m)
                    # print(f"[🔒] Preserving ConvNext normalization: {name}")
                
                # Preserve depthwise convolutions (critical for ConvNext)
                if isinstance(m, nn.Conv2d) and m.groups == m.in_channels:
                    ignored_layers.append(m)
                    # print(f"[🔒] Preserving ConvNext depthwise conv: {name}")
            
            # Standard attention handling
            if isinstance(m, timm.models.swin_transformer.WindowAttention):
                num_heads[m.qkv] = m.num_heads
                ignored_layers.append(m.relative_position_bias_table)
            
            if isinstance(m, timm.models.vision_transformer.Attention):
                num_heads[m.qkv] = m.num_heads
        
        return ignored_layers, num_heads
    

    def _evaluate_model(self, model, loader, device, dataset: str):
        """Dataset-aware model evaluation"""
        model.eval()
        correct = 0
        total = 0

        # For ImageNet, also track Top-5
        if dataset.lower() == 'imagenet':
            correct_top5 = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                # 🛠️ Removed the batch cap here

                # Handle different data formats
                if isinstance(batch, dict):
                    # ImageNet arrow format
                    inputs = batch['pixel_values'].to(device)
                    labels = batch['label'].to(device)
                else:
                    # Standard CIFAR-10 format
                    inputs, labels = batch
                    inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)

                # Top-1 accuracy
                _, preds = outputs.max(1)
                total += labels.size(0)
                correct += preds.eq(labels).sum().item()

                # ADD DEBUG HERE - after preds are computed
                # if batch_idx == 0:  # Only first batch
                #     print(f"[DEBUG] Batch {batch_idx} - True labels: {labels[:20]}")
                #     print(f"[DEBUG] Batch {batch_idx} - Predictions: {preds[:20]}")

                # Top-5 accuracy for ImageNet
                if dataset.lower() == 'imagenet':
                    _, top5_preds = outputs.topk(5, 1, True, True)
                    correct_top5 += top5_preds.eq(labels.view(-1, 1).expand_as(top5_preds)).sum().item()

        accuracy = 100. * correct / total

        if dataset.lower() == 'imagenet':
            top5_accuracy = 100. * correct_top5 / total
            print(f"[🎯] Zero-shot accuracy: Top-1: {accuracy:.2f}%, Top-5: {top5_accuracy:.2f}%")
            return accuracy, top5_accuracy
        else:
            print(f"[🎯] Zero-shot accuracy: {accuracy:.2f}%")
            return accuracy, None


    def _is_vision_transformer(self, model: nn.Module, model_name: str) -> bool:
        """Detect if model is a Vision Transformer."""
        vit_indicators = ['vit', 'deit', 'beit', 'swin', 'nvit', 'upop']
        
        if any(indicator in model_name.lower() for indicator in vit_indicators):
            return True
        
        # Check for attention mechanisms
        has_attention = any(
            hasattr(m, 'qkv') or hasattr(m, 'num_heads')
            for m in model.modules()
        )
        return has_attention


    def _validate_model_integrity(self, model):
        """Ensures the pruned model can forward pass without crashing"""
        test_input = torch.randn(1, 3, 224, 224).to(next(model.parameters()).device)
        try:
            # Tokens should already exist - DON'T recreate them!
            # If they're None, something went wrong earlier
            if hasattr(model, "cls_token") and model.cls_token is None:
                print("[⚠️] WARNING: cls_token is None after pruning - this shouldn't happen!")
                print("[⚠️] Creating random token as fallback, but accuracy will be poor")
                model.cls_token = nn.Parameter(
                    torch.randn(1, 1, model.embed_dim, device=test_input.device) * 0.02
                )
            
            if hasattr(model, "dist_token") and model.dist_token is None:
                print("[⚠️] WARNING: dist_token is None after pruning - this shouldn't happen!")  
                model.dist_token = nn.Parameter(
                    torch.randn(1, 1, model.embed_dim, device=test_input.device) * 0.02
                )

            _ = model(test_input)
            print("[✅] Model integrity validated")
        except Exception as e:
            raise RuntimeError(f"Model broken after pruning: {e}")

    def _prune_attention_heads(self, group: IsomorphicGroup):
        """Prune attention heads by reducing num_heads."""
        for layer in group.layers:
            if hasattr(layer, 'num_heads') and hasattr(layer, 'qkv'):
                original_heads = layer.num_heads
                heads_to_remove = max(1, int(original_heads * group.pruning_ratio))
                new_heads = max(1, original_heads - heads_to_remove)
                
                # Update attention module
                layer.num_heads = new_heads
                if hasattr(layer, 'qkv'):
                    embed_dim = layer.qkv.out_features // 3
                    layer.head_dim = embed_dim // new_heads
                    layer.scale = layer.head_dim ** -0.5
                
                print(f"[🔧] Attention heads: {original_heads} -> {new_heads}")

    def _fix_model_structure(self, model: nn.Module):
        """Fix any structural issues after direct pruning."""
        print(f"[🔧] Fixing model structure after direct pruning...")
        
        # Update any cached dimension attributes
        for name, module in model.named_modules():
            if hasattr(module, 'qkv') and hasattr(module, 'num_heads'):
                try:
                    qkv_out = module.qkv.out_features
                    embed_dim = qkv_out // 3
                    
                    # Ensure num_heads divides embed_dim
                    if embed_dim % module.num_heads != 0:
                        # Find the largest factor of embed_dim that's <= num_heads
                        for new_heads in range(module.num_heads, 0, -1):
                            if embed_dim % new_heads == 0:
                                print(f"[🔧] Adjusting {name}: heads {module.num_heads} -> {new_heads}")
                                module.num_heads = new_heads
                                break
                    
                    # Update head_dim and scale
                    module.head_dim = embed_dim // module.num_heads
                    module.scale = module.head_dim ** -0.5
                    
                except Exception as e:
                    print(f"[⚠️] Could not fix {name}: {e}")

    def _fix_vit_attention_dimensions(self, model: nn.Module):
        """Fix ViT attention dimensions after pruning."""
        for name, module in model.named_modules():
            if hasattr(module, 'qkv') and hasattr(module, 'num_heads'):
                try:
                    qkv_out = module.qkv.out_features
                    embed_dim = qkv_out // 3
                    
                    # Ensure divisibility
                    if embed_dim % module.num_heads != 0:
                        # Find compatible number of heads
                        for heads in range(module.num_heads, 0, -1):
                            if embed_dim % heads == 0:
                                module.num_heads = heads
                                break
                    
                    module.head_dim = embed_dim // module.num_heads
                    module.scale = module.head_dim ** -0.5
                    
                except Exception as e:
                    print(f"[⚠️] Could not fix {name}: {e}")


    def _coordinate_layer_dimensions(self, model: nn.Module):
        """Coordinate dimensions between connected layers after pruning."""
        print(f"[🔧] Coordinating layer dimensions after pruning...")
        
        for name, module in model.named_modules():
            if hasattr(module, 'qkv') and hasattr(module, 'proj'):
                try:
                    # Get QKV output dimensions
                    qkv_out = module.qkv.out_features
                    embed_dim = qkv_out // 3  # Q, K, V each get embed_dim channels
                    
                    # Current projection layer expects this input
                    current_proj_in = module.proj.in_features
                    
                    if embed_dim != current_proj_in:
                        print(f"[🔧] Fixing {name}: QKV outputs {embed_dim}, but proj expects {current_proj_in}")
                        
                        # Option 1: Adjust projection layer to match QKV output
                        if embed_dim < current_proj_in:
                            # Shrink projection layer input
                            self._resize_linear_input(module.proj, embed_dim)
                            print(f"[✂️] Resized {name}.proj input: {current_proj_in} -> {embed_dim}")
                        
                        # Update attention module attributes
                        if hasattr(module, 'num_heads'):
                            # Ensure divisibility
                            while embed_dim % module.num_heads != 0 and module.num_heads > 1:
                                module.num_heads -= 1
                            
                            module.head_dim = embed_dim // module.num_heads
                            module.scale = module.head_dim ** -0.5
                            
                            print(f"[🔧] Updated {name}: embed_dim={embed_dim}, heads={module.num_heads}, head_dim={module.head_dim}")
                            
                except Exception as e:
                    print(f"[⚠️] Could not coordinate {name}: {e}")

    def _resize_linear_input(self, layer: nn.Linear, new_input_dim: int):
        """Resize a linear layer's input dimension."""
        if layer.in_features <= new_input_dim:
            return  # No need to resize
            
        print(f"[✂️] Resizing linear layer input: {layer.in_features} -> {new_input_dim}")
        
        # Select most important input channels
        with torch.no_grad():
            # Calculate importance of input channels
            importance = torch.norm(layer.weight, dim=0)  # Norm per input channel
            _, top_indices = torch.topk(importance, new_input_dim, largest=True)
            top_indices = torch.sort(top_indices)[0]
            
            # Create new smaller weight matrix
            new_weight = layer.weight.data[:, top_indices]
            
            # Update layer
            layer.weight = nn.Parameter(new_weight)
            layer.in_features = new_input_dim

    def _coordinate_mlp_dimensions(self, model: nn.Module):
        """Fix MLP layer dimension mismatches."""
        print(f"[🔧] Coordinating MLP dimensions...")
        
        for name, module in model.named_modules():
            if hasattr(module, 'fc1') and hasattr(module, 'fc2'):
                try:
                    fc1_out = module.fc1.out_features
                    fc2_in = module.fc2.in_features
                    
                    if fc1_out != fc2_in:
                        print(f"[🔧] MLP mismatch in {name}: fc1 outputs {fc1_out}, fc2 expects {fc2_in}")
                        
                        # Adjust fc2 input to match fc1 output
                        if fc1_out < fc2_in:
                            self._resize_linear_input(module.fc2, fc1_out)
                            print(f"[✂️] Resized {name}.fc2 input: {fc2_in} -> {fc1_out}")
                            
                except Exception as e:
                    print(f"[⚠️] Could not coordinate MLP {name}: {e}")    

    def _detect_model_architecture(self, model_name, model=None):
        """Enhanced model detection for new architectures"""
        model_name_lower = model_name.lower()
        
        # Check for transformer/attention-based models (ViT family)
        vit_indicators = ['vit', 'deit', 'beit', 'swin', 'nvit', 'upop']
        
        # Only check model.modules() if model is not None
        has_attention = False
        if model is not None:
            has_attention = any(isinstance(m, timm.models.vision_transformer.Attention) for m in model.modules())
        
        if any(indicator in model_name_lower for indicator in vit_indicators) or has_attention:
            return 'vit'
        
        # Check for ConvNext models
        if 'convnext' in model_name_lower or 'convneXt' in model_name:
            return 'convnext'
        
        # Check for ResNet variants
        if 'resnet' in model_name_lower:
            return 'resnet'
        
        # Default to CNN for unknown models
        return 'cnn'


    @time_it_async("4. Pruning Agent")
    async def execute_pruning(self, state: Dict) -> Dict:
        """Enhanced pruning with Analysis Agent recommendations"""
        print("\n[🔧] Executing dataset-aware pruning with Analysis Agent recommendations...")

        self._current_state = state

        # Extract dataset information
        dataset = state.get("dataset", "cifar10")
        num_classes = state.get("num_classes", 10)
        input_size = state.get("input_size", 224)
        data_path = state.get("data_path", "./data")
        
        # print(f"[🔧] Pruning for {dataset}: {num_classes} classes, {input_size}x{input_size}")

        # Extract model name
        model_name = state.get("model_name")
        if not model_name:
            query = state.get('query', '')
            match = re.search(r'Prune\s+(\w+(?:_\w+)*)\s+model', query)
            if match:
                model_name = match.group(1)
                # print(f"[📋] Extracted model name from query: {model_name}")
            else:
                raise ValueError("Model name is missing and couldn't be parsed from query.")
                
        # ONLY CHANGE: Enhanced CNN detection and routing
        model_name = state.get('model_name', 'unknown')

        # Detect model architecture type  
        arch_type = self._detect_model_architecture(model_name)

        cnn_types = ['resnet', 'efficientnet', 'mobilenet', 'densenet', 'resnext', 'convnext']
        vit_types = ['vit', 'deit', 'beit', 'swin', 'nvit', 'upop']

        is_vit = any(vit_type in model_name.lower() for vit_type in vit_types)
        is_cnn = (not is_vit and 
                (arch_type in ['resnet', 'convnext', 'cnn'] or 
                any(cnn_type in model_name.lower() for cnn_type in cnn_types)))

        if is_cnn:
            # print(f"[🔍] Detected CNN/ConvNext architecture: {model_name}")
            # print(f"[🔄] Routing to enhanced CNN pruning method...")
            return await self._execute_cnn_pruning(state, model_name)
        else:
            pass
            # print(f"[🔍] Detected ViT/Transformer architecture: {model_name}")
            # print(f"[🔄] Using existing ViT pruning method...")


        analysis_results = state.get("analysis_results", {})
        
        # Store analysis results for access in isomorphic pruning
        self._current_analysis_results = analysis_results

        target_macs = state.get("target_macs")
        target_ratio = None

        if target_macs is None:
            # Fallback to ratio-based approach
            target_ratio = analysis_results.get("pruning_ratio")
            if target_ratio is None:
                target_ratio = analysis_results.get("suggested_pruning_ratio")
                if target_ratio is None:
                    target_ratio = state.get("target_pruning_ratio")

            
            # Get base MACs to calculate target
            device = self.get_device()
            temp_model = self._prepare_model(model_name, device, dataset, num_classes)
            example_inputs = (torch.randn(1, 3, input_size, input_size).to(device),)
            base_macs, _ = tp.utils.count_ops_and_params(temp_model, example_inputs)
            target_macs = float(base_macs) * (1.0 - float(target_ratio))
            del temp_model  # Clean up
        else:
            target_ratio = None  # Will be derived from MACs later
                
        print(f"[🎯] Target pruning ratio: {target_ratio:.4f}" if target_ratio is not None else "[🎯] Using MACs-first approach, target ratio will be derived")
        
        # Extract importance criterion with FIXED logic
        current_revision = state.get('revision_number', 0)
        if current_revision == 0 and dataset.lower() == 'imagenet':
            importance_type = "taylor"
            print(f"[🔧] FORCING taylor for revision 0 (ImageNet baseline)")
        else:
            # Extract importance criterion with FIXED logic
            importance_type = None
            
            # 1. First try direct from analysis results
            importance_type = analysis_results.get("importance_criterion")
            
            # 2. Then try from strategy_dict  
            if importance_type is None:
                strategy_dict = analysis_results.get("strategy_dict", {})
                importance_type = strategy_dict.get("importance_criterion")
            
            # 3. Then try from master results (the LLM's original suggestion!)
            if importance_type is None:
                master_results = state.get("master_results", {})
                recommended_strategy = master_results.get("recommended_strategy", {})
                importance_type = recommended_strategy.get("importance_criterion")

            if importance_type is None:
                importance_type = "taylor"  # Always use Taylor for ImageNet
                    
        print(f"[🧮] FIXED: Using importance criterion: {importance_type}")
        
        # Extract round-to value
        round_to_value = analysis_results.get("round_to_value")
        if round_to_value is None:
            strategy_dict = analysis_results.get("strategy_dict", {})
            round_to_value = strategy_dict.get("round_to", None)
        print(f"[📏] Using round-to value: {round_to_value}")
        
        # Extract isomorphic group ratios
        isomorphic_group_ratios = analysis_results.get("isomorphic_group_ratios")
        if isomorphic_group_ratios is None:
            strategy_dict = analysis_results.get("strategy_dict", {})
            isomorphic_group_ratios = strategy_dict.get("isomorphic_group_ratios", {})

        current_revision = state.get('revision_number', 0)

        
        ATTN_ENABLED = os.getenv("IPRUN_ENABLE_ATTN", "1").lower() in ("1", "true", "yes")

        if dataset.lower() == "imagenet" and is_vit and current_revision == 0:
            ATTN_ENABLED = os.getenv("IPRUN_ENABLE_ATTN", "1").lower() in ("1", "true", "yes")
            if not ATTN_ENABLED:
                print("[🛡️] Attention pruning disabled via IPRUN_ENABLE_ATTN=0 (first attempt)")
                isomorphic_group_ratios = (isomorphic_group_ratios or {})
                isomorphic_group_ratios["qkv_multiplier"] = 0.0
            else:
                print("[🟢] Attention pruning enabled - ensuring QKV gets a fair chance")
                if isomorphic_group_ratios is None:
                    isomorphic_group_ratios = {}
                if "qkv_multiplier" not in isomorphic_group_ratios or isomorphic_group_ratios.get("qkv_multiplier", 0) == 0:
                    # ✅ FIX: persist the forced value and keep it accessible
                    forced_qkv = 0.15
                    isomorphic_group_ratios["qkv_multiplier"] = forced_qkv
                    print(f"[🔧] Set baseline QKV={forced_qkv} for revision 0 (LLM will learn from here)")



        elif current_revision > 0:
            print(f"[🧠] Revision {current_revision}: Using Analysis Agent's learned ratios")
            print(f"[🔧] Analysis Agent suggested ratios: {isomorphic_group_ratios}")
        else:
            print(f"[🔧] Using LLM-suggested isomorphic ratios: {isomorphic_group_ratios}")

        # If still no isomorphic ratios, use defaults that honor IPRUN_ENABLE_ATTN
        if not isomorphic_group_ratios:
            ATTN_ENABLED = os.getenv("IPRUN_ENABLE_ATTN", "1").lower() in ("1", "true", "yes")
            if dataset.lower() == "imagenet":
                isomorphic_group_ratios = {
                    "qkv_multiplier": 0.2 if ATTN_ENABLED else 0.0,  # Conservative 15%
                    "mlp_multiplier": 0.2,
                    "proj_multiplier": 0.2,
                    "head_multiplier": 0.2,
                }
            else:  # CIFAR-10
                isomorphic_group_ratios = {
                    "qkv_multiplier": 0.25 if ATTN_ENABLED else 0.0,  # Slightly more aggressive
                    "mlp_multiplier": 0.50,  # 50% for simpler dataset
                    "proj_multiplier": 0.0,
                    "head_multiplier": 0.0,
                }
            print(f"[🔧] Using conservative default {dataset} isomorphic ratios: {isomorphic_group_ratios}")
        else:
            print(f"[🔧] Using LLM-suggested isomorphic ratios: {isomorphic_group_ratios}")

        # Get rationale if available
        rationale = analysis_results.get("rationale")
        if rationale:
            print(f"[💡] Rationale: {rationale}")

        # Setup pruning parameters
        tolerance = 0.02  # Acceptable deviation from target MACs
        device = self.get_device()
        print(f"[💻] Using device: {device}")

        # Set global pruning to True for isomorphic pruning
        global_flag = True

        # Setup dataset-appropriate data loader
        try:
            # Adjust batch size based on dataset and available memory
            if dataset.lower() == 'imagenet':
                batch_size = 64  # Smaller batch size for ImageNet
            else:
                batch_size = 64  # Standard for CIFAR-10
                
            train_loader, val_loader = self._setup_dataset_loader(dataset, data_path, batch_size)
            criterion = nn.CrossEntropyLoss().to(device)

            # 💾 Save the unpruned original model for reattachment later
            print("[💾] Creating deep copy of unpruned original model before pruning...")
            original_model = self._prepare_model(model_name, device, dataset, num_classes)
            original_model.eval()
            state["original_model"] = copy.deepcopy(original_model)
            print("[✅] Original unpruned model stored in state for later reattachment.")
            

            # print(f"[📊] Setup {dataset} data loader with batch size {batch_size}")

            # # --- Baseline eval of the UNPRUNED model (must be sane) ---
            # print("[🧪] Evaluating UNPRUNED baseline model on validation set...")
            # _baseline_model = self._prepare_model(model_name, device, dataset, num_classes)
            #
            # # Important: put the model in eval mode
            # _baseline_model.eval()
            #
            # # Evaluate
            # base_eval = self._evaluate_model(_baseline_model, val_loader, device, dataset)
            #
            # # Log & sanity gate
            # if dataset.lower() == 'imagenet':
            #     base_top1, base_top5 = base_eval
            #     print(f"[📊] Baseline Top-1: {base_top1:.2f}% | Top-5: {base_top5:.2f}%")
            #     # Fail fast if clearly broken (adjust threshold if needed)
            #     if base_top1 < 50.0:
            #         raise RuntimeError(
            #             f"Baseline Top-1={base_top1:.2f}% is too low. "
            #             "Check ImageNet val transforms/normalization or pretrained weights."
            #         )
            #     # Optional: log to wandb
            #     log_to_wandb({
            #         "baseline_top1_accuracy": float(base_top1),
            #         "baseline_top5_accuracy": float(base_top5),
            #     }, step_name="baseline_eval", dataset=dataset)
            # else:
            #     base_top1 = base_eval[0]
            #     print(f"[📊] Baseline Top-1: {base_top1:.2f}%")
            #     if base_top1 < 80.0:
            #         raise RuntimeError(
            #             f"Baseline Top-1={base_top1:.2f}% is too low. "
            #             "Check data transforms/normalization or pretrained weights."
            #         )
            #     log_to_wandb({
            #         "baseline_top1_accuracy": float(base_top1),
            #     }, step_name="baseline_eval", dataset=dataset)
            #
            # # Cleanup
            # del _baseline_model
            # torch.cuda.empty_cache()


        except Exception as e:
            # print(f"[❌] Failed to setup data loader: {e}")
            return {**state, 'error': f'Data loader setup failed: {str(e)}'}

        # Tracking variables
        best_model = None
        best_accuracy = -float('inf')
        best_ratio = None
        best_state_dict = None
        best_top5_accuracy = None
        attempted_ratios = state.get('attempted_pruning_ratios', [])


        if target_ratio is None:
            # We're in MACs-first mode, need to calculate ratio
            device = self.get_device()
            temp_model = self._prepare_model(model_name, device, dataset, num_classes)
            example_inputs = (torch.randn(1, 3, input_size, input_size).to(device),)
            base_macs, _ = tp.utils.count_ops_and_params(temp_model, example_inputs)
            target_ratio = 1.0 - (target_macs / base_macs)
            del temp_model
            # print(f"[🔄] Calculated target ratio from MACs: {target_ratio:.4f}")

        # Now target_ratio is guaranteed to exist
        current_ratio = target_ratio
        if current_ratio in attempted_ratios:
            print(f"[⚠️] Ratio {current_ratio:.4f} was previously attempted, but Analysis Agent suggested it again.")

        # Record this attempt
        attempted_ratios.append(current_ratio)

        # Attempt pruning with the ratio suggested by Analysis Agent
        print(f"\n[🔁] Attempting pruning with ratio = {current_ratio:.4f}")

        try:
            # 🧩 1. Prepare model and base MACs
            model = self._prepare_model(model_name, device, dataset, num_classes)

            # INSERT AFTER LINE 717
            # print(f"\n[🔍] DEBUG 1: cls_token after _prepare_model")
            # if hasattr(model, 'cls_token') and model.cls_token is not None:
            #     print(f"  Shape: {model.cls_token.shape}, Mean: {model.cls_token.mean():.6f}, Std: {model.cls_token.std():.6f}")
            # else:
            #     print(f"  cls_token is None!")

            example_inputs = (torch.randn(1, 3, input_size, input_size).to(device),)
            base_params = sum(p.numel() for p in model.parameters())
            print(f"[📊] Base model parameters: {base_params:,}")

            print(f"[🧠] Using dependency-aware isomorphic pruning")
            analyzer = ViTIsomorphicAnalyzer(model)

            # Compute base MACs if not done yet
            if 'base_macs' not in locals():
                base_macs, _ = tp.utils.count_ops_and_params(model, example_inputs)

            target_ratio = 1.0 - (target_macs / base_macs)

            # Patch: avoid extremely small head_dim collapse
            if 'qkv_multiplier' in isomorphic_group_ratios:
                E = model.embed_dim
                pruning_ratio = isomorphic_group_ratios['qkv_multiplier']
                internal_dim = int(E * (1 - pruning_ratio))
                num_heads = getattr(model.blocks[0].attn, 'num_heads', 12)
                head_dim = internal_dim // num_heads

                if head_dim < MIN_HEAD_DIM:
                    head_dim = MIN_HEAD_DIM
                    internal_dim = head_dim * num_heads
                    new_pruning_ratio = 1 - (internal_dim / E)
                    new_pruning_ratio = round(new_pruning_ratio, 3)
                    print(f"[⚠️] Adjusted qkv_multiplier to avoid collapse: using {new_pruning_ratio:.3f} (head_dim = {head_dim})")
                    isomorphic_group_ratios['qkv_multiplier'] = new_pruning_ratio


            mlp_ratio = isomorphic_group_ratios.get("mlp_multiplier", 0.5)
            qkv_ratio = isomorphic_group_ratios.get("qkv_multiplier", 0.5)

            E = model.embed_dim
            internal_dim_qkv = int(E * (1 - qkv_ratio))
            num_heads = getattr(model.blocks[0].attn, "num_heads", 12)
            head_dim = internal_dim_qkv // num_heads

            mlp_hidden_dim = int(model.blocks[0].mlp.fc1.in_features * (1 - mlp_ratio))


            # 🚨 Check if MLP pruning is too aggressive (hidden dim collapse)
            if mlp_hidden_dim < MIN_MLP_DIM:
                print(f"[⚠️] MLP hidden dim = {mlp_hidden_dim} < {MIN_MLP_DIM}")
                print(f"[🛡️] KEEPING MLP at minimum instead of shifting load to attention!")
                
                # Clamp MLP to minimum viable dimension
                mlp_ratio = 1 - (MIN_MLP_DIM / model.blocks[0].mlp.fc1.in_features)
                mlp_ratio = max(0.0, min(0.99, round(mlp_ratio, 3)))
                isomorphic_group_ratios["mlp_multiplier"] = mlp_ratio
                
                # DO NOT increase attention pruning!
                # Attention is MORE critical than MLP - let the model miss MAC target instead
                print(f"[🛡️] Keeping QKV pruning at {qkv_ratio:.3f} - not shifting MLP shortage to attention")
                print(f"[💡] Model will have higher MACs than target, but preserve attention capacity")

            # 🛡️ SAFETY CAP: Never let attention pruning exceed safe limit
            MAX_SAFE_ATTN = 0.70  # 50% max
            if qkv_ratio > MAX_SAFE_ATTN:
                print(f"[🛡️] SAFETY CAP: QKV pruning {qkv_ratio:.3f} exceeds safe limit {MAX_SAFE_ATTN:.3f}")
                qkv_ratio = MAX_SAFE_ATTN
                isomorphic_group_ratios["qkv_multiplier"] = qkv_ratio
                print(f"[🛡️] Capped QKV pruning to {MAX_SAFE_ATTN:.3f}")


            E = model.embed_dim
            internal_dim = int(E * (1 - isomorphic_group_ratios.get('qkv_multiplier', 0.5)))
            num_heads = getattr(model.blocks[0].attn, 'num_heads', 12)
            final_head_dim = internal_dim // num_heads

            if final_head_dim < MIN_HEAD_DIM:
                print(f"[❌] ABORT: Final head_dim={final_head_dim} < {MIN_HEAD_DIM} is catastrophic!")
                print(f"[💡] LLM suggested too aggressive attention pruning")

                # Force safe head dimension
                safe_internal_dim = MIN_HEAD_DIM * num_heads
                safe_qkv_ratio = 1 - (safe_internal_dim / E)
                safe_qkv_ratio = round(safe_qkv_ratio, 3)
                isomorphic_group_ratios['qkv_multiplier'] = safe_qkv_ratio
                print(f"[🛡️] Forced QKV to {safe_qkv_ratio:.3f} to maintain head_dim >= {MIN_HEAD_DIM}")

            # ---- 3️⃣ Enforce minimum MLP hidden dim (redundant protection) ----
            mlp_hidden = int(model.blocks[0].mlp.fc1.in_features * (1 - mlp_ratio))
            if mlp_hidden < MIN_MLP_DIM:
                new_prune = 1 - (MIN_MLP_DIM / model.blocks[0].mlp.fc1.in_features)
                new_prune = round(new_prune, 3)
                print(f"[⚠️] Adjusted mlp_multiplier to avoid collapse: using {new_prune:.3f} (hidden_dim = {MIN_MLP_DIM})")
                isomorphic_group_ratios["mlp_multiplier"] = new_prune


            groups = analyzer.create_isomorphic_groups(target_macs, isomorphic_group_ratios, base_macs)

            # INSERT AFTER LINE 811
            # print(f"\n[🔍] DEBUG 2: cls_token after create_isomorphic_groups")
            # if hasattr(model, 'cls_token') and model.cls_token is not None:
            #     print(f"  Shape: {model.cls_token.shape}, Mean: {model.cls_token.mean():.6f}, Std: {model.cls_token.std():.6f}")
            # else:
            #     print(f"  cls_token is None!")

            # ✂️ 2. Execute dependency-aware pruning
            pruning_result = self._execute_dependency_aware_pruning(model, groups)

            # INSERT AFTER LINE 814 (but check if pruning was successful first)
            # if pruning_result['success']:
            #     print(f"\n[🔍] DEBUG 3: cls_token after _execute_dependency_aware_pruning")
            #     if hasattr(model, 'cls_token') and model.cls_token is not None:
            #         print(f"  Shape: {model.cls_token.shape}, Mean: {model.cls_token.mean():.6f}, Std: {model.cls_token.std():.6f}")
            #     else:
            #         print(f"  cls_token is None!")

            if not pruning_result['success']:
                print(f"[❌] Pruning failed: {pruning_result.get('error', 'Unknown error')}")
                return {**state, 'error': f"Pruning failed: {pruning_result.get('error', 'Unknown error')}"}

            achieved_ratio = pruning_result['actual_ratio']
            print(f"[✅] Pruning successful. Achieved parameter reduction: {achieved_ratio*100:.2f}%")

            # print(f"\n{'='*80}")
            # print(f"🔍 DEBUG CHECKPOINT 1: Right after pruning")
            # print(f"  groups['attention_blocks'].pruning_ratio = {groups['attention_blocks'].pruning_ratio}")
            # print(f"  groups['mlp_blocks'].pruning_ratio = {groups['mlp_blocks'].pruning_ratio}")
            # print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
            final_macs_check1, _ = tp.utils.count_ops_and_params(model, example_inputs)
            # print(f"  Model MACs: {final_macs_check1/1e9:.3f}G")
            # print(f"{'='*80}\n")

            # 🩹 3. Patch missing norms if any
            for name, module in model.named_modules():
                is_timm_attn = isinstance(module, getattr(timm.models.vision_transformer, "Attention", tuple()))
                is_duck_attn = hasattr(module, "qkv") and hasattr(module, "proj")
                if is_timm_attn or is_duck_attn:
                    if not hasattr(module, "norm"):
                        module.norm = nn.Identity()
                        # print(f"[🔧] Post-pass: Added Identity norm to {name}")

            # print(f"\n{'='*80}")
            # print(f"🔍 DEBUG CHECKPOINT 2: After norm patching, before evaluation")
            # print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
            checkpoint2_macs, _ = tp.utils.count_ops_and_params(model, example_inputs)
            # print(f"  Model MACs: {checkpoint2_macs/1e9:.3f}G")

            
            if hasattr(model, 'blocks') and len(model.blocks) > 0:
                attn = model.blocks[0].attn
                # print(f"  Block 0 attention:")
                # print(f"    qkv.out_features: {attn.qkv.out_features}")
                # print(f"    proj.in_features: {attn.proj.in_features}, proj.out_features: {attn.proj.out_features}")
                # print(f"    num_heads: {attn.num_heads}, head_dim: {attn.head_dim}")
                expected = 3 * attn.num_heads * attn.head_dim
                # print(f"    Expected qkv.out: {expected}, Match: {attn.qkv.out_features == expected}")
            # print(f"{'='*80}\n")

            # ==============================================

            # ✅ [PATCH] Reattach classification heads and tokens before zero-shot evaluation
            from eval_agent import reattach_heads_and_tokens

            original_model = state.get("original_model", None)
            if original_model is not None:
                try:
                    model = reattach_heads_and_tokens(model, original_model)
                    print("[🔗] Successfully reattached heads, cls/dist tokens, and pos_embed before zero-shot eval.")
                except Exception as e:
                    print(f"[⚠️] Warning: Failed to reattach heads/tokens before zero-shot eval: {e}")
            else:
                print("[⚠️] No original_model found in state — skipping reattachment.")

            # INSERT AFTER LINE 872 (before evaluation)
            # print(f"\n[🔍] DEBUG 4: cls_token right before evaluation")
            # if hasattr(model, 'cls_token') and model.cls_token is not None:
                # print(f"  Shape: {model.cls_token.shape}, Mean: {model.cls_token.mean():.6f}, Std: {model.cls_token.std():.6f}")
            # else:
                # print(f"  cls_token is None!")
            # if hasattr(model, 'head'):
                # print(f"  head weights - Mean: {model.head.weight.mean():.6f}, Std: {model.head.weight.std():.6f}")
            # At DEBUG 4, add this line after the head weights check:
            # if hasattr(model, 'pos_embed') and model.pos_embed is not None:
                # print(f"  pos_embed - Shape: {model.pos_embed.shape}, Mean: {model.pos_embed.mean():.6f}, Std: {model.pos_embed.std():.6f}")
            
            # Right before "Testing forward pass manually", add:
            # print("\n[🔍] Checking data loader sample...")
            try:
                for batch_idx, batch in enumerate(val_loader):
                    if isinstance(batch, dict):
                        inputs = batch['pixel_values']
                        labels = batch['label']
                    else:
                        inputs, labels = batch
                    # print(f"  Batch input shape: {inputs.shape}")
                    # print(f"  Input mean: {inputs.mean():.6f}, std: {inputs.std():.6f}")
                    # print(f"  Input min: {inputs.min():.6f}, max: {inputs.max():.6f}")
                    # print(f"  Labels sample: {labels[:5].tolist()}")
                    break
            except Exception as e:
                print(f"  Failed to load batch: {e}")
            
            
            # Test forward pass manually
            # print("\n[🧪] Testing forward pass manually...")
            try:
                test_input = torch.randn(2, 3, 224, 224).to(next(model.parameters()).device)
                with torch.no_grad():
                    output = model(test_input)
                # print(f"  Forward pass works! Output shape: {output.shape}")
                # print(f"  Output mean: {output.mean():.6f}, std: {output.std():.6f}")
                # print(f"  Output max: {output.max():.6f}, min: {output.min():.6f}")
                
                # Check if outputs are reasonable
                probs = torch.softmax(output, dim=1)
                top5 = torch.topk(probs, 5, dim=1)
                # print(f"  Top-5 probabilities: {top5.values[0].tolist()}")
                # print(f"  Top-5 classes: {top5.indices[0].tolist()}")
            except Exception as e:
                # print(f"  Forward pass FAILED: {e}")
                import traceback
                traceback.print_exc()

            # 🧪 5. Evaluate pruned model (TRUE zero-shot)
            print(f"[🧪] Evaluating pruned model (TRUE zero-shot) on {dataset}...")
            evaluation_result = self._evaluate_model(model, val_loader, device, dataset)

            if dataset.lower() == 'imagenet':
                zero_shot_top1, zero_shot_top5 = evaluation_result
                zero_shot_accuracy = zero_shot_top1
                best_top5_accuracy = zero_shot_top5
                print(f"[📊] TRUE Zero-shot - Top-1: {zero_shot_top1:.2f}%, Top-5: {zero_shot_top5:.2f}%")
            else:
                zero_shot_accuracy = evaluation_result[0]
                zero_shot_top1 = zero_shot_accuracy
                zero_shot_top5 = None
                print(f"[📊] TRUE Zero-shot accuracy: {zero_shot_accuracy:.2f}%")

            if dataset.lower() == 'imagenet':
                log_to_wandb({
                    "zero_shot_top1_accuracy": zero_shot_top1,
                    "zero_shot_top5_accuracy": zero_shot_top5,
                }, step_name="pruning", dataset=dataset)

            # 💾 6. Store state for later use
            pruned_model_state = copy.deepcopy(model.state_dict())
            best_accuracy = zero_shot_accuracy
            best_model = model
            best_state_dict = model.state_dict()
            best_ratio = achieved_ratio


        except Exception as e:
            import traceback
            print(f"[❌] Error in pruning attempt: {e}")
            print(traceback.format_exc())
            
            # If pruning failed, report failure
            state["pruning_results"] = {
                "success": False,
                "error": f"Pruning failed with ratio {current_ratio:.4f}: {str(e)}",
                "achieved_ratio": 0.0,
                "zero_shot_accuracy": 0.0,
                "dataset": dataset
            }
            state['attempted_pruning_ratios'] = attempted_ratios
            return state

        # Pruning succeeded - save model and calculate metrics
        if not best_model:
            print(f"[❌] Pruning failed with ratio {current_ratio:.4f}")
            state["pruning_results"] = {
                "success": False,
                "error": f"Pruning failed with ratio {current_ratio:.4f}",
                "achieved_ratio": 0.0,
                "zero_shot_accuracy": 0.0,
                "dataset": dataset
            }
            state['attempted_pruning_ratios'] = attempted_ratios
            return state

        # Save pruned model checkpoint
        checkpoint_dir = state.get('checkpoint_dir', './checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_filename = f"pruned_{model_name}_{dataset}_rev{state.get('revision_number', 0)}_ratio{best_ratio:.3f}.pth"
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)

        torch.save({
            'complete_model': best_model,
            'state_dict': best_state_dict,
            'pruning_ratio': best_ratio,
            'dataset': dataset,
            'num_classes': num_classes,
            'model_name': model_name,
            'zero_shot_accuracy': zero_shot_accuracy,
            'revision_number': state.get('revision_number', 0),
            'job_id': os.environ.get('SLURM_JOB_ID', 'local'),
            'timestamp': datetime.now().isoformat()
        }, checkpoint_path)

        print(f"[💾] Saved pruned checkpoint: {checkpoint_path}")

        # Calculate MACs reduction using already established values
        print(f"[📊] Calculating MACs reduction...")
        final_macs, _ = tp.utils.count_ops_and_params(best_model, example_inputs)
        macs_reduction = float((base_macs - final_macs) / base_macs)


        state['target_macs'] = float(target_macs)

        macs_error_pct = (
            100.0 * (float(final_macs) - float(target_macs)) / max(float(target_macs), 1e-9)
            if target_macs is not None else None
            )


        log_payload = {
            "achieved_ratio": best_ratio,
            "achieved_ratio_pct": best_ratio * 100,
            "macs_reduction": macs_reduction,
            "macs_reduction_pct": macs_reduction * 100,
            "zero_shot_accuracy": zero_shot_accuracy,
            "parameters_before": base_params,
            "parameters_after": sum(p.numel() for p in best_model.parameters()),
            "revision_number": state.get('revision_number', 0),
            "baseline_macs": float(base_macs),
            "achieved_macs": float(final_macs),
        }
        if target_macs is not None:
            log_payload["target_macs"] = float(target_macs)
            log_payload["macs_error_pct"] = float(macs_error_pct)

        log_to_wandb(log_payload, step_name="pruning", dataset=dataset)



        pruning_results = {
            'success': True,

            # --- MACs-first (NEW) ---
            'baseline_macs': float(base_macs),
            'target_macs': float(target_macs) if target_macs is not None else None,
            'achieved_macs': float(final_macs),
            'macs_reduction': float(macs_reduction),
            'macs_error_pct': float(macs_error_pct) if macs_error_pct is not None else None,

            # --- Legacy (kept for back-compat) ---
            'achieved_ratio': float(best_ratio),

            # --- Existing fields unchanged ---
            'checkpoint_path': checkpoint_path,
            'dataset': dataset,
            'num_classes': num_classes,
            'pruned_model_state': pruned_model_state,
        }


        # Add TRUE zero-shot metrics based on dataset
        if dataset.lower() == 'imagenet':
            pruning_results.update({
                'zero_shot_top1_accuracy': float(zero_shot_top1),
                'zero_shot_top5_accuracy': float(zero_shot_top5),
                'zero_shot_accuracy': float(zero_shot_top1)  # For compatibility
            })
        else:
            pruning_results['zero_shot_accuracy'] = float(zero_shot_accuracy)


        # ✅ CRITICAL FIX: Update history with TRUE zero-shot results
        strategy_dict = analysis_results.get("strategy_dict", {})

        # Create strategy_used with all the existing fields
        strategy_used = {
            'importance_criterion': importance_type,
            'round_to': round_to_value,
            'global_pruning': global_flag,
            'pruning_ratio': current_ratio,
            'pruning_ratio_requested': current_ratio,
            'rationale': rationale if rationale else "No rationale provided"
        }

        # print(f"\n{'='*80}")
        # print(f"🔍 DEBUG CHECKPOINT 3: Before storing in history")
        # print(f"  groups['attention_blocks'].pruning_ratio = {groups['attention_blocks'].pruning_ratio}")
        # print(f"  groups['mlp_blocks'].pruning_ratio = {groups['mlp_blocks'].pruning_ratio}")
        # print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        checkpoint3_macs, _ = tp.utils.count_ops_and_params(model, example_inputs)
        # print(f"  Model MACs: {checkpoint3_macs/1e9:.3f}G")
        # print(f"  strategy_dict from analysis_results: {strategy_dict.get('isomorphic_group_ratios', 'NOT FOUND')}")
        # print(f"{'='*80}\n")

        # ✅ CRITICAL FIX: Store ACTUAL PRUNING RATIOS, not retention rates!
        # This ensures the LLM learns the correct relationship between multipliers and MACs
        strategy_used['isomorphic_group_ratios'] = {
            'qkv_multiplier': float(groups['attention_blocks'].pruning_ratio),  # Actual pruning ratio applied
            'mlp_multiplier': float(groups['mlp_blocks'].pruning_ratio),        # Actual pruning ratio applied
            'proj_multiplier': 0.0,
            'head_multiplier': 0.0
        }
        print(f"[📝] Storing APPLIED pruning ratios in history: {strategy_used['isomorphic_group_ratios']}")

        # ✅ ADD: Copy channel_pruning_ratio if it exists (for CNN models)
        if 'channel_pruning_ratio' in strategy_dict:
            strategy_used['channel_pruning_ratio'] = strategy_dict['channel_pruning_ratio']
        
        applied_qkv = float(strategy_used['isomorphic_group_ratios'].get('qkv_multiplier', 0.0))

        history_entry = {
            'revision': state.get('revision_number', 0),
            'target_ratio': target_ratio,
            'achieved_ratio': best_ratio,    
            'target_macs': target_macs,
            'achieved_macs': final_macs,
            'baseline_macs': base_macs,
            'macs_error_pct': (float(final_macs) - float(target_macs)) / max(float(target_macs), 1e-9) * 100.0,
            'macs_reduction': macs_reduction,
            'dataset': dataset,
            'strategy_used': strategy_used,
            'qkv_multiplier': applied_qkv,
        }


        # Add TRUE zero-shot results to history (these should NEVER be overwritten)
        if dataset.lower() == 'imagenet':
            history_entry.update({
                'zero_shot_top1_accuracy': float(zero_shot_top1),
                'zero_shot_top5_accuracy': float(zero_shot_top5),
                'zero_shot_accuracy': float(zero_shot_top1)  # For compatibility
            })
        else:
            history_entry['zero_shot_accuracy'] = float(zero_shot_accuracy)

        # Add to history
        state['history'] = state.get('history', [])
        state['history'].append(history_entry)

        # Print comprehensive results summary
        print(f"\n[📊] {dataset.upper()} Pruning Results Summary:")
        print(f"Parameters reduction: {best_ratio*100:.2f}%")
        print(f"MACs reduction: {macs_reduction*100:.2f}%")
        
        if dataset.lower() == 'imagenet':
            print(f"TRUE Zero-shot Top-1 accuracy: {zero_shot_top1:.2f}%")
            if zero_shot_top5 is not None:
                print(f"TRUE Zero-shot Top-5 accuracy: {zero_shot_top5:.2f}%")
        else:
            print(f"TRUE Zero-shot accuracy: {zero_shot_accuracy:.2f}%")
        
        # print(f"Model saved to: {checkpoint_path}")

        # Update attempted ratios in state
        state['attempted_pruning_ratios'] = attempted_ratios

        return {
            **state,
            'pruning_results': pruning_results,
            'prune': {
                'model': best_model,
                'pruning_results': pruning_results
            },
            'revision_number': state.get('revision_number', 0) + 1,
            'model_name': model_name
        }
    
    async def _execute_cnn_pruning(self, state: Dict, model_name: str) -> Dict:
        """CNN pruning with Analysis Agent learning (no mathematical assumptions)"""
        
        dataset = state.get("dataset", "cifar10")
        num_classes = state.get("num_classes", 10)
        input_size = state.get("input_size", 224)
        data_path = state.get("data_path", "./data")
        target_ratio = state.get("target_pruning_ratio")
        
        print(f"[🔧] Enhanced CNN Pruning: {model_name} on {dataset}")
        if target_ratio is not None:
            print(f"[🎯] Target PARAMETER reduction: {target_ratio*100:.1f}%")
        else:
            print(f"[🎯] MAC-first mode: Target parameter reduction will be derived from MAC constraints")
        
        # Get analysis results from Analysis Agent (with historical learning)
        analysis_results = state.get("analysis_results", {})
        strategy_dict = analysis_results.get("strategy_dict", {})
        
        # Extract ALL parameters from Analysis Agent
        channel_pruning_ratio = strategy_dict.get('channel_pruning_ratio')
        suggested_round_to = strategy_dict.get('round_to', 8)
        importance_criterion = strategy_dict.get('importance_criterion', 'taylor')
        
        # LEARNING-FIRST APPROACH: Use Analysis Agent or conservative fallback
        if channel_pruning_ratio is not None:
            print(f"[✅] Using Analysis Agent's learned channel ratio: {channel_pruning_ratio:.4f}")
            # print(f"[✅] This includes historical learning corrections")
        else:
            # Simple conservative fallback - let Analysis Agent learn the real relationship
            if target_ratio is not None:
                channel_pruning_ratio = target_ratio * 0.8  # Conservative starting point
                print(f"[⚠️] No learned ratio - using conservative starting point: {channel_pruning_ratio:.4f}")
            else:
                # MAC-first mode fallback
                channel_pruning_ratio = 0.3  # Conservative default for MAC-first mode
                print(f"[⚠️] MAC-first mode - using conservative fallback: {channel_pruning_ratio:.4f}")
            print(f"[🧠] Analysis Agent will learn actual {model_name} relationship from results")        
        print(f"[🧮] Using Analysis Agent parameters:")
        print(f"   Channel ratio: {channel_pruning_ratio:.4f}")
        print(f"   Round_to: {suggested_round_to}")
        print(f"   Importance: {importance_criterion}")

        try:
            device = self.get_device()
            model = self._prepare_model(model_name, device, dataset, num_classes)
            original_params = sum(p.numel() for p in model.parameters())
            
            print(f"[📊] Base CNN parameters: {original_params:,}")
            
            # Setup data loader for importance calculation
            batch_size = 64 if dataset.lower() == 'imagenet' else 64
            train_loader, val_loader = self._setup_dataset_loader(dataset, data_path, batch_size)
            criterion = nn.CrossEntropyLoss().to(device)
            
            # RESPECT Analysis Agent's importance choice (no override!)
            if importance_criterion == 'taylor':
                imp = self._calculate_importance(model, train_loader, criterion, device, 'taylor')
                print(f"[🧮] Using Taylor importance (Analysis Agent choice)")
            elif importance_criterion == 'l1norm':
                imp = tp.importance.MagnitudeImportance(p=1)
                print(f"[🧮] Using L1 importance (Analysis Agent choice)")
            elif importance_criterion == 'l2norm':
                imp = tp.importance.MagnitudeImportance(p=2)
                print(f"[🧮] Using L2 importance (Analysis Agent choice)")
            elif importance_criterion == 'random':
                imp = tp.importance.RandomImportance()
                print(f"[🧮] Using Random importance (Analysis Agent choice)")
            elif importance_criterion == 'magnitude':
                imp = tp.importance.MagnitudeImportance(p=1)
                print(f"[🧮] Using Magnitude importance (Analysis Agent choice)")
            else:
                # Fallback - but warn about unknown criterion
                print(f"[⚠️] Unknown importance criterion: {importance_criterion}, falling back to L1")
                imp = tp.importance.MagnitudeImportance(p=1)
                print(f"[🧮] Using L1 importance (fallback)")
            
            # Setup ignored layers (preserve classifier and first conv)
            ignored_layers = []
            for name, m in model.named_modules():
                if isinstance(m, nn.Linear) and m.out_features == num_classes:
                    ignored_layers.append(m)
                    print(f"[🔒] Preserving classifier: {name}")
                elif name == 'conv1' and isinstance(m, nn.Conv2d):
                    ignored_layers.append(m)
                    print(f"[🔒] Preserving first conv: {name}")
            
            # Example inputs
            example_inputs = (torch.randn(1, 3, input_size, input_size).to(device),)
            
            print(f"[🔧] Creating MetaPruner with Analysis Agent ratio: {channel_pruning_ratio:.4f}")
            
            pruner = tp.pruner.MetaPruner(
                model,
                example_inputs=example_inputs,
                global_pruning=True,
                importance=imp,
                isomorphic=False,
                pruning_ratio=channel_pruning_ratio,
                ignored_layers=ignored_layers,
                num_heads={},
                prune_head_dims=False,
                prune_num_heads=False,
                customized_pruners=pbench.extension.EXTENDED_PRUNERS,
                round_to=suggested_round_to,
            )
            
            # Execute pruning
            print(f"[✂️] Executing CNN structured pruning...")
            for i, g in enumerate(pruner.step(interactive=True)):
                # print(f"[✂️] Pruning group {i+1}")
                g.prune()
            
            # Calculate results
            final_params = sum(p.numel() for p in model.parameters())
            achieved_ratio = 1 - (final_params / original_params)
            
            print(f"[📊] CNN Pruning Results:")
            print(f"  - Original: {original_params:,} params")
            print(f"  - Final: {final_params:,} params")
            
            print(f"  - Achieved reduction: {achieved_ratio*100:.2f}%")

            if target_ratio is not None:
                print(f"  - Target reduction: {target_ratio*100:.2f}%")
                print(f"  - Difference: {abs(achieved_ratio - target_ratio)*100:.2f}%")
            
                # Success check
                tolerance = 0.02
                ratio_diff = achieved_ratio - target_ratio
                success = (0 <= ratio_diff <= tolerance) or (-0.01 <= ratio_diff < 0)
                print(f"  - Meets target (≤{tolerance*100:.0f}% overshoot): {success}")
            else:
                # MAC-first mode - success determined by MAC constraints elsewhere
                print(f"  - MAC-first mode: Target reduction derived from MAC constraints")
                print(f"  - Achieved reduction: {achieved_ratio*100:.2f}%")
                success = True  # Will be determined by MAC tolerance in routing logic
            
            # Evaluate and log accuracy (but don't make decisions based on it)
            evaluation_result = self._evaluate_model(model, val_loader, device, dataset)


            if dataset.lower() == 'imagenet':
                zero_shot_top1, zero_shot_top5 = evaluation_result
                zero_shot_accuracy = zero_shot_top1
                print(f"[📊] Zero-shot - Top-1: {zero_shot_top1:.2f}%, Top-5: {zero_shot_top5:.2f}%")
            else:
                zero_shot_accuracy = evaluation_result[0]
                zero_shot_top5 = None
                print(f"[📊] Zero-shot accuracy: {zero_shot_accuracy:.2f}%")
            
            # Calculate MACs
            base_model = self._prepare_model(model_name, device, dataset, num_classes)
            base_macs, _ = tp.utils.count_ops_and_params(base_model, example_inputs)
            final_macs, _ = tp.utils.count_ops_and_params(model, example_inputs)
            macs_reduction = (base_macs - final_macs) / base_macs
            
            # Save checkpoint
            checkpoint_path = f"pruned_{model_name}_{dataset}_cnn_learning.pth"
            torch.save({
                'complete_model': model,
                'state_dict': model.state_dict(),
                'pruning_ratio': achieved_ratio,
                'dataset': dataset,
                'model_name': model_name,
                'approach': 'cnn_learning_based',
                'channel_ratio_used': channel_pruning_ratio,
                'target_param_reduction': target_ratio
            }, checkpoint_path)
            
            # MACs-first + keep ratio fields for back-compat
            pruning_results = {
                'success': True,

                'baseline_macs': float(base_macs),  # absolute MACs of the unpruned model
                'target_macs': float(state.get('target_macs')) if state.get('target_macs') is not None else None,
                'achieved_macs': float(final_macs), # absolute MACs after pruning
                'macs_reduction': float(macs_reduction),  # fraction reduced (e.g., 0.35 = 35%)
                'macs_error_pct': (
                    (float(final_macs) - float(state.get('target_macs', 0))) / max(float(state.get('target_macs', 1)), 1e-9) * 100.0
                    if state.get('target_macs') is not None else None
                ),

                # --- Legacy ratio fields (kept so nothing else breaks) ---
                'achieved_ratio': float(achieved_ratio) if 'achieved_ratio' in locals() else None,
                'channel_ratio_used': channel_pruning_ratio,

                # --- Unchanged fields ---
                'checkpoint_path': checkpoint_path,
                'dataset': dataset,
                'num_classes': num_classes,
                'approach': 'cnn_learning_based',
                'learning_based': True
            }

            
            # Add accuracy results
            if dataset.lower() == 'imagenet':
                pruning_results.update({
                    'zero_shot_top1_accuracy': float(zero_shot_top1),
                    'zero_shot_top5_accuracy': float(zero_shot_top5),
                    'zero_shot_accuracy': float(zero_shot_top1)
                })
            else:
                pruning_results['zero_shot_accuracy'] = float(zero_shot_accuracy)
            
            # Create history entry
            history_entry = {
                'revision': state.get('revision_number', 0),
                'target_ratio': target_ratio,
                'achieved_ratio': achieved_ratio,
                'macs_reduction': macs_reduction,
                'dataset': dataset,
                'strategy_used': {
                    'importance_criterion': importance_criterion,
                    'approach': 'cnn_learning_based',
                    'isomorphic': False,
                    'round_to': suggested_round_to,
                    'channel_ratio_used': channel_pruning_ratio,
                    'learning_based': True
                }
            }
            
            # Add accuracy to history
            if dataset.lower() == 'imagenet':
                history_entry.update({
                    'zero_shot_top1_accuracy': float(zero_shot_top1),
                    'zero_shot_top5_accuracy': float(zero_shot_top5),
                    'zero_shot_accuracy': float(zero_shot_top1)
                })
            else:
                history_entry['zero_shot_accuracy'] = float(zero_shot_accuracy)
            
            # Update state
            state['history'] = state.get('history', [])
            state['history'].append(history_entry)
            
            return {
                **state,
                'pruning_results': pruning_results,
                'prune': {
                    'model': model,
                    'pruning_results': pruning_results
                },
                'revision_number': state.get('revision_number', 0) + 1,
                'model_name': model_name
            }
            
        except Exception as e:
            print(f"[❌] CNN pruning failed: {e}")
            import traceback
            print(traceback.format_exc())
            
            return {
                **state,
                'pruning_results': {
                    'success': False,
                    'error': str(e),
                    'achieved_ratio': 0.0,
                    'dataset': dataset
                }
            }

    def _update_global_embeddings(self, new_embed_dim: int):
        """Update all global embeddings to match pruned embed_dim"""
        with torch.no_grad():
            # Get the scattered indices to use for pruning
            scatter_indices = getattr(self, '_global_embed_scatter_indices', None)
            
            if scatter_indices is None:
                # Fallback: use contiguous if no scattered indices available
                print("[⚠️] No scattered indices found, using contiguous")
                scatter_indices = torch.arange(new_embed_dim)
            
            # Update positional embeddings
            if hasattr(self.model, 'pos_embed') and self.model.pos_embed is not None:
                old_shape = self.model.pos_embed.shape
                if old_shape[-1] != new_embed_dim:
                    # Use scattered indices to select important dimensions
                    self.model.pos_embed = nn.Parameter(
                        self.model.pos_embed[:, :, scatter_indices]
                    )
                    print(f"[🔧] pos_embed: {old_shape[-1]} -> {new_embed_dim}")
            
            # Update class token
            if hasattr(self.model, 'cls_token') and self.model.cls_token is not None:
                old_shape = self.model.cls_token.shape
                if old_shape[-1] != new_embed_dim:
                    self.model.cls_token = nn.Parameter(
                        self.model.cls_token[:, :, scatter_indices]
                    )
                    print(f"[🔧] cls_token: {old_shape[-1]} -> {new_embed_dim}")
            
            # Update distillation token (DeiT specific)
            if hasattr(self.model, 'dist_token') and self.model.dist_token is not None:
                old_shape = self.model.dist_token.shape
                if old_shape[-1] != new_embed_dim:
                    self.model.dist_token = nn.Parameter(
                        self.model.dist_token[:, :, scatter_indices]
                    )
                    print(f"[🔧] dist_token: {old_shape[-1]} -> {new_embed_dim}")

            # Update patch embedding projection
            if hasattr(self.model, 'patch_embed') and hasattr(self.model.patch_embed, 'proj'):
                proj_layer = self.model.patch_embed.proj
                if isinstance(proj_layer, nn.Conv2d):
                    old_out_channels = proj_layer.out_channels
                    if old_out_channels != new_embed_dim:
                        # Use scattered indices
                        new_weight = proj_layer.weight[scatter_indices, :, :, :]
                        proj_layer.weight = nn.Parameter(new_weight)
                        
                        if proj_layer.bias is not None:
                            new_bias = proj_layer.bias[scatter_indices]
                            proj_layer.bias = nn.Parameter(new_bias)
                        
                        proj_layer.out_channels = new_embed_dim
                        print(f"[🔧] patch_embed.proj: {old_out_channels} -> {new_embed_dim}")

            # Update classifier heads (use scattered indices)
            for head_name in ['head', 'head_dist', 'fc']:
                if hasattr(self.model, head_name):
                    head = getattr(self.model, head_name)
                    if isinstance(head, nn.Linear):
                        old_in_features = head.in_features
                        if old_in_features != new_embed_dim:
                            new_weight = head.weight[:, scatter_indices]
                            head.weight = nn.Parameter(new_weight)
                            head.in_features = new_embed_dim
                            print(f"[🔧] {head_name}: {old_in_features} -> {new_embed_dim}")


    def _update_layer_norms(self, new_embed_dim: int):
        """Update all LayerNorm layers to match pruned embed_dim"""
        with torch.no_grad():
            # ✅ CRITICAL FIX: Use the SAME scattered indices from global embedding pruning
            scatter_indices = getattr(self, '_global_embed_scatter_indices', None)
            
            if scatter_indices is None:
                # Fallback: use importance-based selection
                print("[⚠️] No scattered indices found for LayerNorm, using importance")
            
            for name, module in self.model.named_modules():
                if isinstance(module, nn.LayerNorm):
                    old_normalized_shape = module.normalized_shape[0] if isinstance(module.normalized_shape, tuple) else module.normalized_shape
                    
                    if old_normalized_shape != new_embed_dim:
                        # Create new LayerNorm with correct dimensions
                        new_layer_norm = nn.LayerNorm(new_embed_dim, 
                                                    eps=module.eps, 
                                                    elementwise_affine=module.elementwise_affine)
                        
                        # Transfer pruned weights and biases if they exist
                        if module.weight is not None:
                            if scatter_indices is not None:
                                # ✅ Use the SAME scattered indices as global embeddings
                                new_layer_norm.weight.data = module.weight.data[scatter_indices]
                                if module.bias is not None:
                                    new_layer_norm.bias.data = module.bias.data[scatter_indices]
                            else:
                                # Fallback: compute importance
                                importance = torch.abs(module.weight)
                                _, keep_indices = torch.topk(importance, new_embed_dim, largest=True)
                                keep_indices = torch.sort(keep_indices)[0]
                                
                                new_layer_norm.weight.data = module.weight.data[keep_indices]
                                if module.bias is not None:
                                    new_layer_norm.bias.data = module.bias.data[keep_indices]
                        
                        # Replace the module in the model
                        parent_name = '.'.join(name.split('.')[:-1])
                        child_name = name.split('.')[-1]

                        if parent_name:
                            parent = dict(self.model.named_modules())[parent_name]
                            setattr(parent, child_name, new_layer_norm.to(module.weight.device))
                        else:
                            # Handle top-level modules (like 'norm')
                            setattr(self.model, child_name, new_layer_norm.to(module.weight.device))

                        print(f"[🔧] {name}: LayerNorm {old_normalized_shape} -> {new_embed_dim}")


    def _execute_dependency_aware_pruning(self, model: nn.Module, groups: Dict[str, IsomorphicGroup]) -> Dict:
        """Execute pruning while respecting dependencies, with a forced safe order:
        1) attention_blocks  -> updates global emb/LN once (basis fixed)
        2) mlp_blocks        -> prunes with the same basis
        3) output_projections (optional / skipped)
        """
        # Keep a reference for helper methods (global emb/LN updates use self.model)
        self.model = model
        original_params = sum(p.numel() for p in model.parameters())

        try:
            # Helper to prune a single group if present & active
            def _prune_group(group_key: str, handler):
                if group_key not in groups:
                    return
                group = groups[group_key]
                if not group.layers:
                    return
                pruning_ratio = group.pruning_ratio
                if pruning_ratio <= 0.0:
                    return

                print(f"[✂️] Pruning {group.name} with ratio {pruning_ratio:.3f} ({len(group.layers)} couples)")
                print(f"    MAC count: {group.mac_count/1e9:.3f}G, Target MAC: {group.target_macs/1e9:.3f}G")
                handler(group)

            # ── 1) ATTENTION FIRST ──────────────────────────────────────────────
            # This computes/reuses ONE global keep set and immediately updates:
            #   - pos_embed / cls_token / dist_token / patch_embed.proj
            #   - ALL LayerNorms
            # which fixes the representation basis for the whole network.
            _prune_group("attention_blocks", self._prune_attention_couples)

            # ── 2) THEN MLP ─────────────────────────────────────────────────────
            # Now MLP couples use the SAME global embedding basis, so fc1/fc2
            # residual alignments are preserved.
            _prune_group("mlp_blocks", self._prune_mlp_couples)

            self._patch_timm_attention_forward(model)

            # ── 3) (Optional) output projections ────────────────────────────────
            # If you have a handler, call it; otherwise we intentionally skip.
            # _prune_group("output_projections", self._prune_output_projections)  # if implemented

            # Validate model integrity after all pruning
            self._validate_model_integrity(model)

            # Clean up global indices to avoid accidental reuse later
            if hasattr(self, "_global_embed_keep_indices"):
                delattr(self, "_global_embed_keep_indices")
            if hasattr(self, "_global_embed_scatter_indices"):
                delattr(self, "_global_embed_scatter_indices")

            final_params = sum(p.numel() for p in model.parameters())
            actual_ratio = 1 - (final_params / original_params)
            print(f"[📊] Parameter reduction: {original_params:,} → {final_params:,} ({actual_ratio:.3f})")

            return {
                "success": True,
                "approach": "dependency_aware_isomorphic",
                "original_params": original_params,
                "final_params": final_params,
                "actual_ratio": actual_ratio,
            }

        except Exception as e:
            print(f"[❌] Dependency-aware pruning failed: {e}")
            import traceback
            print(traceback.format_exc())
            return {"success": False, "error": str(e)}

        finally:
            if hasattr(self, "model"):
                delattr(self, "model")



    def _prune_mlp_couples(self, group: IsomorphicGroup):
        """Prune MLP couples while maintaining fc1.out == fc2.in"""
        
        # Track counts and dimensions before the loop
        self._mlp_prune_count = 0
        original_hidden = None
        target_hidden = None
        
        for mlp_couple in group.layers:
            if not isinstance(mlp_couple, MLPCouple):
                continue
                
            # Validate coupling before pruning
            if not mlp_couple.validate_coupling():
                print(f"[⚠️] MLP couple invalid: {mlp_couple.fc1_name}")
                continue
            
            original_hidden = mlp_couple.get_hidden_dim()
            target_hidden = max(1, int(original_hidden * (1 - group.pruning_ratio)))
            
            # Count successful prunings
            self._mlp_prune_count += 1
            
            # Compute joint importance for both layers
            importance = self._compute_mlp_couple_importance(mlp_couple)
            
            # Select channels to keep
            _, keep_indices = torch.topk(importance, target_hidden, largest=True)
            keep_indices = torch.sort(keep_indices)[0]
            
            # Update both layers atomically
            self._update_mlp_couple(mlp_couple, keep_indices)
            
            # Validate after update
            if not mlp_couple.validate_coupling():
                raise RuntimeError(f"MLP couple broken after pruning: {mlp_couple.fc1_name}")
        
        # Print summary ONCE after all pruning is complete
        if self._mlp_prune_count > 0:
            print(f"[🔧] Pruning {self._mlp_prune_count} MLP layers: {original_hidden} -> {target_hidden} channels each")


    def _compute_mlp_couple_importance(self, mlp_couple: MLPCouple):
        """Compute joint importance for fc1+fc2 pair"""
        with torch.no_grad():
            # fc1 importance: norm of each output channel
            fc1_importance = torch.norm(mlp_couple.fc1.weight, dim=1)
            
            # fc2 importance: norm of each input channel  
            fc2_importance = torch.norm(mlp_couple.fc2.weight, dim=0)
            
            # Joint importance: channels that are important for BOTH layers
            joint_importance = fc1_importance * fc2_importance
            
            return joint_importance

    def _update_mlp_couple(self, mlp_couple: MLPCouple, keep_indices: torch.Tensor):
        """Prune MLP hidden width only: fc1.out & fc2.in. Keep embedding E fixed."""
        with torch.no_grad():
            # fc1: prune outputs only
            mlp_couple.fc1.weight = nn.Parameter(mlp_couple.fc1.weight[keep_indices, :])
            if mlp_couple.fc1.bias is not None:
                mlp_couple.fc1.bias = nn.Parameter(mlp_couple.fc1.bias[keep_indices])
            mlp_couple.fc1.out_features = len(keep_indices)
            # fc1.in_features stays as E

            # fc2: prune inputs only
            mlp_couple.fc2.weight = nn.Parameter(mlp_couple.fc2.weight[:, keep_indices])
            mlp_couple.fc2.in_features = len(keep_indices)
            # fc2.out_features stays as E


    def _patch_timm_attention_forward(self, model):
        """Patch timm's Attention.forward() to handle pruned dimensions"""
        from typing import Optional
        import torch
        
        def create_patched_forward(num_heads, head_dim):
            def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
                B, N, C = x.shape
                qkv = self.qkv(x)
                internal_dim = num_heads * head_dim
                qkv = qkv.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
                q, k, v = qkv.unbind(0)
                
                # COMPLETE attention computation (replaces placeholder!)
                if hasattr(self, 'fused_attn') and self.fused_attn:
                    x = torch.nn.functional.scaled_dot_product_attention(
                        q, k, v, attn_mask=attn_mask,
                        dropout_p=self.attn_drop.p if self.training else 0.)
                else:
                    q = q * self.scale
                    attn = q @ k.transpose(-2, -1)
                    if attn_mask is not None:
                        attn = attn + attn_mask
                    attn = attn.softmax(dim=-1)
                    attn = self.attn_drop(attn)
                    x = attn @ v  # This redefines x!
                
                x = x.transpose(1, 2).reshape(B, N, internal_dim)
                x = self.proj(x)
                x = self.proj_drop(x)
                return x
            return forward
        
        patched_count = 0
        for name, module in model.named_modules():
            if hasattr(module, 'qkv') and hasattr(module, 'proj'):
                if hasattr(module, 'num_heads') and hasattr(module, 'head_dim'):
                    module.forward = create_patched_forward(
                        module.num_heads, module.head_dim
                    ).__get__(module, type(module))
                    patched_count += 1
        
        if patched_count > 0:
            print(f"[✅] Patched {patched_count} attention modules")
        
        return patched_count

    def _prune_attention_couples(self, group: IsomorphicGroup):
        """
        Attention-first pruning WITHOUT shrinking the global embedding:
        • Derive a single target k (head-compatible) from the first valid couple.
        • Apply it by pruning Q/K/V OUTPUT ROWS and PROJ INPUT COLUMNS only.
        • Do NOT update global embeddings / LNs here.
        """
        print(f"[🔧] Pruning {len(group.layers)} attention couples")
        if not group.layers:
            return

        # Find a valid attention couple
        first_couple = None
        for ac in group.layers:
            if isinstance(ac, AttentionCouple) and ac.validate_coupling():
                first_couple = ac
                break
        if first_couple is None:
            print("[⚠️] No valid attention couple found; skipping attention pruning.")
            return

        # Derive keep size once (head-compatible)
        E = first_couple.get_embed_dim()
        k = max(1, int(E * (1 - group.pruning_ratio)))
        k = self._adjust_embed_dim_for_heads(first_couple, k)
        self._last_k_target = k

        # Apply SAME policy to every attention couple (no global embed shrink!)
        self._attention_update_count = 0
        for attn_couple in group.layers:
            if not isinstance(attn_couple, AttentionCouple):
                continue
            if not attn_couple.validate_coupling():
                print(f"[⚠️] Attention couple invalid: {attn_couple.qkv_name}")
                continue

            self._update_attention_couple(attn_couple, None)  # keep set computed inside

            if not attn_couple.validate_coupling():
                raise RuntimeError(f"Attention couple broken after pruning: {attn_couple.qkv_name}")

        if self._attention_update_count > 0:
            print(
                f"[🔧] Pruned {self._attention_update_count} attention layers: "
                f"qkv rows 3*{E}→3*{self._last_k_target}, proj cols {E}→{self._last_k_target} "
                f"(attention internal dim = {self._last_k_target}, output stays {E})"
            )

    def _adjust_embed_dim_for_heads(self, attn_couple: AttentionCouple, target_embed: int):
        """Adjust target embedding dimension to be compatible with attention heads"""
        
        # Try to find the attention module that contains this QKV layer
        for name, module in self.model.named_modules():
            if hasattr(module, 'qkv') and module.qkv is attn_couple.qkv:
                if hasattr(module, 'num_heads'):
                    num_heads = module.num_heads
                    # Ensure target_embed is divisible by num_heads
                    adjusted_embed = (target_embed // num_heads) * num_heads
                    if adjusted_embed < num_heads:
                        adjusted_embed = num_heads  # At least one dimension per head
                    
                    if adjusted_embed != target_embed:
                        # Track adjustments instead of individual prints
                        if not hasattr(self, '_embed_adjustment_count'):
                            self._embed_adjustment_count = 0
                            self._embed_adjustment_details = (target_embed, adjusted_embed, num_heads)
                        self._embed_adjustment_count += 1
                    
                    return adjusted_embed
        
        # If no attention module found, return original target
        return target_embed


    def _compute_attention_couple_row_importance(self, attn_couple: AttentionCouple):
        """
        Return per-row importance for Q,K,V but structured per-head:
        we compute joint row scores (|row| for Q,K,V) and then reshape to (H, Dh).
        """
        with torch.no_grad():
            W = attn_couple.qkv.weight  # (3E, E)
            E = attn_couple.get_embed_dim()
            Wq = W[:E, :]
            Wk = W[E:2*E, :]
            Wv = W[2*E:, :]

            Iq = torch.norm(Wq, dim=1)   # (E,)
            Ik = torch.norm(Wk, dim=1)   # (E,)
            Iv = torch.norm(Wv, dim=1)   # (E,)
            I  = Iq * Ik * Iv            # (E,)

            # head-aware reshape
            # find the owning attention module to get num_heads/head_dim
            attn_mod = None
            for _, m in self.model.named_modules():
                if hasattr(m, "qkv") and m.qkv is attn_couple.qkv:
                    attn_mod = m
                    break
            if attn_mod is None or not hasattr(attn_mod, "num_heads"):
                # fallback: no struct info; return flat scores (not ideal)
                return I

            H = attn_mod.num_heads
            if E % H != 0:
                # should not happen in timm ViTs
                return I
            Dh = E // H
            return I.view(H, Dh).contiguous()   # (H, Dh)


    def _build_headaware_keep(self, scores_h_dh: torch.Tensor, k_total: int):
        """
        scores_h_dh: (H, Dh) row-importance per head/position.
        Choose the same number kh per head so total k = kh * H.
        Return sorted flat indices in [0..E-1] respecting per-head contiguity.
        """
        H, Dh = scores_h_dh.shape
        kh = max(1, int(round(k_total / H)))
        kh = min(kh, Dh)  # guard

        # pick top-kh positions *within each head*
        keep_per_head = []
        for h in range(H):
            _, idx = torch.topk(scores_h_dh[h], kh, largest=True)
            keep_per_head.append(torch.sort(idx)[0])  # ascending within head

        # flatten to global row indices with contiguous head blocks
        keep_global = []
        for h in range(H):
            base = h * Dh
            keep_global.append(keep_per_head[h] + base)
        keep = torch.cat(keep_global, dim=0)

        # final sort keeps head-block contiguity groups together
        keep, _ = torch.sort(keep)
        return keep



    def _compute_attention_couple_importance(self, attn_couple: AttentionCouple):
        # Backward-compat: old callers map to the new row-importance
        return self._compute_attention_couple_row_importance(attn_couple)

    def _update_attention_couple(self, attn_couple: AttentionCouple, _unused_keep_indices: torch.Tensor):
        with torch.no_grad():
            E = attn_couple.get_embed_dim()

            # get target k (already computed once per group)
            k_total = getattr(self, "_last_k_target", None)
            if k_total is None:
                raise RuntimeError("k_target not set; ensure _prune_attention_couples set self._last_k_target.")

            # head-aware scores and keep set
            scores = self._compute_attention_couple_row_importance(attn_couple)  # (H, Dh) or (E,)
            if scores.dim() == 1:
                # fallback (shouldn't happen if head info exists)
                _, keep = torch.topk(scores, k_total, largest=True)
                keep, _ = torch.sort(keep)
            else:
                keep = self._build_headaware_keep(scores, k_total)

            # build row indices for Q,K,V (contiguous per head, same positions for q/k/v)
            device_qkv = attn_couple.qkv.weight.device
            keep_q = keep.to(device_qkv)
            keep_k = keep_q + E
            keep_v = keep_q + 2 * E
            row_idx = torch.cat([keep_q, keep_k, keep_v], dim=0)

            # QKV: prune ROWS (outputs)
            attn_couple.qkv.weight = nn.Parameter(attn_couple.qkv.weight[row_idx, :])
            if attn_couple.qkv.bias is not None:
                attn_couple.qkv.bias = nn.Parameter(attn_couple.qkv.bias[row_idx])
            attn_couple.qkv.out_features = row_idx.numel()  # = 3k_total

            # PROJ: prune COLUMNS (inputs) with the SAME per-head positions (coming from V)
            device_proj = attn_couple.proj.weight.device
            col_idx = keep.to(device_proj)
            attn_couple.proj.weight = nn.Parameter(attn_couple.proj.weight[:, col_idx])
            attn_couple.proj.in_features = col_idx.numel()  # = k_total

            # ✅ CRITICAL FIX: Update attention module's head_dim based on ACTUAL pruned dimensions
            for name, module in self.model.named_modules():
                if hasattr(module, 'qkv') and module.qkv is attn_couple.qkv:
                    if hasattr(module, 'num_heads') and hasattr(module, 'head_dim'):
                        # Calculate actual head_dim from the pruned QKV output dimensions 
                        qkv_out_total = attn_couple.qkv.out_features  # 3 * k_total (after pruning)
                        internal_dim = qkv_out_total // 3  # k_total
                        actual_head_dim = internal_dim // module.num_heads

                        old_head_dim = module.head_dim  # Save old value BEFORE changing it
                        
                        # ✅ CRITICAL: Update the attention's internal dimension expectation
                        module.head_dim = actual_head_dim
                        # ✅ Add this line to fix the reshape operation:
                        if hasattr(module, 'qkv'):
                            module._pruned_embed_dim = internal_dim  # Store for reshape operations
                        
                        # Update scale factor based on new head_dim
                        if hasattr(module, 'scale'):
                            module.scale = actual_head_dim ** -0.5
                        
                        # print(f"[✅] Updated head_dim: {old_head_dim} → {actual_head_dim} (internal_dim={internal_dim})")
                    break

            self._attention_update_count = getattr(self, "_attention_update_count", 0) + 1


def finalize_pruned_model(model, new_embed_dim, calib_loader=None):
    import torch
    import torch.nn as nn

    print(f"\n[🔧] Finalizing pruned model with new_embed_dim = {new_embed_dim}")

    # 🧠 Detect final output dim using forward_features()
    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 224, 224).cuda()
        output = model.forward_features(dummy_input)
        actual_output_dim = output.shape[-1]
        print(f"[🔍] forward_features() output dim: {actual_output_dim}")

    # ✅ Optional: Recalibrate norm stats (BatchNorm, etc.)
    if calib_loader:
        model.train()
        with torch.no_grad():
            for i, (x, _) in enumerate(calib_loader):
                if i >= 50:
                    break
                model.forward_features(x.cuda(non_blocking=True))
        print(f"[🧪] Recalibrated norm stats on {i+1} batches.")

def patch_final_block(model, new_dim):
    for block in reversed(model.blocks):
        if hasattr(block, 'mlp') and hasattr(block.mlp, 'fc1'):
            block.mlp.fc1 = nn.Linear(new_dim, block.mlp.fc1.out_features).to(block.mlp.fc1.weight.device)
            block.mlp.fc2 = nn.Linear(block.mlp.fc1.out_features, new_dim).to(block.mlp.fc2.weight.device)
        if hasattr(block, 'norm2'):
            block.norm2 = nn.LayerNorm(new_dim).to(block.norm2.weight.device)
        if hasattr(block, 'attn') and hasattr(block.attn, 'qkv'):
            block.attn.qkv = nn.Linear(new_dim, block.attn.qkv.out_features).to(block.attn.qkv.weight.device)
        break  # only patch the final block

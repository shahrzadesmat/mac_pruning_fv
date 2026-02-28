import torch
import torch.nn as nn

def replace_linear_with_(model, new_class, exclude=[], groups=None,
                         attn_nm=None, mlp_nm=None, **kwargs):
    """Replace linear layers with new_class in a model. It's an inplace operation.

    Args:
        model: The model to modify
        new_class: The class to replace Linear layers with (MaskedLinear)
        exclude_names: List of module names to exclude from replacement
        groups: Dictionary of IsomorphicGroup objects with pruning ratios
        attn_nm: Optional dict {"N": int, "M": int} for direct attention N:M pattern
        mlp_nm: Optional dict {"N": int, "M": int} for direct MLP N:M pattern
        **kwargs: Additional arguments for new_class
    """
    # Default N:M pattern if no groups specified
    default_N = kwargs.get('N', 2)
    default_M = kwargs.get('M', 4)

    def get_sparsity_config(full_name):
        """Determine N:M pattern based on full layer name"""
        # Extract layer type from full name
        # Example: 'blocks.0.attn.qkv' or 'blocks.0.mlp.fc1'
        name_parts = full_name.split('.')

        # Map to group based on structure
        if len(name_parts) >= 4:
            if 'attn' in name_parts:
                if 'qkv' in name_parts or 'proj' in name_parts:
                    # Direct N:M pattern takes priority over ratio_to_nm
                    if attn_nm is not None:
                        return attn_nm
                    # Attention layers (qkv or proj) - fallback to ratio-based
                    if groups is None:
                        return {'N': default_N, 'M': default_M}
                    if hasattr(groups, 'attention_blocks'):
                        ratio = groups.attention_blocks.pruning_ratio
                    elif 'attention_blocks' in groups:
                        ratio = groups['attention_blocks'].pruning_ratio
                    else:
                        ratio = 0.15  # Default for attention
                    return ratio_to_nm(ratio)
            elif 'mlp' in name_parts:
                if 'fc1' in name_parts or 'fc2' in name_parts:
                    # Direct N:M pattern takes priority over ratio_to_nm
                    if mlp_nm is not None:
                        return mlp_nm
                    # MLP layers - fallback to ratio-based
                    if groups is None:
                        return {'N': default_N, 'M': default_M}
                    if hasattr(groups, 'mlp_blocks'):
                        ratio = groups.mlp_blocks.pruning_ratio
                    elif 'mlp_blocks' in groups:
                        ratio = groups['mlp_blocks'].pruning_ratio
                    else:
                        ratio = 0.4  # Default for MLP
                    return ratio_to_nm(ratio)

        if groups is None:
            return {'N': default_N, 'M': default_M}

        # Output projections (head, etc.)
        if 'head' in full_name or 'fc' == name_parts[-1]:
            if hasattr(groups, 'output_projections'):
                ratio = groups.output_projections.pruning_ratio
            elif 'output_projections' in groups:
                ratio = groups['output_projections'].pruning_ratio
            else:
                ratio = 0.0  # Default no pruning
            return ratio_to_nm(ratio)

        return {'N': default_N, 'M': default_M}
    
    def ratio_to_nm(pruning_ratio):
        """Convert pruning ratio to nearest N:M pattern using M in {5,6,7,8,9}.

        Uses midpoint thresholds for nearest-neighbor matching.
        27 density levels from M=5,6,7,8,9 combinations (sorted high to low):

          Density   N:M   Threshold (midpoint with next lower level)
          88.9%     8:9   >= 0.882
          87.5%     7:8   >= 0.866
          85.7%     6:7   >= 0.845
          83.3%     5:6   >= 0.817
          80.0%     4:5   >= 0.789
          77.8%     7:9   >= 0.764
          75.0%     6:8   >= 0.732
          71.4%     5:7   >= 0.691
          66.7%     4:6   >= 0.646
          62.5%     5:8   >= 0.613
          60.0%     3:5   >= 0.586
          57.1%     4:7   >= 0.564
          55.6%     5:9   >= 0.528
          50.0%     4:8   >= 0.472
          44.4%     4:9   >= 0.437
          42.9%     3:7   >= 0.414
          40.0%     2:5   >= 0.388
          37.5%     3:8   >= 0.354
          33.3%     2:6   >= 0.310
          28.6%     2:7   >= 0.268
          25.0%     2:8   >= 0.236
          22.2%     2:9   >= 0.211
          20.0%     1:5   >= 0.183
          16.7%     1:6   >= 0.155
          14.3%     1:7   >= 0.134
          12.5%     1:8   >= 0.118
          11.1%     1:9   < 0.118
        """
        density = 1 - pruning_ratio

        if density >= 0.882:
            return {'N': 8, 'M': 9}   # 88.9% dense
        elif density >= 0.866:
            return {'N': 7, 'M': 8}   # 87.5% dense
        elif density >= 0.845:
            return {'N': 6, 'M': 7}   # 85.7% dense
        elif density >= 0.817:
            return {'N': 5, 'M': 6}   # 83.3% dense
        elif density >= 0.789:
            return {'N': 4, 'M': 5}   # 80.0% dense
        elif density >= 0.764:
            return {'N': 7, 'M': 9}   # 77.8% dense
        elif density >= 0.732:
            return {'N': 6, 'M': 8}   # 75.0% dense
        elif density >= 0.691:
            return {'N': 5, 'M': 7}   # 71.4% dense
        elif density >= 0.646:
            return {'N': 4, 'M': 6}   # 66.7% dense
        elif density >= 0.613:
            return {'N': 5, 'M': 8}   # 62.5% dense
        elif density >= 0.586:
            return {'N': 3, 'M': 5}   # 60.0% dense
        elif density >= 0.564:
            return {'N': 4, 'M': 7}   # 57.1% dense
        elif density >= 0.528:
            return {'N': 5, 'M': 9}   # 55.6% dense
        elif density >= 0.472:
            return {'N': 4, 'M': 8}   # 50.0% dense
        elif density >= 0.437:
            return {'N': 4, 'M': 9}   # 44.4% dense
        elif density >= 0.414:
            return {'N': 3, 'M': 7}   # 42.9% dense
        elif density >= 0.388:
            return {'N': 2, 'M': 5}   # 40.0% dense
        elif density >= 0.354:
            return {'N': 3, 'M': 8}   # 37.5% dense
        elif density >= 0.310:
            return {'N': 2, 'M': 6}   # 33.3% dense
        elif density >= 0.268:
            return {'N': 2, 'M': 7}   # 28.6% dense
        elif density >= 0.236:
            return {'N': 2, 'M': 8}   # 25.0% dense
        elif density >= 0.211:
            return {'N': 2, 'M': 9}   # 22.2% dense
        elif density >= 0.183:
            return {'N': 1, 'M': 5}   # 20.0% dense
        elif density >= 0.155:
            return {'N': 1, 'M': 6}   # 16.7% dense
        elif density >= 0.134:
            return {'N': 1, 'M': 7}   # 14.3% dense
        elif density >= 0.118:
            return {'N': 1, 'M': 8}   # 12.5% dense
        else:
            return {'N': 1, 'M': 9}   # 11.1% dense
    
    def recursive_replace(module, prefix=''):
        """Recursively replace linear layers"""
        for name, child in module.named_children():
            # Build full name with prefix
            full_name = f"{prefix}.{name}" if prefix else name
            
            # Skip if in exclude list (supports both string names and module objects)
            if full_name in exclude or name in exclude:
                continue
            if any(child is item for item in exclude if isinstance(item, nn.Module)):
                continue
            
            if isinstance(child, nn.Linear):
                # Get sparsity config for this layer
                config = get_sparsity_config(full_name)
                
                # Prepare kwargs for new layer
                layer_kwargs = kwargs.copy()
                layer_kwargs.update(config)
                
                # Create new masked linear layer
                new_layer = new_class(
                    in_features=child.in_features,
                    out_features=child.out_features,
                    bias=child.bias is not None,
                    **layer_kwargs
                )
                
                # Copy weights and bias
                new_layer.weight.data = child.weight.data.clone()
                if child.bias is not None:
                    new_layer.bias.data = child.bias.data.clone()
                
                # Move to same device
                new_layer.to(child.weight.device)
                
                # Replace in parent module
                setattr(module, name, new_layer)
                
                print(f"Replaced {full_name}: {config}")
                
            else:
                # Recursively process children
                recursive_replace(child, full_name)
    
    # Start recursive replacement
    recursive_replace(model)
    return model
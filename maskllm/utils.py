import torch
import torch.nn as nn

def replace_linear_with_(model, new_class, exclude=[], groups=None, **kwargs):
    """Replace linear layers with new_class in a model. It's an inplace operation.
    
    Args:
        model: The model to modify
        new_class: The class to replace Linear layers with (MaskedLinear)
        exclude_names: List of module names to exclude from replacement
        groups: Dictionary of IsomorphicGroup objects with pruning ratios
        **kwargs: Additional arguments for new_class
    """
    # Default N:M pattern if no groups specified
    default_N = kwargs.get('N', 2)
    default_M = kwargs.get('M', 4)
    
    def get_sparsity_config(full_name):
        """Determine N:M pattern based on full layer name"""
        if groups is None:
            return {'N': default_N, 'M': default_M}
        
        # Extract layer type from full name
        # Example: 'blocks.0.attn.qkv' or 'blocks.0.mlp.fc1'
        name_parts = full_name.split('.')
        
        # Map to group based on structure
        if len(name_parts) >= 4:
            if 'attn' in name_parts:
                if 'qkv' in name_parts or 'proj' in name_parts:
                    # Attention layers (qkv or proj)
                    if hasattr(groups, 'attention_blocks'):
                        ratio = groups.attention_blocks.pruning_ratio
                    elif 'attention_blocks' in groups:
                        ratio = groups['attention_blocks'].pruning_ratio
                    else:
                        ratio = 0.15  # Default for attention
                    return ratio_to_nm(ratio)
            elif 'mlp' in name_parts:
                if 'fc1' in name_parts or 'fc2' in name_parts:
                    # MLP layers
                    if hasattr(groups, 'mlp_blocks'):
                        ratio = groups.mlp_blocks.pruning_ratio
                    elif 'mlp_blocks' in groups:
                        ratio = groups['mlp_blocks'].pruning_ratio
                    else:
                        ratio = 0.4  # Default for MLP
                    return ratio_to_nm(ratio)
        
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
        """Convert pruning ratio to N:M pattern"""
        # Density = 1 - pruning_ratio
        density = 1 - pruning_ratio

        # Map density to N:M patterns 
        # TODO: Need to explore more options of M values
        if density >= 0.8:   # ≤20% pruning
            return {'N': 4, 'M': 5}  # 80% dense
        elif density >= 0.75:  # ~25% pruning
            return {'N': 3, 'M': 4}  # 75% dense
        elif density >= 0.5:   # ~50% pruning
            return {'N': 2, 'M': 4}  # 50% dense
        elif density >= 0.4:   # ~60% pruning
            return {'N': 2, 'M': 5}  # 40% dense
        else:                  # High pruning
            return {'N': 1, 'M': 4}  # 25% dense
    
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
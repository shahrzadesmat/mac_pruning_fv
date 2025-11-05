from dataclasses import dataclass
import json, re
from typing import TypedDict, Any, List, Dict
import torch.nn as nn
from utils.analysis_dependency import DependencyAnalyzer
from utils.analysis_structures import MLPCouple, AttentionCouple
import torch_pruning as tp
import torch
import os



@dataclass
class IsomorphicGroup:
    """Represents an isomorphic group of layers with similar structure."""
    name: str
    layers: List[Any]
    layer_names: List[str]
    dimensions: List[int]
    mac_count: float  # Current MAC operations for this group
    target_macs: float  # Target MAC operations after pruning
    pruning_ratio: float
    constraints: List[str]

class ViTIsomorphicAnalyzer:
    """Enhanced analyzer with dependency awareness"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.dependency_analyzer = DependencyAnalyzer(model)
            
    def create_isomorphic_groups(self, target_macs: float, isomorphic_group_ratios: Dict[str, float] = None, baseline_macs: float = None) -> Dict[str, IsomorphicGroup]:
        """Create dependency-aware isomorphic groups based on MAC constraints"""
        
        # Default MAC allocation percentages for different layer types
        default_isomorphic_group_ratios = {
            'qkv_multiplier': 0.15,  # PRUNE only 15% of attention
            'mlp_multiplier': 0.40,  # PRUNE 40% of MLP
            'proj_multiplier': 0.08,    # 8% of target MAC for projection layers
            'head_multiplier': 0.02     # 2% of target MAC for head layers
        }
        
        isomorphic_group_ratios = isomorphic_group_ratios or default_isomorphic_group_ratios
        
        groups = self._create_dependency_aware_groups(target_macs, isomorphic_group_ratios, baseline_macs)
        
        return groups

    def _create_dependency_aware_groups(self, target_macs, isomorphic_group_ratios, baseline_macs):
        """Create groups that respect layer dependencies based on MAC constraints"""
        
        # Find all transformer blocks first
        transformer_blocks = self._find_transformer_blocks()
        
        print(f"[🔍] Found {len(transformer_blocks)} transformer blocks")

        # ✅ Measure total MACs up front
        example_inputs = (torch.randn(1, 3, 224, 224).to(next(self.model.parameters()).device),)
        # --- TEMPORARY FIX: attach dummy tokens if missing ---
        cls_was_none = getattr(self.model, "cls_token", None) is None
        dist_was_none = getattr(self.model, "dist_token", None) is None


        if cls_was_none:
            self.model.cls_token = nn.Parameter(
                torch.zeros(1, 1, self.model.embed_dim, device=example_inputs[0].device)
            )
        if dist_was_none and hasattr(self.model, "dist_token"):
            self.model.dist_token = nn.Parameter(
                torch.zeros(1, 1, self.model.embed_dim, device=example_inputs[0].device)
            )


        # Count ops safely
        total_macs, _ = tp.utils.count_ops_and_params(self.model, example_inputs)

        # --- Restore None to avoid pruning them ---
        # if cls_was_none:
        #     self.model.cls_token = None
        # if dist_was_none:
        #     self.model.dist_token = None


        # ✅ Fallback if baseline not provided
        if baseline_macs is None or baseline_macs <= 0:
            baseline_macs = float(total_macs)

        # ✅ Now it's safe to compute
        target_ratio = 1.0 - (target_macs / baseline_macs)

        print(f"[📊] Total model MACs: {total_macs/1e9:.3f}G")
        print(f"[📊] Target MACs: {target_macs/1e9:.3f}G")
        print(f"[📊] Target ratio: {target_ratio:.3f}")

        
        groups = {
            'mlp_blocks': IsomorphicGroup(
                name='MLP Blocks (Coupled)',
                layers=[],
                layer_names=[],
                dimensions=[],
                mac_count=0,
                target_macs=0,
                pruning_ratio=0.0,  # ✅ Will be set by solver below
                constraints=['fc1.out_features == fc2.in_features']
            ),
            'attention_blocks': IsomorphicGroup(
                name='Attention Blocks (Coupled)', 
                layers=[],
                layer_names=[],
                dimensions=[],
                mac_count=0,
                target_macs=0,
                pruning_ratio=0.0,  # ✅ Will be set by solver below
                constraints=['qkv.out_features == proj.in_features']
            ),
            'output_projections': IsomorphicGroup(
                name='Output Projections',
                layers=[],
                layer_names=[],
                dimensions=[],
                mac_count=0,
                target_macs=0,
                pruning_ratio=0.0,  # ✅ Will be set by solver below
                constraints=['Preserve residual connections']
            )
        }


        # Estimate MAC distribution
        mlp_mac_fraction = self._estimate_mlp_mac_fraction()
        attention_mac_fraction = self._estimate_attention_mac_fraction()

        groups['mlp_blocks'].mac_count = total_macs * mlp_mac_fraction
        groups['attention_blocks'].mac_count = total_macs * attention_mac_fraction


        # ---- POST-ALLOCATION SOLVER ----
        # Caps (stage-tunable via env)
        MLP_CAP   = float(os.getenv("IPRUN_MLP_CAP",   "0.99"))
        ATTN_CAP  = float(os.getenv("IPRUN_ATTN_CAP",  "0.99"))
        ATTN_FLOOR= float(os.getenv("IPRUN_ATTN_FLOOR","0.00"))

        # If user explicitly disables attention pruning, respect it by setting cap to 0
        if os.getenv("IPRUN_ENABLE_ATTN", "1") not in ("1", "true", "True"):
            ATTN_CAP = 0.0
            ATTN_FLOOR = 0.0

        total_group_macs = max(
            1e-9,
            groups['mlp_blocks'].mac_count + groups['attention_blocks'].mac_count
        )
        mlp_frac  = groups['mlp_blocks'].mac_count / total_group_macs
        attn_frac = groups['attention_blocks'].mac_count / total_group_macs

        # Multipliers represent PRUNING RATIOS (how much to remove)
        mlp_pruning = max(0.0, min(1.0, isomorphic_group_ratios.get('mlp_multiplier', 0.4)))
        attn_pruning = max(0.0, min(1.0, isomorphic_group_ratios.get('qkv_multiplier', 0.15)))

        # 🛡️ SAFETY CAP: Prevent catastrophic attention pruning
        # MAX_SAFE_ATTN_PRUNING = 0.30  # Never prune more than 30% of attention
        # if attn_pruning > MAX_SAFE_ATTN_PRUNING:
        #     print(f"[⚠️] SAFETY CAP: LLM suggested {attn_pruning:.2f} attention pruning, capping at {MAX_SAFE_ATTN_PRUNING:.2f}")
        #     attn_pruning = MAX_SAFE_ATTN_PRUNING

        # 🛡️ SAFETY CAP: Prevent catastrophic MLP pruning
        # MAX_SAFE_MLP_PRUNING = 0.85  # Never prune more than 85% of MLP
        # if mlp_pruning > MAX_SAFE_MLP_PRUNING:
        #     print(f"[⚠️] SAFETY CAP: LLM suggested {mlp_pruning:.2f} MLP pruning, capping at {MAX_SAFE_MLP_PRUNING:.2f}")
        #     mlp_pruning = MAX_SAFE_MLP_PRUNING

        # --- Helper: project infeasible request back into the feasible box
        def clamp01(x): return max(0.0, min(0.99, x))

        # Max overall reduction possible under caps
        max_reduction = mlp_frac * MLP_CAP + attn_frac * ATTN_CAP
        best_possible_macs = baseline_macs * (1.0 - max_reduction)
        print(f"[🪫] Max feasible reduction under caps: {max_reduction:.3f} "
            f"→ best achievable ≈ {best_possible_macs/1e9:.3f}G")

        
        # If target is infeasible under current caps, push to boundary
        if target_ratio > max_reduction + 1e-6:
            print(f"[❗] Infeasible: target_ratio={target_ratio:.3f} > "
                f"max_reduction={max_reduction:.3f} with caps (MLP={MLP_CAP:.2f}, ATTN={ATTN_CAP:.2f})")
            r_mlp  = MLP_CAP
            r_attn = ATTN_CAP

        else:
            # Feasible: solve a 2D allocation
            if attn_pruning <= 0.0:  # Changed from attn_retention
                # (1) Attention pruning is 0 → MLP-only pruning
                r_mlp = min(MLP_CAP, target_ratio / max(mlp_frac, 1e-9))
                rem = target_ratio - mlp_frac * r_mlp
                r_attn = rem / max(attn_frac, 1e-9)

                # If attention goes negative, back off MLP
                if r_attn < 0.0:
                    r_mlp = clamp01(target_ratio / max(mlp_frac, 1e-9))
                    r_attn = 0.0

                # Clamp to caps
                r_mlp  = min(r_mlp, MLP_CAP)
                r_attn = min(max(r_attn, 0.0), ATTN_CAP)

                # Enforce a small attention floor if MLP is near its cap and we still miss target
                achieved = mlp_frac * r_mlp + attn_frac * r_attn
                if (ATTN_CAP > 0 and r_attn < ATTN_FLOOR and (MLP_CAP - r_mlp) < 1e-5
                    and achieved + 1e-6 < target_ratio):
                    r_attn = min(ATTN_FLOOR, ATTN_CAP)
                    need   = target_ratio - attn_frac * r_attn
                    r_mlp  = min(MLP_CAP, max(0.0, need) / max(mlp_frac, 1e-9))
                    achieved = mlp_frac * r_mlp + attn_frac * r_attn

                # If still short, try MLP then Attn within caps
                if achieved + 1e-6 < target_ratio:
                    short = target_ratio - achieved
                    extra_mlp = min(MLP_CAP - r_mlp, short / max(mlp_frac, 1e-9))
                    r_mlp += max(0.0, extra_mlp)
                    achieved = mlp_frac * r_mlp + attn_frac * r_attn

                if achieved + 1e-6 < target_ratio:
                    short = target_ratio - achieved
                    extra_attn = min(ATTN_CAP - r_attn, short / max(attn_frac, 1e-9))
                    r_attn += max(0.0, extra_attn)
            else:
                # ✅ USE LLM MULTIPLIERS DIRECTLY
                # Multipliers represent PRUNING RATIOS (how much to remove)
                r_mlp  = clamp01(mlp_pruning)   # Changed from mlp_retention
                r_attn = clamp01(attn_pruning)  # Changed from attn_retention
                
                achieved = mlp_frac * r_mlp + attn_frac * r_attn
                
                print(f"[🎯] LLM suggests: mlp_pruning={mlp_pruning:.3f}, attn_pruning={attn_pruning:.3f}")
                print(f"[🎯] Using as pruning ratios: r_mlp={r_mlp:.3f}, r_attn={r_attn:.3f}")
                print(f"[🎯] Predicted achievement: {achieved:.3f} (target: {target_ratio:.3f})")
                
                # DISABLED: No auto-adjustment - let LLM learn from its mistakes!
                # This allows the LLM agent to get proper feedback and improve over iterations
                # If we auto-adjust, the LLM never learns what works
                
                # Final clamps only (prevent >99% pruning)
                r_mlp  = clamp01(r_mlp)
                r_attn = clamp01(r_attn)


        groups['mlp_blocks'].pruning_ratio       = r_mlp
        groups['attention_blocks'].pruning_ratio = r_attn

        print(f"[⚖️] Post-alloc solve -> r_mlp={r_mlp:.3f}, r_attn={r_attn:.3f} "
            f"(mlp_frac={mlp_frac:.3f}, attn_frac={attn_frac:.3f}, caps: {MLP_CAP:.2f}/{ATTN_CAP:.2f})")
        # ---- END ----


        # Calculate target MACs based on pruning ratios
        groups['mlp_blocks'].target_macs = groups['mlp_blocks'].mac_count * (1 - groups['mlp_blocks'].pruning_ratio)
        groups['attention_blocks'].target_macs = groups['attention_blocks'].mac_count * (1 - groups['attention_blocks'].pruning_ratio)
        
        print(f"[📊] MLP: {groups['mlp_blocks'].mac_count/1e9:.3f}G → {groups['mlp_blocks'].target_macs/1e9:.3f}G (ratio: {groups['mlp_blocks'].pruning_ratio:.3f})")
        print(f"[📊] Attention: {groups['attention_blocks'].mac_count/1e9:.3f}G → {groups['attention_blocks'].target_macs/1e9:.3f}G (ratio: {groups['attention_blocks'].pruning_ratio:.3f})")
        
        # Group layers by their coupled relationships
        for block_idx, block_layers in transformer_blocks.items():
            # Add MLP block (fc1 + fc2 together)
            if 'fc1' in block_layers and 'fc2' in block_layers:
                mlp_couple = MLPCouple(
                    fc1=block_layers['fc1']['layer'],
                    fc2=block_layers['fc2']['layer'],
                    fc1_name=block_layers['fc1']['name'],
                    fc2_name=block_layers['fc2']['name']
                )
                groups['mlp_blocks'].layers.append(mlp_couple)
                groups['mlp_blocks'].layer_names.append(f"block_{block_idx}_mlp")
            else:
                print(f"[❌] Missing MLP layers in block {block_idx}")
            
            # Add Attention block (qkv + proj together) 
            if 'qkv' in block_layers and 'proj' in block_layers:
                attn_couple = AttentionCouple(
                    qkv=block_layers['qkv']['layer'],
                    proj=block_layers['proj']['layer'], 
                    qkv_name=block_layers['qkv']['name'],
                    proj_name=block_layers['proj']['name']
                )
                groups['attention_blocks'].layers.append(attn_couple)
                groups['attention_blocks'].layer_names.append(f"block_{block_idx}_attn")
            else:
                print(f"[❌] Missing Attention layers in block {block_idx}")
        
        # Final summary
        print(f"[📊] Final group summary:")
        print(f"  - MLP blocks: {len(groups['mlp_blocks'].layers)} couples")
        print(f"  - Attention blocks: {len(groups['attention_blocks'].layers)} couples")
        print(f"  - Output projections: {len(groups['output_projections'].layers)} layers")
        
        return groups


    def _find_transformer_blocks(self):
        """Find and organize transformer blocks"""
        blocks = {}

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and name.startswith("blocks."):
                parts = name.split('.')
                if len(parts) >= 4 and parts[0] == 'blocks':
                    try:
                        block_idx = int(parts[1])
                        component = parts[2]
                        layer_type = parts[3]

                        if block_idx not in blocks:
                            blocks[block_idx] = {}

                        if component in ['attn', 'mlp']:
                            blocks[block_idx][layer_type] = {
                                'layer': module,
                                'name': name,
                                'component': component
                            }
                    except (ValueError, IndexError):
                        pass  # skip malformed patterns

        return blocks


    def _estimate_mlp_mac_fraction(self):
        """Estimate what fraction of total MACs come from MLP layers"""
        mlp_params = 0
        total_params = 0
        mlp_layer_count = 0
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                param_count = module.weight.numel()
                total_params += param_count
                if 'mlp' in name or 'fc1' in name or 'fc2' in name:
                    mlp_params += param_count
                    mlp_layer_count += 1
        
        print(f"[🔍] Found {mlp_layer_count} MLP layers ({mlp_params:,} total params)")
        fraction = mlp_params / max(total_params, 1) if total_params > 0 else 0.6
        print(f"[📊] MLP MAC fraction estimated: {fraction:.3f} ({mlp_params:,}/{total_params:,} params)")
        return fraction

    def _estimate_attention_mac_fraction(self):
        """Estimate what fraction of total MACs come from attention layers"""
        attention_params = 0
        total_params = 0
        attention_layer_count = 0
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                param_count = module.weight.numel()
                total_params += param_count
                if 'attn' in name or 'qkv' in name or 'proj' in name:
                    attention_params += param_count
                    attention_layer_count += 1
        
        print(f"[🔍] Found {attention_layer_count} attention layers ({attention_params:,} total params)")
        fraction = attention_params / max(total_params, 1) if total_params > 0 else 0.3
        print(f"[📊] Attention MAC fraction estimated: {fraction:.3f} ({attention_params:,}/{total_params:,} params)")
        return fraction


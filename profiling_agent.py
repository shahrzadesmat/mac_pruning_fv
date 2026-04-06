from __future__ import annotations
from typing import Dict
import traceback
from langchain_core.messages import SystemMessage, HumanMessage
from llm.provider import get_llm
from llm.prompts import PROFILING_PROMPT
from utils.timing import time_it, time_it_async
from utils.analysis_structures import PruningState
from utils.misc import _to_g
import torch
from torch import nn
from data.dataset_content import get_dataset_specific_content
import timm
import pbench
pbench.forward_patch.patch_timm_forward()
# from ptflops import get_model_complexity_info
import re
from utils.model_factory import get_model
import torch_pruning as tp


class ProfilingAgent:
    def __init__(self, llm=None):
        # Use provided LLM or default to ChatOpenAI
        self.llm = llm or get_llm()

    @time_it_async("1. MAC-aware Profiling Agent")
    async def profile_model(self, state: PruningState) -> Dict:
        """MAC-aware model structure analysis and pruning sensitivity identification."""
        # print(f"[DEBUG] State keys in profile_model: {state.keys()}")
        # print(f"[DEBUG] State type: {type(state)}")

        baseline_macs = state.get("baseline_macs")
        target_macs = state.get("target_macs")

        # Extract MAC-aware and dataset information
        model_name = state.get("model_name", "Unknown Model")
        dataset = state.get("dataset", "cifar10")
        num_classes = state.get("num_classes", 10)
        input_size = state.get("input_size", 224)
        is_subsequent = state.get('revision_number', 0) > 0
        
        macs_overshoot_tolerance_pct = state.get('macs_overshoot_tolerance_pct', 1.0)
        macs_undershoot_tolerance_pct = state.get('macs_undershoot_tolerance_pct', 5.0)

        overshoot_upper_bound = (target_macs * (1 + macs_overshoot_tolerance_pct/100)) / 1e9
        undershoot_lower_bound = (target_macs * (1 - macs_undershoot_tolerance_pct/100)) / 1e9

        # print(f"[🔍] MAC-aware profiling of {model_name} for {dataset} ({num_classes} classes, {input_size}x{input_size})")
        baseline_str = f"{baseline_macs/1e9:.3f}G" if baseline_macs is not None else "N/A"
        target_str = f"{target_macs/1e9:.3f}G" if target_macs is not None else "N/A"
        print(f"[🔍] MAC Context: {baseline_str} → {target_str} (+{macs_overshoot_tolerance_pct:.1f}%/-{macs_undershoot_tolerance_pct:.1f}%)")


        def safe_float(value, default):
            """Safely convert a value to float with fallback"""
            if value is None:
                return default
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        # Apply safe handling to baseline_macs and target_macs at the module level
        original_baseline_macs = baseline_macs
        original_target_macs = target_macs
        baseline_macs_for_calc = safe_float(baseline_macs, float('nan'))
        target_macs_for_calc = safe_float(target_macs, float('nan'))

        # Prepare subsequent info if this is a re-profiling
        subsequent_info = ""
        if is_subsequent:
            model_type = state.get('model_type', 'pruned')
            
            # Get MAC results from previous attempts
            achieved_macs = state.get('pruning_results', {}).get('achieved_macs', 
                            state.get('evaluation_results', {}).get('achieved_macs', 0))
            
            # Apply safe handling to achieved_macs
            achieved_macs = safe_float(achieved_macs, 0.0)
            
            # Calculate efficiency safely (baseline_macs is already safe)
            mac_efficiency = (achieved_macs / baseline_macs_for_calc) * 100 if baseline_macs_for_calc > 0 else 0
            
            # Get accuracy based on dataset.
            # For MaskLLM, accuracy lives in pruning_results (not evaluation_results)
            # until the fine-tuning agent runs, so check both.
            _eval = state.get('evaluation_results', {})
            _pruning = state.get('pruning_results', {})
            if dataset.lower() == 'imagenet':
                accuracy = (_eval.get('fine_tuned_top1_accuracy')
                         or _eval.get('zero_shot_top1_accuracy')
                         or _pruning.get('zero_shot_top1_accuracy')
                         or _pruning.get('zero_shot_accuracy')
                         or 0)
                accuracy_type = "Top-1"
            else:
                accuracy = (_eval.get('accuracy')
                         or _eval.get('fine_tuned_accuracy')
                         or _pruning.get('zero_shot_accuracy')
                         or 0)
                accuracy_type = "Accuracy"

            subsequent_info = f"""
            This is a subsequent MAC-aware profile of a {model_type} model on {dataset}.
            MAC Results: Achieved {achieved_macs/1e9:.3f}G from {baseline_macs_for_calc/1e9:.3f}G baseline (efficiency: {mac_efficiency:.1f}%)
            Target MAC: {target_macs_for_calc/1e9:.3f}G (+{macs_overshoot_tolerance_pct:.1f}%/-{macs_undershoot_tolerance_pct:.1f}% tolerance)
            Current {accuracy_type}: {accuracy:.2f}%
            Revision number: {state.get('revision_number', 0)}
            Dataset complexity: {dataset} ({num_classes} classes)
            Focus: Optimize MAC allocation for {target_macs/1e9:.3f}G target
            """

            # For MaskLLM, append N:M pattern history so the profiling LLM can give
            # directional advice (e.g. "go sparser" if previous attempts were too dense).
            if state.get('pruning_method') == 'maskllm':
                history = state.get('history', [])
                if history:
                    nm_lines = ["Previous MaskLLM N:M attempts (most recent last):"]
                    for entry in history:
                        nm = entry.get('strategy_used', {}).get('maskllm_nm_patterns', {})
                        mlp = nm.get('mlp_nm', {})
                        achieved = entry.get('achieved_macs')
                        target_h  = entry.get('target_macs')
                        if mlp and achieved and target_h:
                            err = (float(achieved) - float(target_h)) / float(target_h) * 100
                            if err > macs_overshoot_tolerance_pct:
                                direction = "TOO DENSE — need sparser mlp_nm"
                            elif err < -macs_undershoot_tolerance_pct:
                                direction = "TOO SPARSE — need denser mlp_nm"
                            else:
                                direction = "WITHIN TOLERANCE"
                            nm_lines.append(
                                f"  Rev {entry.get('revision', '?')}: "
                                f"mlp={mlp.get('N')}:{mlp.get('M')} → "
                                f"{float(achieved)/1e9:.3f}G ({err:+.1f}%) [{direction}]"
                            )
                    subsequent_info += "\n" + "\n".join(nm_lines)

        # Create a sample model to analyze its architecture or use the provided model
        profile_results = None  # initialised here so the except block can detect if it was built
        try:
            device = torch.device("cpu")  # Use CPU for profiling
            
            if is_subsequent and 'current_model' in state:
                # print("[🔍] MAC profiling the current model from state")
                model = state['current_model']
            else:
                # print(f"[🔍] Creating new model instance for MAC profiling: {model_name}")
                
                if dataset.lower() == 'imagenet':
                    # Use pretrained for ImageNet, with correct number of classes
                    model = get_model(model_name, num_classes, pretrained=True)
                    # print(f"[🔍] Created ImageNet model with pretrained weights and {num_classes} classes")
                else:
                    # Use pretrained ImageNet backbone even for non-ImageNet datasets
                    try:
                        model = get_model(model_name, num_classes, pretrained=True)
                    except Exception:
                        model = get_model(model_name, num_classes, pretrained=False)
                    # print(f"[🔍] Created {dataset} model with {num_classes} classes")
                
            # Dataset-aware input size
            example_inputs = (torch.randn(1, 3, input_size, input_size),)
            # print(f"[🔍] Using input size: {input_size}x{input_size}")

            # Calculate actual baseline MACs if not provided
            if original_baseline_macs is None:
                model.eval()
                with torch.no_grad():
                    # TODO: Replace with fvcore/ptflops when available
                    # For now, we'll calculate it from layer analysis below
                    measured_baseline_macs = None  # Will be set after layer analysis
                    # print(f"[🔍] Will measure baseline MACs from layer analysis")
            else:
                measured_baseline_macs = original_baseline_macs
                # print(f"[🔍] Using provided baseline MACs: {measured_baseline_macs/1e9:.3f}G")
            
            # Extract MAC-aware layer information
            layer_info = []
            mac_distribution = {}
            dependencies = []
            constraints = []
            sensitivity = []
            critical_layers = []  # Layers that absolutely should not be pruned
            mac_critical_layers = []  # Layers critical for MAC efficiency
            
            # Use ptflops to get accurate model-wide MAC measurement
            # flops, params_str = get_model_complexity_info(model, (3, input_size, input_size), as_strings=True)

            # Extract numeric FLOP value
            # if 'GMac' in flops:
            #     flops_numeric = float(flops.replace(' GMac', '')) * 1e9
            # elif 'MMac' in flops:
            #     flops_numeric = float(flops.replace(' MMac', '')) * 1e6
            # elif 'KMac' in flops:
            #     flops_numeric = float(flops.replace(' KMac', '')) * 1e3
            # else:
            #     numbers = re.findall(r'[\d.]+', flops)
            #     flops_numeric = float(numbers[0]) * 1e9 if numbers else 10e9

            # layer_macs = flops_numeric
            # print(f"[✅] ptflops measured: {flops}, converted to MACs: {layer_macs/1e9:.3f}G")

            # Check if this is a transformer model
            # Use torch_pruning for consistent MAC measurement across all models

            layer_macs, _ = tp.utils.count_ops_and_params(model, example_inputs)
            print(f"[✅] torch_pruning measured: {layer_macs/1e9:.3f} GMACs")

            # Compute per-type MAC fractions for ViT/DeiT models.
            # Used by MaskLLM analysis to replace hardcoded attn/mlp split estimates.
            attn_mac_frac = None
            mlp_mac_frac  = None
            if hasattr(model, 'blocks') and hasattr(model, 'patch_embed'):
                try:
                    patch_size = model.patch_embed.patch_size
                    if isinstance(patch_size, (tuple, list)):
                        patch_size = patch_size[0]
                    num_patches = (input_size // patch_size) ** 2
                    num_tokens  = num_patches + getattr(model, 'num_prefix_tokens', 1)

                    attn_linear_macs = 0
                    mlp_linear_macs  = 0
                    for block in model.blocks:
                        for name, module in block.named_modules():
                            if isinstance(module, nn.Linear):
                                mac = num_tokens * module.in_features * module.out_features
                                parts = name.split('.')
                                if 'attn' in parts:
                                    attn_linear_macs += mac
                                elif 'mlp' in parts:
                                    mlp_linear_macs += mac

                    attn_mac_frac = round(attn_linear_macs / layer_macs, 3)
                    mlp_mac_frac  = round(mlp_linear_macs  / layer_macs, 3)
                    print(f"[📊] ViT MAC split: attn={attn_mac_frac:.1%}, mlp={mlp_mac_frac:.1%}, "
                          f"other={1-attn_mac_frac-mlp_mac_frac:.1%}")
                except Exception as split_err:
                    print(f"[⚠️] MAC split computation failed: {split_err}, using defaults")

            # Build simplified layer_info for compatibility
            layer_info = [{
                "name": "total_model", 
                "type": "calflops_measurement", 
                "estimated_macs": layer_macs, 
                "mac_percentage": 100.0
            }]

            # Set baseline MAC variables
            estimated_total_macs = layer_macs

            # Set measured baseline if it wasn't provided
            if original_baseline_macs is None:
                measured_baseline_macs = layer_macs
                baseline_macs_for_calc = layer_macs
                # print(f"[✅] Measured baseline MACs from calflops: {baseline_macs_for_calc/1e9:.3f}G")
                # print(f"[✅] Measured baseline MACs from layers: {measured_baseline_macs/1e9:.3f}G")

            mac_distribution = {
                'estimated_baseline_macs': estimated_total_macs / 1e9 if estimated_total_macs else 0,
                'mac_reduction_needed_pct': ((baseline_macs_for_calc - target_macs_for_calc) / baseline_macs_for_calc) * 100
            }
            
            # Get model summary
            model_summary = str(model)
            
            # Generate MAC-aware architecture-specific insights
            if "resnet" in model_name.lower():
                dependencies.extend([
                    "Residual connections create MAC dependencies between blocks",
                    "Shortcut connections create strong MAC efficiency dependencies between input and output channels"
                ])
                constraints.extend([
                    "Channel dimensions must match at residual connections for MAC efficiency",
                    "Downsample layers must maintain proper MAC/dimension reduction ratios"
                ])
                sensitivity.extend([
                    "Early layers are more MAC-sensitive to pruning",
                    f"Target {target_macs_for_calc/1e9:.3f}G requires strategic conv layer MAC reduction"
                ])
                
                # Dataset-specific ResNet considerations
                if dataset.lower() == 'imagenet':
                    sensitivity.extend([
                        f"Pretrained ImageNet features should be preserved while achieving {target_macs_for_calc/1e9:.3f}G MAC target",
                        "Early conv layers critical for low-level feature extraction at scale and MAC efficiency"
                    ])
                    constraints.append("Pretrained weight structure should be maintained for MAC efficiency")
                else:
                    sensitivity.append(f"Can be more aggressive with MAC reduction for {target_macs_for_calc/1e9:.3f}G target due to simpler dataset")
                
            elif "vit" in model_name.lower() or "swin" in model_name.lower() or "deit" in model_name.lower():
                if state.get('pruning_method') == 'maskllm':
                    _mlp_frac  = mlp_mac_frac  if mlp_mac_frac  is not None else 0.60
                    _attn_frac = attn_mac_frac if attn_mac_frac is not None else 0.28
                    dependencies.append(
                        "N:M sparsity is applied independently per layer — no structural dependencies between layers"
                    )
                    constraints.extend([
                        "All layer dimensions are preserved — only weights are zeroed in N:M patterns",
                        "Both MLP (fc1/fc2) and attention (q/k/v/proj) layers are pruning targets via N:M sparsity",
                    ])
                    sensitivity.extend([
                        "Both MLP and attention layers contribute to MAC reduction",
                        f"MLP layers account for ~{_mlp_frac:.0%} of total MACs; attention layers ~{_attn_frac:.0%}",
                        "Denser N:M (higher N/M ratio) → more MACs remaining; sparser → fewer MACs",
                        f"To hit {target_macs_for_calc/1e9:.3f}G, pick attn_nm and mlp_nm such that: "
                        f"baseline × (mlp_density × {_mlp_frac:.2f} + attn_density × {_attn_frac:.2f} + {1-_mlp_frac-_attn_frac:.2f}) ≈ target",
                    ])
                    if dataset.lower() == 'imagenet':
                        sensitivity.append(
                            "ImageNet accuracy is sensitive to over-pruning — prefer denser N:M patterns"
                        )
                    else:
                        # CIFAR-10 specific MaskLLM guidance
                        sensitivity.extend([
                            "CIFAR-10 tolerates aggressive sparsity — sparser N:M patterns are acceptable for both MLP and attention",
                            "Fine-tuning on CIFAR-10 recovers accuracy well even after aggressive N:M pruning",
                        ])
                else:
                    dependencies.extend([
                        "Attention mechanisms create complex MAC dependencies",
                        "Multi-head attention requires consistent head dimensions for MAC efficiency"
                    ])
                    constraints.extend([
                        "Head dimensions must be maintained for attention MAC calculations",
                        "Embedding dimensions must be consistent for MAC efficiency across layers"
                    ])
                    sensitivity.extend([
                        "Head pruning generally preferred over full layer pruning for MAC optimization",
                        f"MLP and QKV blocks are primary targets for {target_macs_for_calc/1e9:.3f}G MAC reduction"
                    ])

                    # Dataset-specific ViT considerations
                    if dataset.lower() == 'imagenet':
                        sensitivity.extend([
                            f"Pretrained attention patterns valuable for ImageNet at {target_macs_for_calc/1e9:.3f}G MAC target",
                            "Patch embedding layer critical for image tokenization and MAC efficiency"
                        ])
                        constraints.append("Position embeddings should be preserved for MAC-efficient processing")
                    else:
                        sensitivity.append(f"Attention layers can handle more aggressive MAC reduction for {target_macs_for_calc/1e9:.3f}G target on simpler datasets")

                if "swin" in model_name.lower():
                    constraints.append("Window partition mechanisms must be preserved for MAC efficiency")
                    
            else:
                dependencies.append("Standard feed-forward MAC dependencies between layers")
                sensitivity.extend([
                    "Deeper layers typically less MAC-sensitive to pruning",
                    f"Target {target_macs_for_calc:.3f}G requires systematic MAC reduction strategy"
                ])
                
                # Dataset-specific general considerations
                if dataset.lower() == 'imagenet':
                    sensitivity.append(f"Complex feature hierarchies require conservative MAC reduction to {target_macs_for_calc/1e9:.3f}G")
                else:
                    sensitivity.append(f"Simple feature requirements allow aggressive MAC reduction to {target_macs_for_calc/1e9:.3f}G")
            

            mac_efficiency_target = (
                (target_macs_for_calc / baseline_macs_for_calc) * 100
                if isinstance(baseline_macs_for_calc, (int, float)) and baseline_macs_for_calc > 0
                and isinstance(target_macs_for_calc, (int, float))
                else None
            )

            if dataset.lower() == 'imagenet':
                constraints.extend([
                    f"1000-class classifier requires substantial MAC capacity at {target_macs_for_calc:.3f}G target",
                    f"Complex feature extraction needs sufficient MAC budget ({mac_efficiency_target:.1f}% efficiency)" if mac_efficiency_target is not None else "Complex feature extraction needs sufficient MAC budget",
                    "Pretrained weights contain valuable learned representations for MAC-efficient processing"
                ])
                sensitivity.extend([
                    f"Early layers extract critical low-level features for complex images at {mac_efficiency_target:.1f}% MAC efficiency",
                    f"Final classifier MAC-sensitive due to 1000-way classification at {target_macs_for_calc:.3f}G budget",
                    "Middle layers can be MAC-pruned more aggressively than early/late layers"
                ])
            else:  # CIFAR-10 or similar
                constraints.extend([
                    f"{num_classes}-class classifier can handle significant MAC reduction",
                    f"Simpler images require less complex feature extraction at {target_macs_for_calc/1e9:.3f}G target"
                ])
                sensitivity.extend([
                    f"Final classifier less MAC-sensitive due to fewer classes at {target_macs_for_calc:.3f}G target",
                    f"Can use more aggressive MAC reduction ratios to achieve {mac_efficiency_target:.1f}% efficiency",
                    f"Less risk of feature degradation with aggressive MAC pruning to {target_macs_for_calc:.3f}G"
                ])
            
            # Compile MAC-aware profile results with dataset information
            profile_results = {
                "model_summary": model_summary,
                "layer_info": layer_info,
                "mac_distribution": mac_distribution,
                "dependencies": dependencies,
                "constraints": constraints,
                "sensitivity": sensitivity,
                "critical_layers": critical_layers,
                "mac_critical_layers": mac_critical_layers,
                "is_subsequent_profile": is_subsequent,
                "dataset": dataset,
                "num_classes": num_classes,
                "input_size": input_size,
                "model_complexity": "high" if dataset.lower() == 'imagenet' else "moderate",
                "baseline_macs": measured_baseline_macs or original_baseline_macs or layer_macs,
                "target_macs": original_target_macs,
                "measured_layer_macs": layer_macs,
                "macs_overshoot_tolerance_pct": macs_overshoot_tolerance_pct,
                "macs_undershoot_tolerance_pct": macs_undershoot_tolerance_pct,
                "mac_efficiency_target": mac_efficiency_target
            }
            
            # For subsequent profiles, add MAC analysis of changes since initial profile
            if is_subsequent:
                profile_results["changes_since_initial"] = f"Model has been MAC-pruned and fine-tuned for {dataset} targeting {target_macs:.3f}G operations"
            

            # Get MAC-aware dataset-specific content for the prompt
            dataset_content = get_dataset_specific_content(
                dataset, num_classes, input_size, baseline_macs, target_macs, 
                macs_overshoot_tolerance_pct, macs_undershoot_tolerance_pct,
                state.get('accuracy_threshold', 85.0)
            )
            

            mac_reduction_needed = ((baseline_macs_for_calc - target_macs_for_calc) / baseline_macs_for_calc) * 100
            
            prompt_text = PROFILING_PROMPT.format(
                model_arch=str(model_name),
                dataset=dataset,
                num_classes=num_classes,
                input_size=input_size,
                baseline_macs=baseline_macs_for_calc/1e9,
                target_macs=target_macs_for_calc,
                macs_overshoot_tolerance_pct=macs_overshoot_tolerance_pct,
                macs_undershoot_tolerance_pct=macs_undershoot_tolerance_pct,
                overshoot_upper_bound=overshoot_upper_bound,
                undershoot_lower_bound=undershoot_lower_bound,
                mac_reduction_needed=mac_reduction_needed,
                dataset_considerations=dataset_content['dataset_guidance'],
                is_subsequent=is_subsequent,
                subsequent_info=subsequent_info
            )
            
            messages = [
                SystemMessage(content=prompt_text),
                HumanMessage(content=state['query'])
            ]
            response = await self.llm.ainvoke(messages)
            
            # Combine automated and LLM analysis
            profile_results["analysis"] = response.content
            
            # print(f"[✅] MAC-aware profiling complete for {dataset} model with {len(layer_info)} analyzable layers")
            eff_str = f"{mac_efficiency_target:.1f}%" if mac_efficiency_target is not None else "N/A"
            print(f"[✅] MAC Target: {target_macs_for_calc/1e9:.3f}G ({eff_str} efficiency)")

            
            result = {'profile_results': profile_results}
            if attn_mac_frac is not None:
                result['attn_mac_frac'] = attn_mac_frac
                result['mlp_mac_frac']  = mlp_mac_frac
            return result
            
        except Exception as e:
            print(f"Error in MAC-aware profiling: {str(e)}")
            import traceback
            print(f"[⚠️] MAC profiling error traceback: {traceback.format_exc()}")
            
            # Fallback to LLM-based profiling with MAC context
            try:
                dataset_content = get_dataset_specific_content(
                    dataset, num_classes, input_size, baseline_macs, target_macs,
                    macs_overshoot_tolerance_pct, macs_undershoot_tolerance_pct,
                    state.get('accuracy_threshold', 85.0)
                )
                
                mac_reduction_needed = ((baseline_macs_for_calc - target_macs_for_calc) / baseline_macs_for_calc) * 100

                
                prompt_text = PROFILING_PROMPT.format(
                    model_arch=str(model_name),
                    dataset=dataset,
                    num_classes=num_classes,
                    input_size=input_size,
                    baseline_macs=baseline_macs_for_calc/1e9,
                    target_macs=target_macs_for_calc,
                    macs_overshoot_tolerance_pct=macs_overshoot_tolerance_pct,
                    macs_undershoot_tolerance_pct=macs_undershoot_tolerance_pct,
                    overshoot_upper_bound=overshoot_upper_bound,
                    undershoot_lower_bound=undershoot_lower_bound,
                    mac_reduction_needed=mac_reduction_needed,
                    dataset_considerations=dataset_content['dataset_guidance'],
                    is_subsequent=is_subsequent,
                    subsequent_info=subsequent_info
                )
                
                messages = [
                    SystemMessage(content=prompt_text),
                    HumanMessage(content=state['query'])
                ]
                response = await self.llm.ainvoke(messages)

                # Reuse already-computed profile_results if available (preserves measured
                # baseline_macs). This matters when the first LLM call timed out after
                # MAC measurement already succeeded.
                if profile_results is not None:
                    profile_results["analysis"] = response.content
                    profile_results["error_fallback"] = True
                    return {'profile_results': profile_results}

                safe_bm = baseline_macs_for_calc if (baseline_macs_for_calc is not None and not __import__('math').isnan(baseline_macs_for_calc)) else baseline_macs
                return {'profile_results': {
                    "analysis": response.content,
                    "dataset": dataset,
                    "num_classes": num_classes,
                    "input_size": input_size,
                    "model_complexity": "high" if dataset.lower() == 'imagenet' else "moderate",
                    "baseline_macs": safe_bm,
                    "target_macs": target_macs,
                    "macs_overshoot_tolerance_pct": macs_overshoot_tolerance_pct,
                    "macs_undershoot_tolerance_pct": macs_undershoot_tolerance_pct,
                    "error_fallback": True
                }}

            except Exception as fallback_error:
                print(f"[❌] Fallback MAC profiling also failed: {fallback_error}")
                safe_target_macs = target_macs / 1e9 if target_macs is not None else 5.0
                safe_bm = baseline_macs_for_calc if (baseline_macs_for_calc is not None and not __import__('math').isnan(baseline_macs_for_calc)) else baseline_macs
                if profile_results is not None:
                    profile_results["analysis"] = f"Basic MAC profile for {model_name} on {dataset}. Target: {safe_target_macs:.3f}G. Error during detailed analysis."
                    profile_results["critical_failure"] = True
                    return {'profile_results': profile_results}
                return {
                    'profile_results': {
                        'analysis': f"Basic MAC profile for {model_name} on {dataset}. Target: {safe_target_macs:.3f}G. Error during detailed analysis.",
                        'dataset': dataset,
                        'num_classes': num_classes,
                        'input_size': input_size,
                        'baseline_macs': safe_bm,
                        'target_macs': target_macs,
                        'critical_failure': True
                    }
                }


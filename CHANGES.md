# Branch Changes: `dhawal04` vs `main`

All changes in `.py` files relative to the `main` branch, including both committed and uncommitted modifications.

---

## 1. `deepseek_llm.py`

### What changed
Consolidated two separate imports into one line:
```python
# Before
from langchain_core.messages import BaseMessage
from langchain.schema import HumanMessage

# After
from langchain_core.messages import BaseMessage, HumanMessage
```

### Why
`langchain.schema` is the old deprecated API. `HumanMessage` has moved to `langchain_core.messages` in modern versions of LangChain. This removes a deprecation warning and keeps all LangChain imports from the correct modern package.

---

## 2. `llm/provider.py`

### What changed
Switched the LLM model used by agents:
```python
# Before
model="anthropic/claude-3.5-sonnet"

# After
model="minimax/minimax-m2.1"
```

### Why
Experimenting with a different model on OpenRouter during iterative development to evaluate cost and performance trade-offs for agent decision-making.

---

## 3. `llm/prompts.py`

### What changed

**`get_vit_analysis_content()`**
- Added `pruning_method` detection from `state`.
- When `pruning_method == 'maskllm'`, the function now returns a completely different set of content sections (`guidance`, `safety_limits`, `param_defs`, `output_format`) that ask the LLM to pick N:M sparsity patterns from a 27-level table instead of scalar multipliers.
- The N:M output format asks for `maskllm_nm_patterns: {attn_nm: {N, M}, mlp_nm: {N, M}}` in JSON.
- `round_to` is intentionally omitted from the MaskLLM output format because MaskLLM does not remove channels — it only zeros weights in N:M patterns, so channel-count rounding is irrelevant.
- Fixed a bug where `state.get('revision_number', 0)` would crash when `state=None` (now guarded with `if state else 0`).

**`format_analysis_prompt()`**
- Now passes `state=state` to both calls of `get_vit_analysis_content`, so `pruning_method` is visible inside that function.

**`format_vit_history_analysis()`**
- Added a MaskLLM-specific branch at the top. When history entries contain `maskllm_nm_patterns` in their `strategy_used`, the history is formatted showing `attn N:M → achieved MACs [status]` (TOO DENSE / TOO SPARSE / SUCCESS) instead of multiplier values.
- Also appends a "DO NOT repeat" instruction listing all previously tried combinations.

**Prompt text updates (earlier iteration)**
- Updated guidance text for both ImageNet and CIFAR-10 to say "Use MaskLLM method" instead of specific importance criteria.
- Fixed a typo: `"Ensuremlp_multiplier"` → `"Ensure mlp_multiplier"`.

### Why
The original prompt always asked for scalar multipliers (`isomorphic_group_ratios`), which is correct for structural pruning but wrong for MaskLLM. MaskLLM requires discrete N:M patterns, not continuous ratios. These changes make the analysis prompt adapt based on which pruning method is active, while keeping the full context (profiling agent output, master agent directives, history) intact — the content changes but the flow does not.

---

## 4. `analysis_agent.py`

### What changed

**`analyze()`**
- Added `self._pruning_method = state.get('pruning_method', 'structural')` at the start so all sub-methods can access it via `self`.
- Forces `importance_criterion = 'maskllm'` when `pruning_method == 'maskllm'`, overriding any LLM-suggested criterion like `taylor` or `l1norm`.
- Added `maskllm_nm_patterns` to the returned `analysis_results` dict so it flows downstream to the pruning agent.

**`_execute_vit_analysis_with_history()`**
- Added a MaskLLM early-return branch at the top. When `pruning_method == 'maskllm'`, it calls `format_analysis_prompt(state)` — the full analysis prompt that includes profiling agent output, master agent directives, and history — and invokes the LLM once, parsing `maskllm_nm_patterns` from the response.
- Previously, MaskLLM bypassed the full prompt flow entirely and used isolated standalone prompts that ignored profiling and master agent context. This fix uses the actual flow for both methods.

**`_llm_calculate_vit_strategy_with_history()`**
- Removed the MaskLLM early-return intercept that routed to `_llm_calculate_maskllm_nm_with_history()`. No longer needed since MaskLLM now exits at `_execute_vit_analysis_with_history()`.
- Fixed a bug in the same method where it was calling `_llm_calculate_vit_strategy_baseline` instead of `_llm_calculate_vit_strategy_with_history` when history existed (wrong function, missing `history` arg).

**`_llm_calculate_vit_strategy_baseline()`**
- Removed the MaskLLM early-return intercept that routed to `_llm_calculate_maskllm_nm_baseline()`. No longer needed.

**`_validate_learning_reasoning()`**
- Added MaskLLM-aware validation path: instead of checking multiplier direction, it checks whether N:M density direction matches the MAC error direction (too dense → go sparser; too sparse → go denser).

**`_format_vit_history_for_llm_learning()`**
- Added MaskLLM-specific history formatter showing N:M patterns per attempt instead of multiplier values.

**Deleted methods**
- `_llm_calculate_maskllm_nm_baseline()` — removed. This was a standalone MaskLLM prompt that bypassed the full analysis flow. Replaced by the `_execute_vit_analysis_with_history` branch above.
- `_llm_calculate_maskllm_nm_with_history()` — removed. Same reason.

### Why
The core problem was two contradictory MaskLLM paths: the old bypass path (standalone prompts, no profiling/master context) and the actual flow (`format_analysis_prompt`, rich context). The fix consolidates everything into the actual flow, with MaskLLM only differing in what the LLM is asked to output (N:M patterns vs scalar multipliers).

---

## 5. `pbench/forward_patch.py`

### What changed
Added `attn_mask=None` parameter to the patched `Attention.forward()` and passes it through to `F.scaled_dot_product_attention()`:
```python
# Before
def forward(self, x):
    ...
    x = F.scaled_dot_product_attention(q, k, v, dropout_p=...)

# After
def forward(self, x, attn_mask=None):
    ...
    x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=...)
```

### Why
Newer versions of timm updated the `Attention.forward()` signature to include `attn_mask`. Without this, calling the model with an attention mask would crash with a signature mismatch. This keeps the patched forward compatible with the current timm API.

---

## 6. `utils/io.py`

### What changed

**`save_final_best_model()`**
1. Added `weights_only=False` to `torch.load()`:
   ```python
   # Before
   ckpt = torch.load(pruned_checkpoint, map_location='cpu')
   # After
   ckpt = torch.load(pruned_checkpoint, map_location='cpu', weights_only=False)
   ```

2. Made `pruning_method` dynamic instead of hardcoded:
   ```python
   # Before
   'pruning_method': 'isomorphic_dependency_aware',
   # After
   'pruning_method': best_candidate.get('strategy_used', {}).get('pruning_method', 'isomorphic_dependency_aware'),
   ```

### Why
1. PyTorch 2.x changed the default of `weights_only` to `True`, which breaks loading checkpoints that contain non-tensor objects (configs, model metadata). MaskLLM checkpoints store full dicts including configs, so `weights_only=False` is required.
2. With MaskLLM added as a second pruning method, the saved model metadata should correctly record `'maskllm'` rather than always writing `'isomorphic_dependency_aware'`.

---

## 7. `eval_agent.py`

### What changed

**`reattach_heads_and_tokens()`**
- Added type-based strict check before loading head state dicts:
  ```python
  # Before
  model.head.load_state_dict(original_model.head.state_dict())

  # After
  strict = type(model.head) == type(original_model.head)
  model.head.load_state_dict(original_model.head.state_dict(), strict=strict)
  ```
  Applied to both `head` and `head_dist`.

**`EvaluationAgent.evaluate()`**
- Fixed MAC tolerance calculation bug:
  ```python
  # Before (wrong — applied percentage to raw ops value)
  mac_overshoot_tolerance_g = target_macs * (macs_overshoot_tolerance_pct / 100.0)

  # After (correct — convert to G first, then apply percentage)
  mac_overshoot_tolerance_g = (target_macs / 1e9) * (macs_overshoot_tolerance_pct / 100.0)
  ```

### Why
1. After MaskLLM pruning, the model's `head` remains a regular `nn.Linear` while the original model's head may differ in type (e.g., `MaskedLinearFrozen`). Strict loading with mismatched types raises an error. The type check makes it fall back to non-strict loading when types differ.
2. `target_macs` is stored in raw ops (e.g. `5.65e8`). The original code applied the tolerance percentage directly to that number, producing a tolerance in ops that was millions of times too large. Dividing by `1e9` first gives the correct G-scale tolerance.

---

## 8. `finetune_agent.py`

### What changed

**`_setup_cifar10_loaders()`** — Transform order fixed:
```python
# Before (wrong — Resize after Normalize)
transforms.RandomCrop(32, padding=4),
transforms.ToTensor(),
transforms.Normalize(mean, std),
transforms.Resize((224, 224), antialias=True)

# After (correct — Resize on raw PIL image first)
transforms.Resize((224, 224), antialias=True),
transforms.RandomCrop(224, padding=28),
transforms.ToTensor(),
transforms.Normalize(mean, std),
```

**`_get_dataset_specific_params()`** — ViT CIFAR-10 hyperparameters updated:
```python
# Before
'num_epochs': 5, 'learning_rate': 0.005, 'weight_decay': 0.05

# After
'num_epochs': 30, 'learning_rate': 0.0005, 'weight_decay': 0.01
```

**`finetune()`**
- LR is now read from the centralised `params` dict instead of re-implementing per-dataset logic inline.
- Batch size increased to 128 for ViT models (was hardcoded 64).
- `weight_decay` is now read from `params` dict for both AdamW and SGD.
- Replaced constant LR scheduler with cosine annealing + linear warmup:
  ```python
  # Before
  scheduler = ConstantLR(optimizer, factor=1.0)

  # After
  warmup → CosineAnnealingLR via SequentialLR
  ```
- Replaced naive early stopping ("stop if val drops >2% from best") with patience-based stopping (stop after 5 consecutive non-improving epochs).
- Added `scheduler.step()` call inside the training loop (was missing entirely).

### Why
- **Transform order**: Resizing after normalisation is incorrect — augmentation should happen on raw pixel values before normalisation. The old order also used a `32→224` resize after crop, meaning images were upscaled from 32px crops which degraded quality.
- **Hyperparameters**: 5 epochs at LR 0.005 was too aggressive for recovering accuracy after pruning. 30 epochs at 0.0005 gives the model more iterations at a stable learning rate, directly addressing the poor fine-tuning results observed in experiments.
- **Cosine annealing**: Constant LR is suboptimal for fine-tuning pruned models. Cosine decay with warmup is standard practice and helps converge to a better minimum.
- **Patience stopping**: The old criterion could halt training on a single bad epoch. Patience-based stopping is more robust.

---

## 9. `main.py`

### What changed
- Added `--pruning_method` CLI argument:
  ```python
  parser.add_argument('--pruning_method', type=str, default='structural',
                      choices=['structural', 'unstructured', 'maskllm'])
  ```
- Passes `pruning_method` into `initial_state_mods` so it flows through the entire workflow.
- Cleaned up duplicate imports (`asyncio`, `traceback`, `contextmanager`, `DataLoader`, `typing`) that appeared twice due to merging code from different files.

### Why
`--pruning_method` is the user-facing entry point to switch between structural channel pruning and MaskLLM N:M sparsity. Without it, there was no way to select the pruning strategy at runtime. The import deduplication removes potential name shadowing issues.

---

## 10. `workflow.py`

### What changed
- Reads `pruning_method` from `state_mods` and stores it in `GLOBAL_STATE`:
  ```python
  if 'pruning_method' in state_mods:
      pruning_method = state_mods['pruning_method']
  ```
- Added `'pruning_method': pruning_method` to the `GLOBAL_STATE` dict (appears in both the initial state block and the state passed to agents).
- Removed ~40 lines of duplicate imports that had been copied into `workflow.py` but belonged in `main.py`.

### Why
`pruning_method` must be visible to every agent in the pipeline (analysis → pruning → fine-tune → eval). Without propagating it through `GLOBAL_STATE`, agents default to structural pruning regardless of what was passed on the CLI.

---

## 11. `maskllm/maskllm.py` *(new file)*

### What it is
Defines two custom `nn.Linear` subclasses:
- **`MaskedLinear`**: Learnable N:M mask selection via Gumbel-Softmax gates. During training, gates select which of the `C(M,N)` possible N:M mask patterns to apply, using temperature (`tau`) annealing to sharpen from soft to hard selection.
- **`MaskedLinearFrozen`**: Fixed N:M mask. Used after mask learning is complete — the chosen mask is baked in and the weights are zeroed accordingly during the forward pass.

Also contains `generate_N_M_masks(N, M)` which enumerates all valid N:M binary mask patterns.

### Why
MaskLLM requires replacing standard linear layers with masked variants before training. `MaskedLinear` is used during the mask learning phase; `MaskedLinearFrozen` is used for inference and fine-tuning after masks are decided.

---

## 12. `maskllm/utils.py` *(new file)*

### What it is
`replace_linear_with_(model, cls, exclude, groups)` — walks the model and swaps every `nn.Linear` (except excluded layers like the classifier head) with an instance of `cls` (`MaskedLinear` or `MaskedLinearFrozen`), using isomorphic group definitions to assign the correct N:M values per layer type.

### Why
Encapsulates the layer-replacement logic that must happen before both SparseGPT initialisation and MaskLLM training. Keeping it in a utility function avoids duplicating the traversal logic in `pruning_agent.py` and `adapter.py`.

---

## 13. `maskllm/sparsegpt.py` *(new file)*

### What it is
`prune_sparsegpt(model, loader, nsamples, batch_size, device, groups)` — runs SparseGPT-style N:M pruning on a model that has already had its linear layers replaced with `MaskedLinearFrozen`. Uses Hessian-guided weight reconstruction (OBS) to find near-optimal N:M masks given the calibration data.

### Why
Random N:M initialisation leads to poor accuracy. SparseGPT uses second-order information (Hessian of the loss) to pick which weights to zero such that the output error is minimised. This gives MaskLLM a strong starting point before the learnable gate training phase.

---

## 14. `maskllm/timm_train_simplified.py` *(new file)*

### What it is
A self-contained training loop adapted from timm for the MaskLLM mask-learning phase. Handles AMP, gradient clipping, tau/scaling annealing, and top-1/top-5 validation.

### Why
The existing `finetune_agent.py` training loop is coupled to the workflow state and fine-tuning assumptions. MaskLLM mask training has different requirements (only gate parameters are trainable, tau annealing is needed), so a separate simplified loop avoids coupling and is easier to configure independently.

---

## 15. `maskllm/adapter.py` *(new file)*

### What it is
`MaskLLMRunner` class — a clean, config-driven wrapper around the full MaskLLM workflow (replace linears → optionally run SparseGPT → freeze weights, train gates → validate → save checkpoint). Exposes a single `run(cfg)` method.

Also provides `run_maskllm_from_strategy(cfg)` as a convenience entry point callable from other agents.

### Why
The MaskLLM pipeline embedded in `pruning_agent.py` is tightly coupled to the agent state machine. `adapter.py` extracts the same logic into a reusable class that can be called standalone (e.g., from a notebook or a future agent) without needing the full workflow state.

---

## 16. `pruning_agent.py`

### What changed
- Added imports for `MaskedLinear`, `MaskedLinearFrozen`, `prune_sparsegpt`, `replace_linear_with_`, timm optimizer/scheduler utilities, and `contextlib.suppress`.
- Added routing in `_execute_pruning()`: when `pruning_method == 'maskllm'` and the model is a ViT, it dispatches to `_execute_maskllm_pruning()` before the structural pruning path.
- Added `_execute_maskllm_pruning()` — the full 4-phase MaskLLM pipeline:
  - **Phase 1 (SparseGPT init)**: Replaces linears with `MaskedLinearFrozen`, runs SparseGPT on calibration data to get Hessian-optimal N:M masks.
  - **Phase 2 (MaskLLM training)**: Creates fresh model with `MaskedLinear` (learnable gates), loads SparseGPT weights, freezes all params except gates, trains with cosine LR + AMP + tau/scaling annealing.
  - **Phase 3 (mask extraction)**: Hardens learned gate selections into `MaskedLinearFrozen` with fixed masks.
  - **Phase 4 (evaluate & return)**: Measures final MACs and accuracy, saves checkpoint, logs to W&B, returns structured results compatible with the rest of the workflow.
- Added `_maskllm_train_one_epoch()` and `_maskllm_validate()` helper methods.

### Why
The existing structural pruning path physically removes channels using a dependency graph, which changes the model architecture. MaskLLM instead keeps all channels intact and zeros out weights in N:M patterns supported by modern GPU sparse tensor cores. This avoids the accuracy cliff from aggressive structural changes and is better suited for ViTs where attention head structure must be preserved.

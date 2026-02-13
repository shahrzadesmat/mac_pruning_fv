import math
import time

import torch
import torch.nn as nn
import transformers


DEBUG = False 

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class SparseGPT:

    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out, blocksize=1024):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def fasterprune(
        self, sparsity_ratio, prunen=0, prunem=0, blocksize=128, percdamp=.01, disable_update=False
    ):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()
        M= torch.ones_like(W)

        # Handle divisible size for custom N:M patterns  
        # if hasattr(self.layer, 'divisible_size'):  
        #     # Only process the divisible portion for N:M pruning  
        #     divisible_cols = self.layer.divisible_size // W.shape[0]  # Convert back to columns  
        #     effective_cols = (divisible_cols // prunem) * prunem  # Round down to nearest multiple of prunem  
        # else:  
        #     effective_cols = W.shape[1]  

        tick = time.time()

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        Losses = torch.zeros(self.rows, device=self.dev)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        mask = None

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
        # for i1 in range(0, effective_cols, blocksize):
        #     i2 = min(i1 + blocksize, effective_cols)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            M1 = M[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            if prunen == 0: 
                if mask is not None:
                    mask1 = mask[:, i1:i2]
                else:
                    tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                    thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity_ratio)]
                    mask1 = tmp <= thresh
            else:
                mask1 = torch.zeros_like(W1) == 1

            for i in range(count):
                w = W1[:, i]
                m = M1[:, i]
                d = Hinv1[i, i]

                if prunen != 0 and i % prunem == 0:
                    # tmp = W1[:, i:(i + prunem)] ** 2 / (torch.diag(Hinv1)[i:(i + prunem)].reshape((1, -1))) ** 2
                    # mask1.scatter_(1, i + torch.topk(tmp, prunen, dim=1, largest=False)[1], True)
                    if i + prunem <= W1.shape[1]:  
                        tmp = W1[:, i:(i + prunem)] ** 2 / (torch.diag(Hinv1)[i:(i + prunem)].reshape((1, -1))) ** 2  
                        mask1.scatter_(1, i + torch.topk(tmp, prunen, dim=1, largest=False)[1], True)  
                    else:  
                        # Handle the case where remaining columns < prunem  
                        remaining_cols = W1.shape[1] - i  
                        if remaining_cols > 0 and prunen <= remaining_cols:  
                            tmp = W1[:, i:] ** 2 / (torch.diag(Hinv1)[i:].reshape((1, -1))) ** 2  
                            # Adjust prunen proportionally for the remaining columns  
                            adjusted_prunen = min(prunen, remaining_cols - (remaining_cols % prunem))  
                            if adjusted_prunen > 0:  
                                mask1.scatter_(1, i + torch.topk(tmp, adjusted_prunen, dim=1, largest=False)[1], True)

                q = w.clone()
                q[mask1[:, i]] = 0
                m[mask1[:, i]] = 0

                Q1[:, i] = q
                M1[:, i] = m
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                if not disable_update:
                    err1 = (w - q) / d
                    W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                    Err1[:, i] = err1
            if not disable_update:
                W[:, i1:i2] = Q1
            M[:, i1:i2] = M1
            Losses += torch.sum(Losses1, 1) / 2
            if not disable_update:
                W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()
        print('time %.2f' % (time.time() - tick))
        print('error', torch.sum(Losses).item())

        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
 
        self.layer.mask.data = M.to(dtype=self.layer.weight.dtype)
        if not disable_update:
            self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        torch.cuda.empty_cache()


def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if isinstance(module, tuple(layers)):
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


@torch.no_grad()
def prune_sparsegpt(model, loader, nsamples=128, batch_size=1, device=torch.device("cuda:0"),
                    prune_n=0, prune_m=0, sparsity_ratio=0.0, disable_update=False, groups=None):
    # Initialize input caches (unchanged)
    model.to(device)
    layers = model.blocks
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.num_prefix_tokens + model.patch_embed.num_patches, model.embed_dim), dtype=dtype, device=device
    )
    cache = {'i': 0}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in loader:
        if cache['i'] == nsamples: break
        try:
            model(batch[0].to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i]
        inps, outs = inps.to(device), outs.to(device)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = SparseGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0))[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')

            # Get layer-specific N:M pattern
            layer_n, layer_m = get_layer_sparsity(name, groups, prune_n, prune_m)

            gpts[name].fasterprune(sparsity_ratio, prunen=layer_n, prunem=layer_m,
                                   percdamp=0.01, blocksize=128, disable_update=disable_update)
            gpts[name].free()

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0))[0]

        layers[i] = layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    torch.cuda.empty_cache()


def get_layer_sparsity(layer_name, groups, default_n, default_m):
    """Get N:M pattern for a specific layer based on groups"""
    if groups is None:
        return default_n, default_m

        # Extract layer type from name
    name_parts = layer_name.split('.')

    # Check if this is an attention layer
    if 'attn' in name_parts:
        if 'qkv' in name_parts or 'proj' in name_parts:
            if 'attention_blocks' in groups:
                ratio = groups['attention_blocks'].pruning_ratio
                return ratio_to_nm(ratio)

                # Check if this is an MLP layer
    elif 'mlp' in name_parts:
        if 'fc1' in name_parts or 'fc2' in name_parts:
            if 'mlp_blocks' in groups:
                ratio = groups['mlp_blocks'].pruning_ratio
                return ratio_to_nm(ratio)

                # Default
    return default_n, default_m


def ratio_to_nm(pruning_ratio):
    """Convert pruning ratio to N:M pattern"""
    density = 1 - pruning_ratio

    if density >= 0.8:  # ≤20% pruning
        return 4, 5  # 80% dense
    elif density >= 0.75:  # ~25% pruning
        return 3, 4  # 75% dense
    elif density >= 0.5:  # ~50% pruning
        return 2, 4  # 50% dense
    elif density >= 0.4:  # ~60% pruning
        return 2, 5  # 40% dense
    else:  # High pruning
        return 1, 4  # 25% dense
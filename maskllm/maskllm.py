import torch
import torch.nn as nn
import torch.nn.functional as F

import itertools

def generate_N_M_masks(N, M):
    # Create all possible binary combinations N:M sparse masks
    combinations = list(itertools.combinations(range(M), N))
    # Create a tensor to store the result
    result = torch.zeros((len(combinations), M), dtype=torch.float32)
    # Fill in the ones according to the combinations
    for i, indices in enumerate(combinations):
        result[i, torch.tensor(indices)] = 1
    return result

# class MaskedLinearFrozen(nn.Linear):
#     """A linear layer with a fixed mask that is not updated during training."""
#     def __init__(self, in_features, out_features, bias=True):
#         super(MaskedLinearFrozen, self).__init__(in_features, out_features, bias)
#         self.register_buffer('mask', torch.ones(out_features, in_features))
    
#     def __repr__(self):
#         return f"{self.__class__.__name__}({self.in_features}, {self.out_features}, bias={self.bias is not None})"

#     def forward(self, x):
#         return F.linear(x, self.mask * self.weight, self.bias)


class MaskedLinearFrozen(nn.Linear):
    """A linear layer with a fixed mask that is not updated during training."""  
    def __init__(self, in_features, out_features, bias=True, N=2, M=4):  
        super(MaskedLinearFrozen, self).__init__(in_features, out_features, bias)  
        self.N = N  
        self.M = M  
          
        # Generate N:M mask options  
        self._mask_options = generate_N_M_masks(N, M)  
          
        # Use only the divisible portion (same logic as MaskedLinear)  
        total_elements = self.weight.numel()  
        self.divisible_size = (total_elements // M) * M  
        self.num_blocks = self.divisible_size // M  
          
        # Initialize mask with default N:M pattern (all ones initially)  
        self.register_buffer('mask', torch.ones(out_features, in_features, dtype=torch.float32))  
      
    def __repr__(self):  
        return f"{self.__class__.__name__}({self.in_features}, {self.out_features}, bias={self.bias is not None}, N={self.N}, M={self.M})"  
  
    def forward(self, x):  
        return F.linear(x, self.mask * self.weight, self.bias)  
      
    def set_mask_pattern(self, pattern_idx=0):  
        """Set the mask to a specific N:M pattern"""  
        # Select the pattern and reshape it  
        selected_pattern = self._mask_options[pattern_idx]  
          
        # Create full mask with unmasked remainder  
        full_mask = torch.ones(self.weight.numel(), device=self.weight.device)  
        full_mask[:self.divisible_size] = selected_pattern.repeat(self.num_blocks)  
        self.mask = full_mask.view(self.out_features, self.in_features)


class MaskedLinear(nn.Linear):
    """A linear layer with a learnable mask that sparsifies the weights."""
    def __init__(self, in_features, out_features, bias=True, N=2, M=4, gate_init_std=0.2, tau=1, hard=False, scaling=1):  
        super(MaskedLinear, self).__init__(in_features, out_features, bias)  
        self._mask_options = generate_N_M_masks(N, M)  
        
        # Use only the divisible portion  
        total_elements = self.weight.numel()  
        self.divisible_size = (total_elements // M) * M  
        self.num_blocks = self.divisible_size // M  
        
        self.gate = nn.Parameter(torch.empty(  
                self.num_blocks, self._mask_options.size(0),   
                device=self.weight.device, dtype=self.weight.dtype), requires_grad=True) 
        self.tau = 1
        self.scaling = scaling
        self.hard = hard
        self.register_buffer('mask', torch.ones((out_features, in_features), dtype=torch.float32))
        self.mask_oudated = False
        self.N = N
        self.M = M 

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_features}, {self.out_features}, bias={self.bias is not None}, N={self.N}, M={self.M}, tau={self.tau}, scaling={self.scaling}, hard={self.hard})"

    def sparse_weight_reg(self):
        return self._sparse_weight_reg

    # def forward(self, x):
    #     if self.training:
    #         self.mask_oudated = True # reset selected since we will update it
    #         soft_index = F.gumbel_softmax(self.gate * self.scaling, tau=self.tau, hard=self.hard, dim=1) # (Blocks x Candidate Masks)
    #         soft_mask = soft_index @ self._mask_options.to(x.device) # (Blocks x Candidate Masks) @ (Candidate Masks x M) = (Blocks x M)
    #         soft_mask = soft_mask.view(self.out_features, self.in_features)
    #         self._sparse_weight_reg = (self.weight.detach() * soft_mask).pow(2).sum()
    #         return F.linear(x, soft_mask * self.weight, self.bias)
    #     else:
    #         if self.mask_oudated: # for inference, we only compute the winner masks once for efficiency
    #             self._mask_options = self._mask_options.to(x.device)
    #             self.mask = self._mask_options[torch.argmax(self.gate, dim=1)].view(self.out_features, self.in_features)
    #             self.mask_oudated = False
    #         return F.linear(x, self.mask * self.weight, self.bias)

    def forward(self, x):  
        if self.training:  
            self.mask_oudated = True  
            soft_index = F.gumbel_softmax(self.gate * self.scaling, tau=self.tau, hard=self.hard, dim=1)  
            soft_mask = soft_index @ self._mask_options.to(x.device)  
            soft_mask = soft_mask.view(self.num_blocks, self.M)  
              
            # Create full mask with unmasked remainder  
            full_mask = torch.ones(self.weight.numel(), device=self.weight.device)  
            full_mask[:self.divisible_size] = soft_mask.flatten()  
            soft_mask = full_mask.view(self.out_features, self.in_features)  
              
            self._sparse_weight_reg = (self.weight.detach() * soft_mask).pow(2).sum()  
            return F.linear(x, soft_mask * self.weight, self.bias)  
        else:  
            if self.mask_oudated:  
                self._mask_options = self._mask_options.to(x.device)  
                mask_indices = torch.argmax(self.gate, dim=1)  
                selected_masks = self._mask_options[mask_indices]  
                  
                # Handle truncation in inference  
                full_mask = torch.ones(self.weight.numel(), device=self.weight.device)  
                full_mask[:self.divisible_size] = selected_masks.flatten()  
                self.mask = full_mask.view(self.out_features, self.in_features)  
                self.mask_oudated = False  
            return F.linear(x, self.mask * self.weight, self.bias)

    # def load_mask_prior(self, prior_strength=3):
    #     with torch.no_grad():
    #         sparsity = (self.mask==0).sum().item() / self.mask.numel()
    #         # prior will be the inner product the different candidates to the prior mask
    #         priors = (self._mask_options.unsqueeze(0) * self.mask.view(-1, 1, 4)).sum(dim=2) # (1, Candidate Masks, M) * (Blocks, 1, M) => Blocks x Candidate Masks
    #         self.gate.data += (priors-self.N//2) * self.gate.std() * prior_strength

    #         if torch.distributed.get_rank() == 0:
    #             print(f"initializing with prior (strength={prior_strength}), Prior Sparsity: {sparsity}")
    #             print(f"mean: {self.gate.mean().item()}, std: {self.gate.std().item()}, max: {self.gate.max().item()}, min: {self.gate.min().item()}")

    def load_mask_prior(self, prior_strength=3):  
        with torch.no_grad():  
            sparsity = (self.mask==0).sum().item() / self.mask.numel()  
            # Only process divisible portion for prior  
            prior_mask_flat = self.mask.view(-1)[:self.divisible_size].view(-1, self.M)  
            priors = (self._mask_options.unsqueeze(0) * prior_mask_flat).sum(dim=2)  
            self.gate.data += (priors-self.N//2) * self.gate.std() * prior_strength  
  
            if torch.distributed.get_rank() == 0:  
                print(f"initializing with prior (strength={prior_strength}), Prior Sparsity: {sparsity}")  
                print(f"mean: {self.gate.mean().item()}, std: {self.gate.std().item()}, max: {self.gate.max().item()}, min: {self.gate.min().item()}")

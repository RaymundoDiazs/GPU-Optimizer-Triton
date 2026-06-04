import triton
import triton.language as tl
import torch
import math
from typing import List, Tuple, Optional, Any


# ============================================================================
# Triton Kernels
# ============================================================================

@triton.jit
def adam_kernel(
    param_ptr,
    grad_ptr,
    exp_avg_ptr,
    exp_avg_sq_ptr,
    max_exp_avg_sq_ptr,
    lr,
    beta1,
    beta2,
    eps,
    weight_decay,
    step,
    amsgrad,
    maximize,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for Adam optimization step.
    
    Args:
        param_ptr: Pointer to parameter tensor
        grad_ptr: Pointer to gradient tensor
        exp_avg_ptr: Pointer to first moment estimate (momentum)
        exp_avg_sq_ptr: Pointer to second moment estimate (velocity)
        max_exp_avg_sq_ptr: Pointer to max second moment (for AMSGrad)
        lr: Learning rate
        beta1: Exponential decay rate for first moment
        beta2: Exponential decay rate for second moment
        eps: Small constant for numerical stability
        weight_decay: Weight decay coefficient
        step: Current optimization step
        amsgrad: Whether to use AMSGrad variant
        maximize: Whether to maximize the objective
        n_elements: Total number of elements
        BLOCK_SIZE: Block size for parallel processing
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask for valid elements
    mask = offsets < n_elements
    
    # Load tensors
    param = tl.load(param_ptr + offsets, mask=mask)
    grad = tl.load(grad_ptr + offsets, mask=mask)
    exp_avg = tl.load(exp_avg_ptr + offsets, mask=mask)
    exp_avg_sq = tl.load(exp_avg_sq_ptr + offsets, mask=mask)
    
    # Apply weight decay (L2 regularization)
    if weight_decay != 0:
        grad = grad + weight_decay * param
    
    # Negate gradient if maximizing
    if maximize:
        grad = -grad
    
    # Update biased first moment estimate
    exp_avg = beta1 * exp_avg + (1 - beta1) * grad
    
    # Update biased second raw moment estimate
    exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad * grad
    
    # Bias correction
    bias_correction1 = 1 - beta1 ** step
    bias_correction2 = 1 - beta2 ** step
    
    # Compute bias-corrected estimates
    bias_corrected_exp_avg = exp_avg / bias_correction1
    bias_corrected_exp_avg_sq = exp_avg_sq / bias_correction2
    
    # AMSGrad variant: use max of second moment
    if amsgrad:
        max_exp_avg_sq = tl.load(max_exp_avg_sq_ptr + offsets, mask=mask)
        max_exp_avg_sq = tl.maximum(max_exp_avg_sq, exp_avg_sq)
        denom = tl.sqrt(max_exp_avg_sq) + eps
        tl.store(max_exp_avg_sq_ptr + offsets, max_exp_avg_sq, mask=mask)
    else:
        denom = tl.sqrt(bias_corrected_exp_avg_sq) + eps
    
    # Update parameters
    param = param - lr * bias_corrected_exp_avg / denom
    
    # Store updated values
    tl.store(param_ptr + offsets, param, mask=mask)
    tl.store(exp_avg_ptr + offsets, exp_avg, mask=mask)
    tl.store(exp_avg_sq_ptr + offsets, exp_avg_sq, mask=mask)


# ============================================================================
# Wrapper Function
# ============================================================================

class AdamOptimizer(torch.optim.Optimizer):
    """
    Implements Adam algorithm with Triton acceleration.
    
    Arguments:
        params: iterable of parameters to optimize or dicts defining parameter groups
        lr: learning rate (default: 1e-3)
        betas: coefficients used for computing running averages of gradient 
               and its square (default: (0.9, 0.999))
        eps: term added to the denominator to improve numerical stability (default: 1e-8)
        weight_decay: weight decay (L2 penalty) (default: 0)
        amsgrad: whether to use the AMSGrad variant (default: False)
        foreach: whether to use foreach implementation (default: None)
        maximize: whether to maximize the objective function (default: False)
        capturable: whether to use capturable kernels (default: False)
        differentiable: whether to enable gradient computation through optimizer (default: False)
        fused: whether to use fused kernel (default: None)
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        amsgrad: bool = False,
        foreach: Optional[bool] = None,
        maximize: bool = False,
        capturable: bool = False,
        differentiable: bool = False,
        fused: Optional[bool] = None,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            foreach=foreach,
            maximize=maximize,
            capturable=capturable,
            differentiable=differentiable,
            fused=fused,
        )
        super(AdamOptimizer, self).__init__(params, defaults)
    
    def __setstate__(self, state):
        super(AdamOptimizer, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            group.setdefault('maximize', False)
            group.setdefault('foreach', None)
            group.setdefault('capturable', False)
            group.setdefault('differentiable', False)
            group.setdefault('fused', None)
    
    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.
        
        Arguments:
            closure: a closure that reevaluates the model and returns the loss
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                amsgrad = group['amsgrad']
                maximize = group['maximize']
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                
                state['step'] += 1
                
                # Get hyperparameters
                lr = group['lr']
                beta1, beta2 = group['betas']
                eps = group['eps']
                weight_decay = group['weight_decay']
                step = state['step']
                
                # Get state tensors
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                max_exp_avg_sq = state.get('max_exp_avg_sq', None)
                
                # Determine block size
                BLOCK_SIZE = 1024
                n_elements = p.numel()
                grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
                
                # Prepare max_exp_avg_sq pointer
                max_exp_avg_sq_ptr = max_exp_avg_sq.data_ptr() if amsgrad else 0
                
                # Launch Triton kernel
                adam_kernel[grid](
                    p.data,
                    grad,
                    exp_avg,
                    exp_avg_sq,
                    max_exp_avg_sq_ptr,
                    lr,
                    beta1,
                    beta2,
                    eps,
                    weight_decay,
                    float(step),
                    amsgrad,
                    maximize,
                    n_elements,
                    BLOCK_SIZE=BLOCK_SIZE,
                )
        
        return loss


def Adam(
    params,
    lr: float = 1e-3,
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0,
    amsgrad: bool = False,
    foreach: Optional[bool] = None,
    maximize: bool = False,
    capturable: bool = False,
    differentiable: bool = False,
    fused: Optional[bool] = None,
) -> AdamOptimizer:
    """
    Wrapper function for Adam optimizer with Triton acceleration.
    
    Returns:
        AdamOptimizer: Configured Adam optimizer instance
    """
    return AdamOptimizer(
        params,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        amsgrad=amsgrad,
        foreach=foreach,
        maximize=maximize,
        capturable=capturable,
        differentiable=differentiable,
        fused=fused,
    )

##################################################################################################################################################



import torch

# def Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0):
#     return torch.optim.Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

def test_Adam():
    results = {}

    # Test Case 1: Default parameters
    params1 = [torch.randn(2, 2, device='cuda', requires_grad=True)]
    optimizer1 = Adam(params1)
    results["test_case_1"] = optimizer1.defaults

    # Test Case 2: Custom learning rate
    params2 = [torch.randn(2, 2, device='cuda', requires_grad=True)]
    optimizer2 = Adam(params2, lr=0.01)
    results["test_case_2"] = optimizer2.defaults

    # Test Case 3: Custom betas
    params3 = [torch.randn(2, 2, device='cuda', requires_grad=True)]
    optimizer3 = Adam(params3, betas=(0.85, 0.95))
    results["test_case_3"] = optimizer3.defaults

    # Test Case 4: Custom weight decay
    params4 = [torch.randn(2, 2, device='cuda', requires_grad=True)]
    optimizer4 = Adam(params4, weight_decay=0.01)
    results["test_case_4"] = optimizer4.defaults

    return results

test_results = test_Adam()

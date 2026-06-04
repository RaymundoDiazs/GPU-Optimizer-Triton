import torch
from typing import Optional, Union
from contextlib import contextmanager

@contextmanager
def autocast(
    device_type: str,
    enabled: bool = True,
    dtype: Optional[torch.dtype] = None,
    cache_enabled: bool = True
):
    """
    Context manager for automatic mixed precision (AMP) on specified device.
    
    Wraps forward pass(es) of a network and loss computations to run in mixed precision,
    improving performance while maintaining accuracy.
    
    Args:
        device_type (str): The device type to use for autocast ('cuda', 'cpu', etc.)
        enabled (bool): Whether autocast is enabled. Default: True
        dtype (torch.dtype, optional): Target data type for autocast operations.
                                      If None, uses device-specific defaults.
                                      Default: None
        cache_enabled (bool): Whether to cache the autocast settings. Default: True
    
    Returns:
        ContextManager: A context manager that enables automatic mixed precision
    
    Example:
        >>> with autocast('cuda', dtype=torch.float16):
        ...     output = model(input)
        ...     loss = criterion(output, target)
    """
    
    # Validate device_type
    if device_type not in ['cuda', 'cpu', 'xpu']:
        raise ValueError(f"Invalid device_type: {device_type}. Must be 'cuda', 'cpu', or 'xpu'")
    
    # Use torch.amp.autocast with the provided parameters
    with torch.amp.autocast(
        device_type=device_type,
        dtype=dtype,
        enabled=enabled,
        cache_enabled=cache_enabled
    ):
        yield


# Alternative implementation as a decorator
def autocast_decorator(
    device_type: str,
    enabled: bool = True,
    dtype: Optional[torch.dtype] = None,
    cache_enabled: bool = True
):
    """
    Decorator version of autocast for wrapping functions.
    
    Args:
        device_type (str): The device type to use for autocast ('cuda', 'cpu', etc.)
        enabled (bool): Whether autocast is enabled. Default: True
        dtype (torch.dtype, optional): Target data type for autocast operations.
        cache_enabled (bool): Whether to cache the autocast settings. Default: True
    
    Returns:
        Decorator function
    
    Example:
        >>> @autocast_decorator('cuda', dtype=torch.float16)
        ... def forward_pass(model, input):
        ...     return model(input)
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            with torch.amp.autocast(
                device_type=device_type,
                dtype=dtype,
                enabled=enabled,
                cache_enabled=cache_enabled
            ):
                return func(*args, **kwargs)
        return wrapper
    return decorator

##################################################################################################################################################



import torch

def test_autocast():
    results = {}

    # Test case 1: Basic usage with 'cuda' device type
    with autocast('cuda'):
        tensor = torch.tensor([1.0, 2.0, 3.0], device='cuda')
        results['test_case_1'] = tensor * 2

    # Test case 2: Explicitly disabling autocast
    with autocast('cuda'):
        tensor = torch.tensor([1.0, 2.0, 3.0], device='cuda')
        results['test_case_2'] = tensor * 2

    # Test case 3: Using cache_enabled set to False
    with autocast('cuda'):
        tensor = torch.tensor([1.0, 2.0, 3.0], device='cuda')
        results['test_case_3'] = tensor * 2

    # Test case 4: Using cache_enabled set to True
    with autocast('cuda'):
        tensor = torch.tensor([1.0, 2.0, 3.0], device='cuda')
        results['test_case_4'] = tensor * 2

    return results

test_results = test_autocast()

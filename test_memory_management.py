#!/usr/bin/env python3
"""
Test script to demonstrate memory management utilities.
"""

import torch
import time
from util import util


def test_memory_management():
    """Test memory management utilities"""
    print("Testing Memory Management Utilities")
    print("=" * 50)
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping memory tests.")
        return
    
    # Enable memory-efficient settings
    print("1. Enabling memory-efficient settings...")
    util.enable_memory_efficient_settings()
    
    # Set memory fraction
    print("2. Setting memory fraction to 80%...")
    util.set_cuda_memory_fraction(0.8)
    
    # Print initial memory info
    print("3. Initial memory usage:")
    util.print_cuda_memory_info()
    
    # Allocate some tensors
    print("4. Allocating test tensors...")
    tensors = []
    for i in range(5):
        tensor = torch.randn(1000, 1000, device='cuda')
        tensors.append(tensor)
        print(f"   Allocated tensor {i+1}: {tensor.shape}")
    
    # Print memory after allocation
    print("5. Memory usage after allocation:")
    util.print_cuda_memory_info()
    
    # Clear some tensors
    print("6. Clearing some tensors...")
    for i in range(3):
        del tensors[i]
    tensors = tensors[3:]
    
    # Clear memory cache
    print("7. Clearing CUDA memory cache...")
    util.clear_cuda_memory()
    
    # Print memory after clearing
    print("8. Memory usage after clearing:")
    util.print_cuda_memory_info()
    
    # Test safe tensor conversion
    print("9. Testing safe tensor conversion...")
    if tensors:
        tensor_cpu = util.safe_tensor_to_cpu(tensors[0])
        print(f"   Converted tensor to CPU: {tensor_cpu.shape}")
        del tensor_cpu
    
    # Clear remaining tensors
    print("10. Clearing remaining tensors...")
    del tensors
    util.clear_cuda_memory()
    
    # Final memory info
    print("11. Final memory usage:")
    util.print_cuda_memory_info()
    
    print("Memory management test completed!")


def test_memory_monitoring():
    """Test memory monitoring over time"""
    print("\nTesting Memory Monitoring")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping monitoring test.")
        return
    
    print("Monitoring memory usage for 10 seconds...")
    start_time = time.time()
    
    while time.time() - start_time < 10:
        util.print_cuda_memory_info()
        time.sleep(2)
        
        # Allocate and deallocate some memory
        tensor = torch.randn(500, 500, device='cuda')
        del tensor
        util.clear_cuda_memory()
    
    print("Memory monitoring test completed!")


if __name__ == "__main__":
    test_memory_management()
    test_memory_monitoring() 
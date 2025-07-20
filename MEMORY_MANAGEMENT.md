# PyTorch CUDA Memory Management Guide

This guide explains how to handle PyTorch CUDA memory issues and provides tools to manage GPU memory effectively.

## Why PyTorch Doesn't Release Memory

PyTorch may not immediately release GPU memory for several reasons:

1. **CUDA Memory Fragmentation**: The CUDA memory allocator can become fragmented, leaving small chunks that can't be reused efficiently.

2. **Memory Pooling**: PyTorch maintains memory pools to avoid repeated allocations/deallocations, but these pools may not be released immediately.

3. **Cached Memory**: PyTorch caches some memory for future use, which may not be immediately freed.

4. **Driver-Level Memory**: Some memory may be held at the CUDA driver level and not immediately released.

## Solutions Implemented

### 1. Memory Management Utilities

The codebase now includes memory management functions in `util/util.py`:

- `clear_cuda_memory()`: Clears CUDA cache and garbage collects
- `get_cuda_memory_info()`: Gets detailed memory usage information
- `print_cuda_memory_info()`: Prints current memory usage
- `safe_tensor_to_cpu()`: Safely moves tensors to CPU
- `cleanup_model_memory()`: Cleans up model memory

### 2. Enhanced Training Loop

The training script (`train.py`) now includes:

- Memory monitoring at regular intervals
- Automatic memory clearing after model saves
- Memory cleanup at the end of each epoch
- Final memory cleanup when training completes

### 3. Improved Model Saving

The base model (`models/base_model.py`) now:

- Properly handles GPU memory when saving models
- Clears temporary CPU tensors after saving
- Moves models back to GPU efficiently

### 4. Standalone Memory Management Script

Use `memory_utils.py` for manual memory management:

```bash
# Show current memory usage
python memory_utils.py info

# Clear memory cache
python memory_utils.py clear

# Reset memory and peak stats
python memory_utils.py reset

# Set memory fraction (e.g., 80%)
python memory_utils.py fraction 0.8

# Enable memory-efficient settings
python memory_utils.py enable

# Kill all CUDA processes (use with caution!)
python memory_utils.py kill
```

## Environment Variables

Set these environment variables for better memory management:

```bash
# Enable expandable segments to reduce fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Limit memory usage to 80% of GPU
export CUDA_VISIBLE_DEVICES=0
```

## Best Practices

### 1. Regular Memory Monitoring

Monitor memory usage during training:

```python
from util import util

# Print memory usage
util.print_cuda_memory_info()

# Clear memory when needed
util.clear_cuda_memory()
```

### 2. Proper Tensor Management

```python
# Move tensors to CPU when not needed
tensor_cpu = tensor.detach().cpu()
del tensor  # Delete GPU tensor

# Use safe tensor conversion
tensor_cpu = util.safe_tensor_to_cpu(tensor)
```

### 3. Model Cleanup

```python
# Clean up model memory
util.cleanup_model_memory(model)
```

### 4. Batch Size Management

If you encounter memory issues:

1. Reduce batch size
2. Use gradient accumulation
3. Enable mixed precision training
4. Use memory-efficient attention if available

## Troubleshooting

### Immediate Solutions

1. **Clear Memory Cache**:
   ```python
   torch.cuda.empty_cache()
   gc.collect()
   ```

2. **Reset Peak Memory Stats**:
   ```python
   torch.cuda.reset_peak_memory_stats()
   ```

3. **Kill CUDA Processes** (if necessary):
   ```bash
   nvidia-smi
   kill -9 <PID>
   ```

### Long-term Solutions

1. **Use Memory-Efficient Settings**:
   ```python
   os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
   ```

2. **Limit Memory Usage**:
   ```python
   torch.cuda.set_per_process_memory_fraction(0.8)
   ```

3. **Enable Mixed Precision**:
   ```python
   from torch.cuda.amp import autocast, GradScaler
   ```

## Monitoring Tools

### NVIDIA-SMI

Monitor GPU usage in real-time:

```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Get detailed memory info
nvidia-smi --query-gpu=memory.used,memory.total,memory.free --format=csv
```

### PyTorch Memory Profiling

```python
# Track memory allocations
torch.cuda.memory_summary()

# Get memory stats
print(torch.cuda.memory_stats())
```

## Common Error Messages

### "CUDA out of memory"

**Causes**:
- Batch size too large
- Model too large for GPU
- Memory fragmentation
- Memory leaks

**Solutions**:
1. Reduce batch size
2. Clear memory cache
3. Use gradient accumulation
4. Enable memory-efficient settings

### "Memory fragmentation"

**Causes**:
- Long-running processes
- Frequent allocations/deallocations
- Large tensor operations

**Solutions**:
1. Use `expandable_segments:True`
2. Clear cache regularly
3. Restart process if necessary

## Example Usage

```python
import torch
from util import util

# Enable memory-efficient settings
util.enable_memory_efficient_settings()

# Monitor memory before training
util.print_cuda_memory_info()

# Your training loop here
for epoch in range(num_epochs):
    for batch in dataloader:
        # Training code...
        
        # Clear memory periodically
        if iteration % 100 == 0:
            util.clear_cuda_memory()
    
    # Clear memory after each epoch
    util.clear_cuda_memory()

# Final cleanup
util.cleanup_model_memory(model)
util.print_cuda_memory_info()
```

## Additional Resources

- [PyTorch Memory Management Documentation](https://pytorch.org/docs/stable/notes/cuda.html#memory-management)
- [NVIDIA CUDA Memory Management](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-management)
- [PyTorch Memory Profiling](https://pytorch.org/docs/stable/notes/cuda.html#memory-profiling) 
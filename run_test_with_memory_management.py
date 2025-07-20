#!/usr/bin/env python3
"""
Script to run tests with automatic memory management and image size adjustment.
"""

import subprocess
import sys
import os
from util import util


def run_test_with_memory_management(dataroot, name, model='cut', gpu_ids='0', 
                                   load_size=1024, num_test=50, epoch=155,
                                   cuda_memory_fraction=0.8, memory_clear_freq=5,
                                   **kwargs):
    """
    Run test with memory management and automatic image size adjustment.
    
    Args:
        dataroot: Path to the dataset
        name: Experiment name
        model: Model type (default: 'cut')
        gpu_ids: GPU IDs to use
        load_size: Initial image size to try
        num_test: Number of test images
        epoch: Model epoch to test
        cuda_memory_fraction: Fraction of GPU memory to use
        memory_clear_freq: Frequency of memory clearing
        **kwargs: Additional arguments to pass to test.py
    """
    
    # Clear memory first
    print("Clearing initial memory...")
    util.clear_cuda_memory()
    util.print_cuda_memory_info()
    
    # Check if the requested image size is safe
    image_size = (load_size, load_size)
    if not util.check_memory_before_processing(image_size):
        print(f"Warning: {load_size}x{load_size} images may exceed available memory.")
        safe_size = util.get_safe_image_size()
        print(f"Recommended safe image size: {safe_size[0]}x{safe_size[1]}")
        
        response = input(f"Continue with {load_size}x{load_size} or use safe size {safe_size[0]}x{safe_size[1]}? (c/s): ")
        if response.lower() == 's':
            load_size = safe_size[0]
            print(f"Using safe image size: {load_size}x{load_size}")
    
    # Build the command
    cmd = [
        'python3', 'test.py',
        '--dataroot', dataroot,
        '--name', name,
        '--model', model,
        '--gpu_ids', gpu_ids,
        '--load_size', str(load_size),
        '--num_test', str(num_test),
        '--epoch', str(epoch),
        '--cuda_memory_fraction', str(cuda_memory_fraction),
        '--memory_clear_freq', str(memory_clear_freq),
        '--eval'
    ]
    
    # Add additional arguments
    for key, value in kwargs.items():
        if value is not None:
            if isinstance(value, bool):
                if value:
                    cmd.append(f'--{key}')
            else:
                cmd.extend([f'--{key}', str(value)])
    
    print("Running command:")
    print(' '.join(cmd))
    print()
    
    try:
        # Run the test
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("Test completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Test failed with error code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("Test interrupted by user")
        return False


def main():
    """Main function for command-line usage"""
    if len(sys.argv) < 3:
        print("Usage: python run_test_with_memory_management.py <dataroot> <name> [options]")
        print()
        print("Required arguments:")
        print("  dataroot    Path to the dataset")
        print("  name        Experiment name")
        print()
        print("Optional arguments:")
        print("  --model MODEL              Model type (default: cut)")
        print("  --gpu_ids GPU_IDS          GPU IDs to use (default: 0)")
        print("  --load_size SIZE           Image size (default: 1024)")
        print("  --num_test NUM             Number of test images (default: 50)")
        print("  --epoch EPOCH              Model epoch to test (default: 155)")
        print("  --cuda_memory_fraction F   GPU memory fraction (default: 0.8)")
        print("  --memory_clear_freq F      Memory clear frequency (default: 5)")
        print("  --preprocess PREPROCESS    Preprocessing method")
        print("  --yuv                      Use YUV color space")
        return
    
    # Parse arguments
    dataroot = sys.argv[1]
    name = sys.argv[2]
    
    # Default values
    kwargs = {
        'model': 'cut',
        'gpu_ids': '0',
        'load_size': 1024,
        'num_test': 50,
        'epoch': 155,
        'cuda_memory_fraction': 0.8,
        'memory_clear_freq': 5,
        'preprocess': None,
        'yuv': False
    }
    
    # Parse additional arguments
    i = 3
    while i < len(sys.argv):
        if sys.argv[i].startswith('--'):
            key = sys.argv[i][2:]  # Remove '--'
            if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith('--'):
                value = sys.argv[i + 1]
                if key in ['load_size', 'num_test', 'epoch', 'memory_clear_freq']:
                    kwargs[key] = int(value)
                elif key in ['cuda_memory_fraction']:
                    kwargs[key] = float(value)
                else:
                    kwargs[key] = value
                i += 2
            else:
                # Boolean flag
                kwargs[key] = True
                i += 1
        else:
            i += 1
    
    # Run the test
    success = run_test_with_memory_management(dataroot, name, **kwargs)
    
    if success:
        print("Test completed successfully!")
    else:
        print("Test failed!")
        sys.exit(1)


if __name__ == "__main__":
    main() 
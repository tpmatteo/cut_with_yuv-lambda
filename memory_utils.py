#!/usr/bin/env python3
"""
Memory management utilities for PyTorch CUDA memory issues.
This script provides tools to monitor and clear GPU memory.
"""

import torch
import gc
import os
import subprocess
import sys


def clear_cuda_memory():
    """Clear CUDA memory cache and garbage collect to free up memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    print("CUDA memory cache cleared.")


def get_cuda_memory_info():
    """Get detailed CUDA memory information"""
    if not torch.cuda.is_available():
        return "CUDA not available"
    
    memory_info = {}
    memory_info['allocated'] = torch.cuda.memory_allocated() / 1024**3  # GB
    memory_info['cached'] = torch.cuda.memory_reserved() / 1024**3  # GB
    memory_info['max_allocated'] = torch.cuda.max_memory_allocated() / 1024**3  # GB
    memory_info['max_cached'] = torch.cuda.max_memory_reserved() / 1024**3  # GB
    
    # Get total GPU memory
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
    memory_info['total'] = total_memory
    memory_info['free'] = total_memory - memory_info['cached']
    
    return memory_info


def print_cuda_memory_info():
    """Print current CUDA memory usage"""
    memory_info = get_cuda_memory_info()
    if isinstance(memory_info, str):
        print(memory_info)
        return
    
    print("CUDA Memory Usage:")
    print(f"  Allocated: {memory_info['allocated']:.2f} GB")
    print(f"  Cached: {memory_info['cached']:.2f} GB")
    print(f"  Free: {memory_info['free']:.2f} GB")
    print(f"  Total: {memory_info['total']:.2f} GB")
    print(f"  Max Allocated: {memory_info['max_allocated']:.2f} GB")
    print(f"  Max Cached: {memory_info['max_cached']:.2f} GB")


def reset_cuda_memory():
    """Reset CUDA memory by clearing cache and resetting peak memory stats"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    gc.collect()
    print("CUDA memory reset complete.")


def kill_cuda_processes():
    """Kill all CUDA processes (use with caution!)"""
    try:
        # Get all processes using CUDA
        result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        if result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                if pid.strip():
                    try:
                        subprocess.run(['kill', '-9', pid.strip()], check=True)
                        print(f"Killed process {pid.strip()}")
                    except subprocess.CalledProcessError:
                        print(f"Failed to kill process {pid.strip()}")
        else:
            print("No CUDA processes found.")
    except FileNotFoundError:
        print("nvidia-smi not found. Cannot kill CUDA processes.")
    except Exception as e:
        print(f"Error killing CUDA processes: {e}")


def set_cuda_memory_fraction(fraction=0.8):
    """Set CUDA memory fraction to limit memory usage"""
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(fraction)
        print(f"CUDA memory fraction set to {fraction}")


def enable_memory_efficient_settings():
    """Enable memory-efficient PyTorch settings"""
    # Set environment variables for memory efficiency
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Enable memory efficient attention if available
    if hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda, 'enable_flash_sdp'):
        torch.backends.cuda.enable_flash_sdp(True)
    
    print("Memory-efficient settings enabled.")


def main():
    """Main function for command-line usage"""
    if len(sys.argv) < 2:
        print("Usage: python memory_utils.py [command]")
        print("Commands:")
        print("  info     - Show current CUDA memory usage")
        print("  clear    - Clear CUDA memory cache")
        print("  reset    - Reset CUDA memory and peak stats")
        print("  kill     - Kill all CUDA processes (use with caution!)")
        print("  fraction [0.0-1.0] - Set CUDA memory fraction")
        print("  enable   - Enable memory-efficient settings")
        return
    
    command = sys.argv[1]
    
    if command == "info":
        print_cuda_memory_info()
    elif command == "clear":
        clear_cuda_memory()
        print_cuda_memory_info()
    elif command == "reset":
        reset_cuda_memory()
        print_cuda_memory_info()
    elif command == "kill":
        response = input("This will kill all CUDA processes. Are you sure? (y/N): ")
        if response.lower() == 'y':
            kill_cuda_processes()
        else:
            print("Operation cancelled.")
    elif command == "fraction":
        if len(sys.argv) < 3:
            print("Please provide a fraction value (0.0-1.0)")
            return
        try:
            fraction = float(sys.argv[2])
            if 0.0 <= fraction <= 1.0:
                set_cuda_memory_fraction(fraction)
            else:
                print("Fraction must be between 0.0 and 1.0")
        except ValueError:
            print("Invalid fraction value")
    elif command == "enable":
        enable_memory_efficient_settings()
    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main() 
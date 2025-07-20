"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import importlib
import argparse
from argparse import Namespace
import torchvision
import gc


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def copyconf(default_opt, **kwargs):
    conf = Namespace(**vars(default_opt))
    for key in kwargs:
        setattr(conf, key, kwargs[key])
    return conf


def find_class_in_module(target_cls_name, module):
    target_cls_name = target_cls_name.replace('_', '').lower()
    clslib = importlib.import_module(module)
    cls = None
    for name, clsobj in clslib.__dict__.items():
        if name.lower() == target_cls_name:
            cls = clsobj

    assert cls is not None, "In %s, there should be a class whose name matches %s in lowercase without underscore(_)" % (module, target_cls_name)

    return cls


def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].clamp(-1.0, 1.0).cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio is None:
        pass
    elif aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    elif aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def correct_resize_label(t, size):
    device = t.device
    t = t.detach().cpu()
    resized = []
    for i in range(t.size(0)):
        one_t = t[i, :1]
        one_np = np.transpose(one_t.numpy().astype(np.uint8), (1, 2, 0))
        one_np = one_np[:, :, 0]
        one_image = Image.fromarray(one_np).resize(size, Image.NEAREST)
        resized_t = torch.from_numpy(np.array(one_image)).long()
        resized.append(resized_t)
    return torch.stack(resized, dim=0).to(device)


def correct_resize(t, size, mode=Image.BICUBIC):
    device = t.device
    t = t.detach().cpu()
    resized = []
    for i in range(t.size(0)):
        one_t = t[i:i + 1]
        one_image = Image.fromarray(tensor2im(one_t)).resize(size, Image.BICUBIC)
        resized_t = torchvision.transforms.functional.to_tensor(one_image) * 2 - 1.0
        resized.append(resized_t)
    return torch.stack(resized, dim=0).to(device)


def rgb_to_yuv(img):
    """
    Convert a torch tensor image (B,C,H,W) or (C,H,W) from RGB to YUV.
    Assumes input in [-1, 1] range.
    Output is also in [-1, 1] range.
    """
    original_shape = img.shape
    if img.dim() == 3:
        img = img.unsqueeze(0)
    r, g, b = img[:, 0:1, :, :], img[:, 1:2, :, :], img[:, 2:3, :, :]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = -0.14713 * r - 0.28886 * g + 0.436 * b
    v = 0.615 * r - 0.51499 * g - 0.10001 * b
    yuv = torch.cat([y, u, v], dim=1)
    # Preserve the original shape
    if len(original_shape) == 3:
        yuv = yuv.squeeze(0)
    return yuv


def yuv_to_rgb(img):
    """
    Convert a torch tensor image (B,C,H,W) or (C,H,W) from YUV to RGB.
    Assumes input in [-1, 1] range.
    Output is also in [-1, 1] range.
    """
    original_shape = img.shape
    if img.dim() == 3:
        img = img.unsqueeze(0)
    y, u, v = img[:, 0:1, :, :], img[:, 1:2, :, :], img[:, 2:3, :, :]
    r = y + 1.13983 * v
    g = y - 0.39465 * u - 0.58060 * v
    b = y + 2.03211 * u
    rgb = torch.cat([r, g, b], dim=1)
    rgb = torch.clamp(rgb, -1, 1)
    # Preserve the original shape
    if len(original_shape) == 3:
        rgb = rgb.squeeze(0)
    return rgb


def clear_cuda_memory():
    """Clear CUDA memory cache and garbage collect to free up memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


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


def safe_tensor_to_cpu(tensor):
    """Safely move tensor to CPU and clear GPU memory"""
    if tensor is None:
        return None
    
    if tensor.is_cuda:
        tensor_cpu = tensor.detach().cpu()
        del tensor
        return tensor_cpu
    return tensor


def cleanup_model_memory(model):
    """Clean up model memory by moving to CPU and clearing cache"""
    if hasattr(model, 'model_names'):
        for name in model.model_names:
            if isinstance(name, str):
                net = getattr(model, 'net' + name, None)
                if net is not None and hasattr(net, 'cpu'):
                    net.cpu()
    
    # Clear any cached tensors
    if hasattr(model, 'real_A'):
        del model.real_A
    if hasattr(model, 'real_B'):
        del model.real_B
    if hasattr(model, 'fake_B'):
        del model.fake_B
    if hasattr(model, 'idt_B'):
        del model.idt_B
    
    clear_cuda_memory()


def enable_memory_efficient_settings():
    """Enable memory-efficient PyTorch settings"""
    # Set environment variables for memory efficiency
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Enable memory efficient attention if available
    if hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda, 'enable_flash_sdp'):
        torch.backends.cuda.enable_flash_sdp(True)
    
    print("Memory-efficient settings enabled.")


def set_cuda_memory_fraction(fraction=0.8):
    """Set CUDA memory fraction to limit memory usage"""
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(fraction)
        print(f"CUDA memory fraction set to {fraction}")
    else:
        print("CUDA not available")


def estimate_memory_usage_for_image(image_size, batch_size=1, channels=3):
    """Estimate GPU memory usage for processing an image of given size"""
    if not torch.cuda.is_available():
        return 0
    
    # More realistic estimation for GAN models:
    # - Input tensor: 4 bytes per float32
    # - Generator: typically 4-6x input size
    # - Discriminator: typically 2-3x input size  
    # - Intermediate activations: 3-4x input size
    # - CUDA overhead: ~30%
    
    input_memory = image_size[0] * image_size[1] * batch_size * channels * 4 / (1024**3)  # GB
    
    # For inference (test mode), we mainly need generator + activations
    # Conservative estimate: 6x input size for GAN models during inference
    estimated_memory = input_memory * 6 * 1.3  # 30% overhead
    
    return estimated_memory


def get_safe_image_size(max_memory_gb=8.0, batch_size=1, channels=3):
    """Calculate a safe image size that won't exceed the specified memory limit"""
    if not torch.cuda.is_available():
        return (512, 512)  # Default safe size
    
    # Start with a reasonable size and calculate memory usage
    base_size = 256
    memory_usage = estimate_memory_usage_for_image((base_size, base_size), batch_size, channels)
    
    # Scale up until we approach the memory limit
    while memory_usage < max_memory_gb * 0.6:  # Use 60% of available memory for safety
        base_size *= 1.4
        memory_usage = estimate_memory_usage_for_image((base_size, base_size), batch_size, channels)
    
    # Scale back down to be safe
    safe_size = int(base_size / 1.4)
    return (safe_size, safe_size)


def check_memory_before_processing(image_size, batch_size=1, channels=3, safety_margin=0.2):
    """Check if there's enough memory to process an image of given size"""
    if not torch.cuda.is_available():
        return True
    
    # Get current memory info
    memory_info = get_cuda_memory_info()
    if isinstance(memory_info, str):
        return True
    
    # Estimate required memory
    required_memory = estimate_memory_usage_for_image(image_size, batch_size, channels)
    
    # Check if we have enough free memory
    available_memory = memory_info['free'] * (1 - safety_margin)
    
    return required_memory <= available_memory

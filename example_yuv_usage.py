#!/usr/bin/env python3
"""
Example script demonstrating how to use YUV domain and lambda_NCE_Y features in CUT training.

This script shows the command line arguments and usage patterns for the new features.
"""

import argparse
import sys
import os

def print_example_commands():
    """Print example commands for using YUV domain and lambda_NCE_Y features"""
    
    print("=" * 80)
    print("YUV Domain and lambda_NCE_Y Features for CUT Model")
    print("=" * 80)
    print()
    
    print("The CUT model has been patched to support:")
    print("1. YUV color space for NCE loss computation")
    print("2. Y channel-specific NCE loss with configurable weight")
    print()
    
    print("New Command Line Options:")
    print("  --yuv                    : Enable YUV color space for NCE loss and GAN")
    print("  --lambda_NCE_Y <float>   : Weight for Y channel NCE loss (default: 0.0)")
    print()
    
    print("Example Usage:")
    print()
    
    # Example 1: Basic YUV usage
    print("1. Basic YUV domain training:")
    print("   python train.py --dataroot ./datasets/grumpifycat \\")
    print("                   --name grumpycat_CUT_YUV \\")
    print("                   --CUT_mode CUT \\")
    print("                   --yuv")
    print()
    
    # Example 2: YUV with Y channel NCE loss
    print("2. YUV domain with Y channel NCE loss:")
    print("   python train.py --dataroot ./datasets/grumpifycat \\")
    print("                   --name grumpycat_CUT_YUV_Y \\")
    print("                   --CUT_mode CUT \\")
    print("                   --yuv \\")
    print("                   --lambda_NCE_Y 0.5")
    print()
    
    # Example 3: FastCUT with YUV
    print("3. FastCUT with YUV domain:")
    print("   python train.py --dataroot ./datasets/grumpifycat \\")
    print("                   --name grumpycat_FastCUT_YUV \\")
    print("                   --CUT_mode FastCUT \\")
    print("                   --yuv")
    print()
    
    # Example 4: Single image CUT with YUV
    print("4. Single image CUT with YUV domain:")
    print("   python train.py --dataroot ./datasets/single_image \\")
    print("                   --name single_CUT_YUV \\")
    print("                   --model sincut \\")
    print("                   --yuv \\")
    print("                   --lambda_NCE_Y 0.3")
    print()
    
    # Example 5: Testing with YUV
    print("5. Testing with YUV domain:")
    print("   python test.py --dataroot ./datasets/grumpifycat \\")
    print("                  --name grumpycat_CUT_YUV \\")
    print("                  --CUT_mode CUT \\")
    print("                  --yuv \\")
    print("                  --phase train")
    print()
    
    # Example 6: Testing trained YUV model
    print("6. Testing a model trained with YUV features:")
    print("   python test.py --dataroot ./datasets/grumpifycat \\")
    print("                  --name grumpycat_CUT_YUV_Y \\")
    print("                  --CUT_mode CUT \\")
    print("                  --yuv \\")
    print("                  --lambda_NCE_Y 0.5 \\")
    print("                  --phase test")
    print()
    
    print("Technical Details:")
    print("- YUV conversion uses BT.601 standard")
    print("- Y channel represents luminance information")
    print("- U and V channels represent chrominance information")
    print("- Y channel NCE loss focuses on structural similarity")
    print("- RGB to YUV and YUV to RGB conversions are lossless")
    print()
    
    print("Benefits:")
    print("- YUV domain can improve color consistency in translations")
    print("- Y channel NCE loss emphasizes structural preservation")
    print("- Can be combined with existing CUT/FastCUT features")
    print("- Backward compatible with existing models")
    print()
    
    print("=" * 80)

def print_implementation_details():
    """Print implementation details for developers"""
    
    print("Implementation Details:")
    print("=" * 50)
    print()
    
    print("Files Modified:")
    print("- models/cut_model.py: Added YUV support and Y channel NCE loss")
    print("- models/sincut_model.py: Inherited YUV support from CUT model")
    print("- options/train_options.py: Added YUV and lambda_NCE_Y options")
    print("- util/util.py: Contains RGB<->YUV conversion functions")
    print()
    
    print("Key Functions Added:")
    print("- calculate_Y_channel_NCE_loss(): Y channel specific NCE loss")
    print("- YUV conversion in calculate_NCE_loss() when --yuv is enabled")
    print("- YUV conversion in compute_D_loss() when --yuv is enabled")
    print("- YUV conversion in compute_G_loss() when --yuv is enabled")
    print()
    
    print("Loss Components:")
    print("- loss_NCE: Standard NCE loss (RGB or YUV domain)")
    print("- loss_NCE_Y: Identity NCE loss (when nce_idt=True)")
    print("- loss_NCE_Y_channel: Y channel specific NCE loss (when lambda_NCE_Y > 0)")
    print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YUV Domain and lambda_NCE_Y Features Example')
    parser.add_argument('--details', action='store_true', help='Show implementation details')
    
    args = parser.parse_args()
    
    print_example_commands()
    
    if args.details:
        print_implementation_details() 
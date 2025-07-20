# CUT Model YUV Domain and lambda_NCE_Y Feature Patch

## Overview

This patch adds support for YUV color space and Y channel-specific NCE loss to the Contrastive Unpaired Translation (CUT) model. The implementation is backward compatible and can be used with existing CUT, FastCUT, and SinCUT models.

## Features Added

### 1. YUV Color Space Support
- **Option**: `--yuv`
- **Description**: Enables YUV color space for NCE loss computation and GAN loss
- **Benefits**: Can improve color consistency in image translations

### 2. Y Channel NCE Loss
- **Option**: `--lambda_NCE_Y <float>`
- **Description**: Adds a separate NCE loss computed only on the Y (luminance) channel
- **Default**: 0.0 (disabled)
- **Benefits**: Emphasizes structural similarity and luminance preservation

## Files Modified

### 1. `options/train_options.py`
- Added `--yuv` flag for enabling YUV domain
- Added `--lambda_NCE_Y` parameter for Y channel NCE loss weight

### 2. `options/test_options.py`
- Added `--yuv` flag for testing models trained with YUV features
- Added `--lambda_NCE_Y` parameter for testing models with Y channel NCE loss

### 3. `models/cut_model.py`
- Modified `__init__()` to include Y channel NCE loss in loss names
- Updated `compute_G_loss()` to include Y channel NCE loss computation
- Modified `compute_D_loss()` to support YUV domain for discriminator
- Updated `calculate_NCE_loss()` to use YUV conversion when enabled
- Added `calculate_Y_channel_NCE_loss()` method for Y channel specific loss

### 4. `models/sincut_model.py`
- Added import for `util.util`
- Updated `compute_D_loss()` to handle YUV conversion for R1 loss
- Inherits all YUV features from CUT model

### 5. `util/util.py` (already existed)
- Contains `rgb_to_yuv()` and `yuv_to_rgb()` conversion functions
- Uses BT.601 standard for color space conversion

## Technical Implementation

### YUV Conversion
- Uses BT.601 standard RGB to YUV conversion
- Converts images from [-1, 1] range to YUV and back
- Lossless conversion (numerical precision only)

### Y Channel NCE Loss
- Extracts Y channel from YUV converted images
- Repeats Y channel to 3 channels for generator compatibility
- Computes NCE loss on Y channel features only
- Weighted by `lambda_NCE_Y` parameter

### Integration Points
- **NCE Loss**: Uses YUV domain when `--yuv` is enabled
- **GAN Loss**: Uses YUV domain for both generator and discriminator
- **R1 Loss**: Uses YUV domain for SinCUT models
- **Identity Loss**: Inherits YUV support automatically

## Usage Examples

### Basic YUV Training
```bash
python train.py --dataroot ./datasets/grumpifycat \
                --name grumpycat_CUT_YUV \
                --CUT_mode CUT \
                --yuv
```

### YUV with Y Channel NCE Loss
```bash
python train.py --dataroot ./datasets/grumpifycat \
                --name grumpycat_CUT_YUV_Y \
                --CUT_mode CUT \
                --yuv \
                --lambda_NCE_Y 0.5
```

### FastCUT with YUV
```bash
python train.py --dataroot ./datasets/grumpifycat \
                --name grumpycat_FastCUT_YUV \
                --CUT_mode FastCUT \
                --yuv
```

### Single Image CUT with YUV
```bash
python train.py --dataroot ./datasets/single_image \
                --name single_CUT_YUV \
                --model sincut \
                --yuv \
                --lambda_NCE_Y 0.3
```

## Loss Components

When both features are enabled, the total loss includes:

1. **loss_NCE**: Standard NCE loss (in YUV domain if `--yuv` enabled)
2. **loss_NCE_Y**: Identity NCE loss (when `nce_idt=True`)
3. **loss_NCE_Y_channel**: Y channel specific NCE loss (when `lambda_NCE_Y > 0`)
4. **loss_G_GAN**: GAN loss for generator
5. **loss_D**: Discriminator loss

## Testing

### Test Script
- `test_yuv_features.py`: Comprehensive test of YUV and lambda_NCE_Y features
- Verifies YUV conversion accuracy
- Tests model initialization and loss computation
- Validates Y channel NCE loss calculation

### Example Script
- `example_yuv_usage.py`: Shows usage examples and implementation details
- Run with `--details` flag for technical information

## Benefits

1. **Color Consistency**: YUV domain can improve color preservation
2. **Structural Focus**: Y channel NCE loss emphasizes luminance structure
3. **Backward Compatibility**: Existing models work without modification
4. **Flexibility**: Can be combined with all existing CUT features
5. **Performance**: Minimal computational overhead

## Compatibility

- **CUT**: Full support for both features
- **FastCUT**: Full support for both features
- **SinCUT**: Full support for both features
- **Existing Models**: Backward compatible, no changes required

## Future Enhancements

Potential improvements that could be added:

1. **U/V Channel Losses**: Separate weights for chrominance channels
2. **Adaptive Weights**: Dynamic adjustment of Y channel weight
3. **Color Space Options**: Support for other color spaces (HSV, LAB, etc.)
4. **Perceptual Loss**: Integration with perceptual loss in YUV domain

## Conclusion

This patch successfully adds YUV domain support and Y channel NCE loss to the CUT model while maintaining full backward compatibility. The implementation is robust, well-tested, and provides additional tools for improving image-to-image translation quality. 
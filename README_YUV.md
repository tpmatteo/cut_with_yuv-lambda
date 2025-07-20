# CUT with YUV Domain and lambda_NCE_Y Features

This repository contains the Contrastive Unpaired Translation (CUT) model with additional YUV color space support and Y channel-specific NCE loss features.

## 🆕 New Features

### YUV Color Space Support
- **Option**: `--yuv`
- **Description**: Enables YUV color space for NCE loss computation and GAN loss
- **Benefits**: Can improve color consistency in image translations

### Y Channel NCE Loss
- **Option**: `--lambda_NCE_Y <float>`
- **Description**: Adds a separate NCE loss computed only on the Y (luminance) channel
- **Default**: 0.0 (disabled)
- **Benefits**: Emphasizes structural similarity and luminance preservation

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/tpmatteo/cut_with_yuv-lambda.git
cd cut_with_yuv-lambda
pip install -r requirements.txt
```

### Basic Usage

#### Training with YUV Domain
```bash
python train.py --dataroot ./datasets/grumpifycat \
                --name grumpycat_CUT_YUV \
                --CUT_mode CUT \
                --yuv
```

#### Training with Y Channel NCE Loss
```bash
python train.py --dataroot ./datasets/grumpifycat \
                --name grumpycat_CUT_YUV_Y \
                --CUT_mode CUT \
                --yuv \
                --lambda_NCE_Y 0.5
```

#### Testing with YUV Features
```bash
python test.py --dataroot ./datasets/grumpifycat \
               --name grumpycat_CUT_YUV_Y \
               --CUT_mode CUT \
               --yuv \
               --lambda_NCE_Y 0.5 \
               --phase test
```

## 📖 Examples

See `example_yuv_usage.py` for comprehensive usage examples:
```bash
python example_yuv_usage.py
```

For implementation details:
```bash
python example_yuv_usage.py --details
```

## 🔧 Supported Models

- **CUT**: Full support for both features
- **FastCUT**: Full support for both features  
- **SinCUT**: Full support for both features

## 📋 Command Line Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--yuv` | flag | False | Enable YUV color space for loss computation |
| `--lambda_NCE_Y` | float | 0.0 | Weight for Y channel NCE loss |

## 🎯 Use Cases

### Color Consistency
Use `--yuv` to improve color preservation in image translations:
```bash
python train.py --dataroot ./datasets/horse2zebra \
                --name horse2zebra_yuv \
                --CUT_mode CUT \
                --yuv
```

### Structural Preservation
Use `--lambda_NCE_Y` to emphasize luminance structure:
```bash
python train.py --dataroot ./datasets/facades \
                --name facades_yuv_y \
                --CUT_mode CUT \
                --yuv \
                --lambda_NCE_Y 0.3
```

### Combined Approach
Use both features for optimal results:
```bash
python train.py --dataroot ./datasets/grumpifycat \
                --name grumpycat_optimal \
                --CUT_mode CUT \
                --yuv \
                --lambda_NCE_Y 0.5
```

## 🔬 Technical Details

### YUV Conversion
- Uses BT.601 standard RGB to YUV conversion
- Converts images from [-1, 1] range to YUV and back
- Lossless conversion (numerical precision only)

### Y Channel NCE Loss
- Extracts Y channel from YUV converted images
- Repeats Y channel to 3 channels for generator compatibility
- Computes NCE loss on Y channel features only
- Weighted by `lambda_NCE_Y` parameter

### Loss Components
When both features are enabled:
1. **loss_NCE**: Standard NCE loss (in YUV domain if `--yuv` enabled)
2. **loss_NCE_Y**: Identity NCE loss (when `nce_idt=True`)
3. **loss_NCE_Y_channel**: Y channel specific NCE loss (when `lambda_NCE_Y > 0`)
4. **loss_G_GAN**: GAN loss for generator
5. **loss_D**: Discriminator loss

## 📚 Documentation

- **YUV_PATCH_SUMMARY.md**: Comprehensive technical documentation
- **example_yuv_usage.py**: Usage examples and implementation details
- **Original CUT README**: See the main README.md for base CUT documentation

## 🤝 Compatibility

- **Backward Compatible**: Existing models work without modification
- **All CUT Variants**: Works with CUT, FastCUT, and SinCUT
- **All Phases**: Supports training, testing, and validation
- **All Datasets**: Compatible with all existing dataset modes

## 🔄 Migration from Original CUT

To use YUV features with existing models:

1. **No changes needed** for models trained without YUV features
2. **Add YUV flags** when testing models trained with YUV features
3. **Retrain models** to take advantage of YUV features

## 📈 Performance

- **Minimal overhead**: YUV conversion adds negligible computational cost
- **Improved results**: Can lead to better color consistency and structural preservation
- **Flexible tuning**: Adjust `lambda_NCE_Y` to balance different aspects

## 🐛 Troubleshooting

### Common Issues

1. **YUV options not recognized**: Ensure you're using the latest version
2. **Model loading errors**: Check that YUV flags match training configuration
3. **Memory issues**: YUV features have minimal memory impact

### Getting Help

- Check `example_yuv_usage.py` for usage patterns
- Review `YUV_PATCH_SUMMARY.md` for technical details
- Ensure all dependencies are installed: `pip install -r requirements.txt`

## 📄 License

This project maintains the same license as the original CUT implementation.

## 🙏 Acknowledgments

- Original CUT implementation by Taesung Park et al.
- YUV color space implementation for improved image translation
- Community contributions and feedback

---

**Note**: This is an enhanced version of the original CUT model with additional YUV domain and Y channel NCE loss features. All original CUT functionality is preserved and enhanced. 
# Document OCR Pipeline

A complete document OCR (Optical Character Recognition) solution based on **PaddleOCR v5**, optimized for invoice and accounting document recognition with support for Chinese and English text.

## Features

- **High Accuracy**: Utilizes PP-OCRv5 server models for superior recognition quality
- **Multi-language Support**: Chinese, English, and mixed text recognition
- **Document Preprocessing**: Automatic orientation correction and document unwarping
- **Bounding Box Visualization**: Draw recognition results directly on images
- **Multiple Output Formats**: JSON, Excel, and annotated images
- **Easy Integration**: Simple Python API for quick deployment

## Architecture Overview

This project implements a complete OCR pipeline with the following components:

```
Input Image → Preprocessing → Text Detection → Text Recognition → Output
```

### Pipeline Components

| Component | Model | Description |
|-----------|-------|-------------|
| Document Orientation | PP-LCNet | Classifies document rotation (0°/90°/180°/270°) |
| Document Unwarping | UVDoc | Corrects curved/warped documents |
| Text Detection | PP-OCRv5_det (DB) | Detects text regions using Differentiable Binarization |
| Text Line Classification | PP-LCNet | Determines text line orientation (0°/180°) |
| Text Recognition | PP-OCRv5_rec (SVTR) | Recognizes text using Scene Text Recognition with CTC |

### Model Architecture

- **Text Detection (DB Algorithm)**
  - Backbone: ResNet50_vd
  - Neck: Feature Pyramid Network (FPN)
  - Head: Differentiable Binarization with probability, threshold, and binary maps

- **Text Recognition (SVTR)**
  - Patch Embedding with positional encoding
  - 4-stage encoder with local and global mixing
  - CTC (Connectionist Temporal Classification) decoder
  - Vocabulary: 6,624 characters

For detailed architecture diagrams, see the `diagrams/` folder.

## Installation

### Requirements

- Python 3.8+
- Windows/Linux/macOS

### Install Dependencies

```bash
pip install -r paddleocr_requirements.txt
```

Or install manually:

```bash
pip install paddlepaddle>=2.5.0
pip install paddleocr>=3.0.0
pip install pillow>=9.0.0
pip install opencv-python>=4.5.0
pip install PyMuPDF>=1.21.0
```

## Quick Start

### Basic Usage

```python
from paddleocr import PaddleOCR

# Initialize OCR engine
ocr = PaddleOCR(
    use_doc_orientation_classify=True,  # Enable document orientation detection
    use_doc_unwarping=False,            # Disable unwarping for flat documents
    use_textline_orientation=True       # Enable text line orientation
)

# Perform OCR on an image
result = ocr.predict('path/to/your/image.png')

# Process results
for item in result:
    if hasattr(item, 'rec_texts'):
        for i, text in enumerate(item.rec_texts):
            score = item.rec_scores[i]
            print(f"Text: {text}, Confidence: {score:.4f}")
```

### Draw Bounding Boxes

```python
from PIL import Image, ImageDraw, ImageFont

def draw_ocr_results(image_path, result, output_path):
    """Draw OCR results with bounding boxes on the image."""
    image = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(image)

    for item in result:
        if hasattr(item, 'dt_polys') and hasattr(item, 'rec_texts'):
            for i, poly in enumerate(item.dt_polys):
                # Draw polygon
                points = [(int(p[0]), int(p[1])) for p in poly]
                draw.polygon(points, outline='red', width=2)

                # Draw text
                text = item.rec_texts[i]
                score = item.rec_scores[i]
                draw.text(points[0], f"{text} ({score:.2f})", fill='blue')

    image.save(output_path)
    print(f"Result saved to: {output_path}")

# Usage
result = ocr.predict('invoice.png')
draw_ocr_results('invoice.png', result, 'result.png')
```

### Run Demo Script

```bash
python paddleocr_demo.py
```

## Project Structure

```
document-ocr/
├── docs/
│   ├── diagrams/
│   │   ├── 01_ocr_inference_pipeline.puml   # Inference pipeline diagram
│   │   ├── 02_model_architecture.puml       # Model architecture diagram
│   │   └── 03_training_pipeline.puml        # Training pipeline diagram
│   ├── LICENSE-MIT
│   ├── README_EN.md                         # English documentation
│   └── README_TW.md                         # Traditional Chinese documentation
├── paddleocr_demo.py                        # Demo script
├── paddleocr_requirements.txt               # Dependencies
└── CLAUDE.md                                # Development guidance
```

## Diagrams

The `docs/diagrams/` folder contains detailed PlantUML diagrams:

1. **01_ocr_inference_pipeline.puml** - Complete inference flow from input to output
2. **02_model_architecture.puml** - Detailed neural network architectures
3. **03_training_pipeline.puml** - Data preparation, training, and evaluation

To view diagrams:
- Use VS Code with PlantUML extension (Alt + D to preview)
- Use online PlantUML viewer: https://www.plantuml.com/plantuml/

## Performance

| Model | Size | Inference Time (CPU) | Accuracy |
|-------|------|---------------------|----------|
| PP-OCRv5_det | ~88MB | ~200ms/image | F1: 0.85+ |
| PP-OCRv5_rec | ~85MB | ~50ms/text line | Acc: 0.95+ |
| PP-LCNet (orientation) | ~7MB | ~10ms/image | Acc: 0.99+ |

*Times measured on Intel i7-10700 CPU

## Supported Document Types

- Invoices and receipts
- Financial statements
- Contracts and agreements
- ID cards and certificates
- General printed documents
- Mixed Chinese/English documents

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError: No module named 'paddle'**
   ```bash
   pip install paddlepaddle>=2.5.0
   ```

2. **CUDA out of memory**
   - Use CPU mode or reduce image size
   - Set `use_gpu=False` in PaddleOCR initialization

3. **Poor recognition accuracy**
   - Ensure image resolution is sufficient (>300 DPI recommended)
   - Enable document orientation correction
   - Check if image is properly lit and not blurry

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Author

**Jammy Lin**
Email: a0925281767s@gmail.com

## License

This project is licensed under the MIT License - see the [LICENSE-MIT](LICENSE-MIT) file for details.

## Acknowledgments

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) - The underlying OCR engine
- [PaddlePaddle](https://www.paddlepaddle.org.cn/) - Deep learning framework

## References

- PP-OCRv5: [PaddleOCR Documentation](https://paddlepaddle.github.io/PaddleOCR/)
- DB Algorithm: [Real-time Scene Text Detection with Differentiable Binarization](https://arxiv.org/abs/1911.08947)
- SVTR: [SVTR: Scene Text Recognition with a Single Visual Model](https://arxiv.org/abs/2205.00159)

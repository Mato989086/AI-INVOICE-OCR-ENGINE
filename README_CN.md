# AI-INVOICE-OCR-ENGINE

中文版 | [English](README_EN.md)

一個基於 **PaddleOCR v5** 的完整 AI 文件光學字元辨識 (OCR) 解決方案，針對發票和會計文件辨識進行優化，支援中文和英文文字識別。

## 展示 Demo

| 輸入 | 輸出 |
|------|------|
| ![輸入](demo/input.png) | ![輸出](demo/output.png) |

> 將你的展示圖片放入 `demo/` 資料夾

## 功能特色

- **高準確率**：採用 PP-OCRv5 伺服器級模型，提供卓越的辨識品質
- **多語言支援**：支援中文、英文及混合文字辨識
- **文件前處理**：自動方向校正和文件扭曲矯正
- **邊框視覺化**：直接在圖片上繪製辨識結果
- **多種輸出格式**：支援 JSON、Excel 和標註圖片輸出
- **易於整合**：簡單的 Python API，快速部署

## 架構概述

```
輸入圖片 → 前處理 → 文字偵測 → 文字辨識 → 輸出結果
```

### 流程元件

| 元件 | 模型 | 說明 |
|------|------|------|
| 文件方向偵測 | PP-LCNet | 分類文件旋轉角度 (0°/90°/180°/270°) |
| 文件扭曲矯正 | UVDoc | 校正彎曲/扭曲的文件 |
| 文字偵測 | PP-OCRv5_det (DB) | 使用可微分二值化演算法偵測文字區域 |
| 文字行分類 | PP-LCNet | 判斷文字行方向 (0°/180°) |
| 文字辨識 | PP-OCRv5_rec (SVTR) | 使用場景文字辨識模型搭配 CTC 解碼 |

### 模型架構

- **文字偵測 (DB 演算法)**
  - 主幹網路：ResNet50_vd
  - 特徵融合：特徵金字塔網路 (FPN)
  - 輸出頭：可微分二值化，包含機率圖、閾值圖和二值圖

- **文字辨識 (SVTR)**
  - 區塊嵌入與位置編碼
  - 四階段編碼器，包含局部和全域混合
  - CTC（連結時序分類）解碼器
  - 詞彙表：6,624 個字元

## 安裝說明

### 系統需求

- Python 3.8+
- Windows/Linux/macOS

### 安裝相依套件

```bash
pip install paddlepaddle>=2.5.0
pip install paddleocr>=3.0.0
pip install pillow>=9.0.0
pip install opencv-python>=4.5.0
pip install PyMuPDF>=1.21.0
```

## 快速開始

### 基本使用

```python
from paddleocr import PaddleOCR

# 初始化 OCR 引擎
ocr = PaddleOCR(
    use_doc_orientation_classify=True,
    use_doc_unwarping=False,
    use_textline_orientation=True
)

# 對圖片執行 OCR
result = ocr.predict('path/to/your/image.png')

# 處理結果
for item in result:
    if hasattr(item, 'rec_texts'):
        for i, text in enumerate(item.rec_texts):
            score = item.rec_scores[i]
            print(f"文字: {text}, 信心度: {score:.4f}")
```

### 繪製邊框

```python
from PIL import Image, ImageDraw

def draw_ocr_results(image_path, result, output_path):
    image = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(image)

    for item in result:
        if hasattr(item, 'dt_polys') and hasattr(item, 'rec_texts'):
            for i, poly in enumerate(item.dt_polys):
                points = [(int(p[0]), int(p[1])) for p in poly]
                draw.polygon(points, outline='red', width=2)
                text = item.rec_texts[i]
                score = item.rec_scores[i]
                draw.text(points[0], f"{text} ({score:.2f})", fill='blue')

    image.save(output_path)

# 使用範例
result = ocr.predict('invoice.png')
draw_ocr_results('invoice.png', result, 'result.png')
```

## 專案結構

```
AI-INVOICE-OCR-ENGINE/
├── README_EN.md
├── README_CN.md
├── LICENSE-MIT
├── setup.py
├── config.py
├── engine.py
├── cli.py
├── demo/                    # 展示圖片
├── models/
│   └── pretrained/
│       └── weights/         # 模型檔案 (.onnx, .pdmodel, .pdiparams)
├── preprocess/
├── detect/
├── recognize/
├── data/
├── losses/
├── train/
└── utils/
```

## 效能表現

| 模型 | 大小 | 推論時間 (CPU) | 準確率 |
|------|------|----------------|--------|
| PP-OCRv5_det | ~88MB | ~200ms/張圖片 | F1: 0.85+ |
| PP-OCRv5_rec | ~85MB | ~50ms/文字行 | 準確率: 0.95+ |
| PP-LCNet (方向) | ~7MB | ~10ms/張圖片 | 準確率: 0.99+ |

*測試環境：Intel i7-10700 CPU

## 支援的文件類型

- 發票和收據
- 財務報表
- 合約和協議書
- 身分證件和證書
- 一般印刷文件
- 中英文混合文件

## 作者

**Jammy Lin**
Email: a0925281767s@gmail.com

## 授權條款

本專案採用 MIT 授權條款 - 詳見 [LICENSE-MIT](LICENSE-MIT) 檔案。

## 致謝

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) - 底層 OCR 引擎
- [PaddlePaddle](https://www.paddlepaddle.org.cn/) - 深度學習框架

## 參考資料

- PP-OCRv5：[PaddleOCR 官方文件](https://paddlepaddle.github.io/PaddleOCR/)
- DB 演算法：[Real-time Scene Text Detection with Differentiable Binarization](https://arxiv.org/abs/1911.08947)
- SVTR：[SVTR: Scene Text Recognition with a Single Visual Model](https://arxiv.org/abs/2205.00159)

---

## 支持這個專案

如果這個專案對你有幫助，請給一個 ⭐ **Star**！

你的支持是我持續改進的動力！

[![GitHub stars](https://img.shields.io/github/stars/xup6jammy/AI-INVOICE-OCR-ENGINE?style=social)](https://github.com/xup6jammy/AI-INVOICE-OCR-ENGINE)

**分享這個專案：**
- Fork 並貢獻程式碼
- 回報問題和建議
- 幫忙宣傳！

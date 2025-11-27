# 文件 OCR 辨識系統

一個基於 **PaddleOCR v5** 的完整文件光學字元辨識 (OCR) 解決方案，針對發票和會計文件辨識進行優化，支援中文和英文文字識別。

## 功能特色

- **高準確率**：採用 PP-OCRv5 伺服器級模型，提供卓越的辨識品質
- **多語言支援**：支援中文、英文及混合文字辨識
- **文件前處理**：自動方向校正和文件扭曲矯正
- **邊框視覺化**：直接在圖片上繪製辨識結果
- **多種輸出格式**：支援 JSON、Excel 和標註圖片輸出
- **易於整合**：簡單的 Python API，快速部署

## 架構概述

本專案實作了完整的 OCR 處理流程：

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

詳細架構圖請參閱 `diagrams/` 資料夾。

## 安裝說明

### 系統需求

- Python 3.8+
- Windows/Linux/macOS

### 安裝相依套件

```bash
pip install -r paddleocr_requirements.txt
```

或手動安裝：

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
    use_doc_orientation_classify=True,  # 啟用文件方向偵測
    use_doc_unwarping=False,            # 平面文件可停用扭曲矯正
    use_textline_orientation=True       # 啟用文字行方向偵測
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
from PIL import Image, ImageDraw, ImageFont

def draw_ocr_results(image_path, result, output_path):
    """在圖片上繪製 OCR 結果和邊框。"""
    image = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(image)

    for item in result:
        if hasattr(item, 'dt_polys') and hasattr(item, 'rec_texts'):
            for i, poly in enumerate(item.dt_polys):
                # 繪製多邊形
                points = [(int(p[0]), int(p[1])) for p in poly]
                draw.polygon(points, outline='red', width=2)

                # 繪製文字
                text = item.rec_texts[i]
                score = item.rec_scores[i]
                draw.text(points[0], f"{text} ({score:.2f})", fill='blue')

    image.save(output_path)
    print(f"結果已儲存至: {output_path}")

# 使用範例
result = ocr.predict('invoice.png')
draw_ocr_results('invoice.png', result, 'result.png')
```

### 執行範例程式

```bash
python paddleocr_demo.py
```

## 專案結構

```
document-ocr/
├── docs/
│   ├── diagrams/
│   │   ├── 01_ocr_inference_pipeline.puml   # 推論流程圖
│   │   ├── 02_model_architecture.puml       # 模型架構圖
│   │   └── 03_training_pipeline.puml        # 訓練流程圖
│   ├── LICENSE-MIT
│   ├── README_EN.md                         # 英文文件
│   └── README_TW.md                         # 繁體中文文件
├── paddleocr_demo.py                        # 範例程式
├── paddleocr_requirements.txt               # 相依套件
└── CLAUDE.md                                # 開發指引
```

## 架構圖說明

`docs/diagrams/` 資料夾包含詳細的 PlantUML 架構圖：

1. **01_ocr_inference_pipeline.puml** - 完整推論流程，從輸入到輸出
2. **02_model_architecture.puml** - 詳細神經網路架構
3. **03_training_pipeline.puml** - 資料準備、訓練和評估流程

檢視架構圖的方式：
- 使用 VS Code 搭配 PlantUML 擴充套件 (Alt + D 預覽)
- 使用線上 PlantUML 檢視器：https://www.plantuml.com/plantuml/

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

## 疑難排解

### 常見問題

1. **ModuleNotFoundError: No module named 'paddle'**
   ```bash
   pip install paddlepaddle>=2.5.0
   ```

2. **CUDA 記憶體不足**
   - 使用 CPU 模式或縮小圖片尺寸
   - 在 PaddleOCR 初始化時設定 `use_gpu=False`

3. **辨識準確率不佳**
   - 確保圖片解析度足夠（建議 >300 DPI）
   - 啟用文件方向校正
   - 檢查圖片是否光線充足且不模糊

## 貢獻指南

歡迎貢獻！請隨時提交 Pull Request。

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

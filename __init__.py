# -*- coding: utf-8 -*-
"""文件OCR流程 - 核心套件"""
__version__='2.1.0';__author__='Jammy';__all__=['engine','preprocess','detect','recognize','utils','postprocess']
from.engine import OCREngine as _E;from.config import Config as _C
def init(c=None):
    """初始化OCR引擎"""
    return _E(_C()if c is None else c)

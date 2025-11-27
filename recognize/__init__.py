# -*- coding: utf-8 -*-
"""文字辨識模組 - SVTR演算法與CTC解碼"""
from.recognizer import Recognizer
from.svtr import SVTRRecognizer
from.ctc import CTCDecoder
from.vocab import Vocabulary
__all__=['Recognizer','SVTRRecognizer','CTCDecoder','Vocabulary']

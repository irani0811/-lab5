# -*- coding: utf-8 -*-
"""
常量定义模块。
保存标签映射等在多个脚本间共享的信息。
"""

from typing import Dict, List

LABEL2ID: Dict[str, int] = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL: List[str] = ["negative", "neutral", "positive"]

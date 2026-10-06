# -*- coding: utf-8 -*-
"""AI Blocks 配色主题 —— 面向 AI 训练的四类积木。"""

CATEGORIES = [
    ("data",     "数据", "#2E9E8F", "#20756A"),
    ("model",    "模型", "#4C6FE0", "#3550A8"),
    ("train",    "训练", "#E07B1A", "#B05F12"),
    ("predict",  "推理", "#8250E0", "#5F3AA8"),
    ("control",  "流程", "#E0A020", "#A87812"),
    ("output",   "输出", "#C2453C", "#8F332C"),
]

CAT_MAP = {k: (n, c, d) for k, n, c, d in CATEGORIES}

WORKSPACE_BG = "#EEF1F6"
PALETTE_BG = "#F7F9FC"
STAGE_BG = "#FFFFFF"
DASH_BG = "#0F172A"
DASH_PANEL = "#16213B"
DASH_BORDER = "#233153"
DASH_TEXT = "#E6EDF7"
DASH_MUTED = "#7C8BA8"
ACCENT = "#4C97FF"
ACCENT_2 = "#22C55E"
CANVAS_BG = "#FFFFFF"
PANEL_BORDER = "#D8DFEA"

TEXT_DARK = "#575E75"
TEXT_WHITE = "#FFFFFF"
SHADOW = "#D8DEE8"
EVENT_COLOR = "#E0A020"
EVENT_DARK = "#A87812"

HEAT_LOW = (238, 242, 248)
HEAT_HIGH = (76, 110, 224)

def heat_color(t):
    """t in [0,1] → 蓝白热力色。"""
    t = max(0.0, min(1.0, t))
    r = HEAT_LOW[0] + (HEAT_HIGH[0] - HEAT_LOW[0]) * t
    g = HEAT_LOW[1] + (HEAT_HIGH[1] - HEAT_LOW[1]) * t
    b = HEAT_LOW[2] + (HEAT_HIGH[2] - HEAT_LOW[2]) * t
    return "#%02x%02x%02x" % (int(r), int(g), int(b))

# -*- coding: utf-8 -*-
"""AI 训练可视化面板 —— 在 Tkinter Canvas 上绘制学习过程。

每个板块一个视图类，统一接口：
    view = XxxView(parent_canvas)
    view.update_xxx(...)   # 训练中每步调用，重绘
"""

import math
import tkinter as tk

from themes import (DASH_BG, DASH_PANEL, DASH_BORDER, DASH_TEXT, DASH_MUTED,
                    ACCENT, ACCENT_2, heat_color)
import i18n as _i18n

def _T(s):
    return _i18n.t(s) if isinstance(s, str) else s

CLASS_COLORS = ["#4C97FF", "#F25C54", "#22C55E", "#FBBF24", "#A855F7", "#06B6D4"]

def _map(v, a, b, c, d):
    """把 v 从 [a,b] 线性映射到 [c,d]。"""
    if b - a == 0:
        return (c + d) / 2
    return c + (v - a) / (b - a) * (d - c)

class LineChart:
    """通用折线图：可画多条曲线，带坐标轴与网格。"""

    def __init__(self, canvas, x0, y0, x1, y1, title="", y_min=0, y_max=1,
                 y_fmt="{:.2f}"):
        self.c = canvas
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1
        self.title = title
        self.y_min, self.y_max = y_min, y_max
        self.y_fmt = y_fmt

    def _px(self, i, n):
        if n <= 1:
            return self.x0
        return _map(i, 0, n - 1, self.x0 + 30, self.x1 - 8)

    def _py(self, v):
        lo, hi = self.y_min, self.y_max
        if hi <= lo:
            hi = lo + 1
        return _map(v, lo, hi, self.y1 - 18, self.y0 + 22)

    def draw(self, series, tags="chart"):
        """series: [(values, color, label), ...] 每条一个数组。"""
        c = self.c
        c.delete(tags)
        c.create_rectangle(self.x0, self.y0, self.x1, self.y1,
                           fill=DASH_PANEL, outline=DASH_BORDER, tags=tags)
        if self.title:
            c.create_text(self.x0 + 8, self.y0 + 8, text=self.title, anchor="nw",
                          fill=DASH_TEXT, font=("Microsoft YaHei UI", 9, "bold"),
                          tags=tags)
        for k in range(5):
            v = self.y_min + (self.y_max - self.y_min) * k / 4
            py = self._py(v)
            c.create_line(self.x0 + 30, py, self.x1 - 8, py,
                          fill="#22304E", tags=tags)
            c.create_text(self.x0 + 26, py, text=self.y_fmt.format(v), anchor="e",
                          fill=DASH_MUTED, font=("Consolas", 7), tags=tags)
        for vals, color, label in series:
            if len(vals) < 2:
                continue
            n = len(vals)
            pts = []
            for i, v in enumerate(vals):
                pts.append(self._px(i, n))
                pts.append(self._py(v))
            c.create_line(*pts, fill=color, width=2, smooth=False, tags=tags)
            c.create_oval(self._px(n - 1, n) - 3, self._py(vals[-1]) - 3,
                          self._px(n - 1, n) + 3, self._py(vals[-1]) + 3,
                          fill=color, outline="", tags=tags)
        lx = self.x0 + 40
        for vals, color, label in series:
            if label:
                c.create_rectangle(lx, self.y0 + 8, lx + 10, self.y0 + 16,
                                   fill=color, outline="", tags=tags)
                c.create_text(lx + 14, self.y0 + 12, text=label, anchor="w",
                              fill=DASH_TEXT, font=("Microsoft YaHei UI", 8),
                              tags=tags)
                lx += 20 + len(label) * 12

class ScatterView:
    """分类散点图 + 背景决策边界（颜色越饱和表示越确信）。"""

    def __init__(self, canvas, x0, y0, x1, y1, title=""):
        self.c = canvas
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1
        self.title = title

    def _map_point(self, p, bounds):
        bx0, bx1, by0, by1 = bounds
        px = _map(p[0], bx0, bx1, self.x0 + 8, self.x1 - 8)
        py = _map(p[1], by0, by1, self.y1 - 8, self.y0 + 22)
        return px, py

    def draw(self, dataset, grid=None, bounds=None, tags="scatter",
             highlight=None, classes=None, centroids=None, curves=None):
        c = self.c
        c.delete(tags)
        c.create_rectangle(self.x0, self.y0, self.x1, self.y1,
                           fill=DASH_PANEL, outline=DASH_BORDER, tags=tags)
        if self.title:
            c.create_text(self.x0 + 8, self.y0 + 8, text=self.title, anchor="nw",
                          fill=DASH_TEXT, font=("Microsoft YaHei UI", 9, "bold"),
                          tags=tags)
        if bounds is None and dataset is not None:
            bounds = dataset.bounds
        if bounds is None:
            return
        bx0, bx1, by0, by1 = bounds
        if grid:
            res = len(grid)
            rx = (self.x1 - 16) / res
            ry = (self.y1 - 30) / res
            for gy in range(res):
                for gx in range(res):
                    cls = grid[gy][gx]
                    col = CLASS_COLORS[cls % len(CLASS_COLORS)]
                    px = self.x0 + 8 + gx * rx
                    py = self.y1 - 8 - (gy + 1) * ry
                    c.create_rectangle(px, py, px + rx + 0.5, py + ry + 0.5,
                                       fill=col, outline="", stipple="gray50",
                                       tags=tags)
        if curves:
            for pts, color in curves:
                poly = []
                for wx, wy in pts:
                    px = _map(wx, bx0, bx1, self.x0 + 8, self.x1 - 8)
                    py = _map(wy, by0, by1, self.y1 - 8, self.y0 + 22)
                    poly.extend([px, py])
                if len(poly) >= 4:
                    c.create_line(*poly, fill=color, width=2, tags=tags)
        if dataset:
            for i, p in enumerate(dataset.X):
                px, py = self._map_point(p, bounds)
                col = CLASS_COLORS[dataset.y[i] % len(CLASS_COLORS)]
                c.create_oval(px - 3, py - 3, px + 3, py + 3,
                              fill=col, outline="#0B1220", width=1, tags=tags)
        if centroids:
            for j, cpt in enumerate(centroids):
                px, py = self._map_point(cpt, bounds)
                col = CLASS_COLORS[j % len(CLASS_COLORS)]
                c.create_line(px - 7, py, px + 7, py, fill="#0B1220", width=3, tags=tags)
                c.create_line(px, py - 7, px, py + 7, fill="#0B1220", width=3, tags=tags)
                c.create_oval(px - 6, py - 6, px + 6, py + 6, outline=col,
                              width=3, tags=tags)
        if highlight:
            for (px, py, kind) in highlight:
                if kind == "query":
                    c.create_oval(px - 6, py - 6, px + 6, py + 6,
                                  outline="#FFFFFF", width=2, dash=(4, 2), tags=tags)
                elif kind == "neighbor":
                    c.create_oval(px - 8, py - 8, px + 8, py + 8,
                                  outline="#FBBF24", width=2, tags=tags)

class HeatmapView:
    def __init__(self, canvas, x0, y0, x1, y1, title=""):
        self.c = canvas
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1
        self.title = title

    def draw(self, labels, matrix, tags="heat"):
        c = self.c
        c.delete(tags)
        c.create_rectangle(self.x0, self.y0, self.x1, self.y1,
                           fill=DASH_PANEL, outline=DASH_BORDER, tags=tags)
        if self.title:
            c.create_text(self.x0 + 8, self.y0 + 8, text=self.title, anchor="nw",
                          fill=DASH_TEXT, font=("Microsoft YaHei UI", 9, "bold"),
                          tags=tags)
        if not labels or not matrix:
            c.create_text((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2,
                          text=_T("尚未训练"), fill=DASH_MUTED,
                          font=("Microsoft YaHei UI", 10), tags=tags)
            return
        n = len(labels)
        top = self.y0 + 26
        size = min((self.x1 - self.x0 - 60) / n, (self.y1 - top - 24) / n)
        ox = self.x0 + 44
        for j, lab in enumerate(labels):
            c.create_text(ox + j * size + size / 2, top - 8, text=lab,
                          fill=DASH_MUTED, font=("Microsoft YaHei UI", 7),
                          tags=tags)
            c.create_text(ox - 6, top + j * size + size / 2, text=lab,
                          anchor="e", fill=DASH_MUTED,
                          font=("Microsoft YaHei UI", 7), tags=tags)
        for gy in range(n):
            for gx in range(n):
                v = matrix[gy][gx]
                col = heat_color(v)
                px = ox + gx * size
                py = top + gy * size
                c.create_rectangle(px, py, px + size, py + size,
                                   fill=col, outline="#0B1220", tags=tags)

class NetView:
    """画神经网络结构，节点亮度随激活值变化。"""

    def __init__(self, canvas, x0, y0, x1, y1, title=""):
        self.c = canvas
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1
        self.title = title

    def draw(self, layers, activations=None, tags="net"):
        """layers: [n_in, h1, h2, ..., n_out]"""
        c = self.c
        c.delete(tags)
        c.create_rectangle(self.x0, self.y0, self.x1, self.y1,
                           fill=DASH_PANEL, outline=DASH_BORDER, tags=tags)
        if self.title:
            c.create_text(self.x0 + 8, self.y0 + 8, text=self.title, anchor="nw",
                          fill=DASH_TEXT, font=("Microsoft YaHei UI", 9, "bold"),
                          tags=tags)
        if not layers:
            return
        ncol = len(layers)
        top, bot = self.y0 + 30, self.y1 - 14
        cols = []
        for li, n in enumerate(layers):
            x = _map(li, 0, max(1, ncol - 1), self.x0 + 34, self.x1 - 26)
            shown = min(n, 8)
            ys = []
            for k in range(shown):
                y = _map(k, 0, max(1, shown - 1), top + 8, bot - 8) if shown > 1 \
                    else (top + bot) / 2
                ys.append(y)
            cols.append((x, ys, shown, n))
        for li in range(ncol - 1):
            x0c, ys0, s0, _ = cols[li]
            x1c, ys1, s1, _ = cols[li + 1]
            for ya in ys0:
                for yb in ys1:
                    c.create_line(x0c, ya, x1c, yb, fill="#1E2A44", width=1, tags=tags)
        for li, (x, ys, shown, n) in enumerate(cols):
            for k, y in enumerate(ys):
                act = 0.5
                if activations and li < len(activations) and activations[li]:
                    arr = activations[li]
                    if k < len(arr):
                        act = max(0.0, min(1.0, abs(arr[k])))
                col = heat_color(act)
                r = 7
                c.create_oval(x - r, y - r, x + r, y + r, fill=col,
                              outline="#3A4A6E", tags=tags)
            if n > shown:
                c.create_text(x, cols[li][1][-1] + 14, text=f"+{n-shown}",
                              fill=DASH_MUTED, font=("Consolas", 7), tags=tags)
        for li, lab in enumerate([_T("输入")] + [f"{_T('隐藏')}{i+1}" for i in range(ncol - 2)] + [_T("输出")]):
            if li < ncol:
                c.create_text(cols[li][0], self.y1 - 4, text=lab,
                              fill=DASH_MUTED, font=("Microsoft YaHei UI", 7),
                              tags=tags)

class RecogBars:
    """把各类别的预测概率画成横向条形，最高的那条高亮。"""

    def __init__(self, canvas, x0, y0, x1, y1, title=""):
        self.c = canvas
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1
        self.title = title

    def draw(self, rows, best=None, tags="recog"):
        """rows: [(类别名, 概率 0~1), ...]；best: 命中的类别名。"""
        c = self.c
        c.delete(tags)
        c.create_rectangle(self.x0, self.y0, self.x1, self.y1,
                           fill=DASH_PANEL, outline=DASH_BORDER, tags=tags)
        if self.title:
            c.create_text(self.x0 + 10, self.y0 + 10, text=self.title, anchor="nw",
                          fill=DASH_TEXT, font=("Microsoft YaHei UI", 9, "bold"),
                          tags=tags)

        if not rows:
            c.create_text((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2,
                          text=_T("训练后这里会显示每个类别的概率"),
                          fill=DASH_MUTED, font=("Microsoft YaHei UI", 9), tags=tags)
            return

        top = self.y0 + 34
        avail = self.y1 - top - 12
        n = len(rows)
        bh = max(8, min(26, avail / max(1, n) - 8))
        gap = 8
        bar_x0 = self.x0 + 96
        bar_x1 = self.x1 - 46
        bar_w = max(30, bar_x1 - bar_x0)

        for i, (name, p) in enumerate(rows):
            y = top + i * (bh + gap)
            if y + bh > self.y1 - 6:
                break
            p = max(0.0, min(1.0, float(p)))
            hit = (best is not None and name == best)
            c.create_text(self.x0 + 88, y + bh / 2, text=str(name)[:10], anchor="e",
                          fill=(ACCENT_2 if hit else DASH_TEXT),
                          font=("Microsoft YaHei UI", 9, "bold" if hit else "normal"),
                          tags=tags)
            c.create_rectangle(bar_x0, y, bar_x0 + bar_w, y + bh,
                               fill="#16223C", outline="", tags=tags)
            w = bar_w * p
            if w > 0:
                c.create_rectangle(bar_x0, y, bar_x0 + w, y + bh,
                                   fill=(ACCENT_2 if hit else ACCENT),
                                   outline="", tags=tags)
            c.create_text(bar_x0 + bar_w + 22, y + bh / 2, text=f"{p:.0%}",
                          fill=(ACCENT_2 if hit else DASH_MUTED),
                          font=("Consolas", 9), tags=tags)

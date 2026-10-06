# -*- coding: utf-8 -*-
"""积木可视化组件：几何计算、绘制、拖拽、吸附。"""

import math
import tkinter as tk

from themes import CAT_MAP, TEXT_WHITE, TEXT_DARK, SHADOW
from blocks_def import (
    BLOCKS, COMMON_BLOCKS, BOARDS,
    SHAPE_HAT, SHAPE_CBLOCK, SHAPE_CAP,
    IN_NUMBER, IN_TEXT, IN_COLOR, IN_SELECT, IN_PATH,
)
import i18n as _i18n

def _T(s):
    """翻译 helper：None / 非字符串安全。"""
    if not isinstance(s, str):
        return s
    return _i18n.t(s)

_ACTIVE_BLOCKS = {}

def set_blocks(blocks):
    """替换当前生效的积木定义（板块切换时调用）。"""
    global _ACTIVE_BLOCKS
    _ACTIVE_BLOCKS = blocks

NOTCH_W = 16
NOTCH_D = 4
PAD_L = 12
PAD_R = 12
ROW_H = 26
INNER_INDENT = 14
INNER_PAD_V = 10
RADIUS = 6

FONT_CN = ("Microsoft YaHei UI", 10, "bold")
FONT_CN_S = ("Microsoft YaHei UI", 9, "bold")
FONT_MONO = ("Consolas", 10, "bold")

def hex_to_rgb(h):
    h = h.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))

def rgb_to_hex(rgb):
    return "#%02x%02x%02x" % tuple(max(0, min(255, int(v))) for v in rgb)

def darken(hex_color, factor=0.78):
    r, g, b = hex_to_rgb(hex_color)
    return rgb_to_hex((r * factor, g * factor, b * factor))

def lighten(hex_color, factor=0.3):
    r, g, b = hex_to_rgb(hex_color)
    return rgb_to_hex((r + (255 - r) * factor, g + (255 - g) * factor, b + (255 - b) * factor))

def text_width(text, big=True):
    w = 0.0
    for ch in text:
        if ord(ch) > 0x2E80:
            w += 12.5
        elif ch in "iljt.,:;!|'":
            w += 4.0
        else:
            w += 7.2
    return w

class BlockView:
    def __init__(self, app, kind, x, y, values=None, cond=None,
                 palette=False, suppress_widgets=False):
        self.app = app
        self.kind = kind
        self.spec = (_ACTIVE_BLOCKS or BLOCKS)[kind]
        self.x = float(x)
        self.y = float(y)
        self.palette = palette
        self.suppress_widgets = suppress_widgets
        self.canvas = app.palette_canvas if palette else app.ws_canvas

        self.children = []
        self.next_block = None
        self.parent = None
        self.slot_owner = None

        self.items = []
        self.fv_items = []
        self.input_regions = []

        self.values = []
        for i, (default, itype, opts, w) in enumerate(self.spec["inputs"]):
            self.values.append(default if values is None or i >= len(values) else values[i])
        self.cond = cond

        self._layout()
        self.draw()

    def _seg_layout(self):
        """把 label 拆成 [('t',文字,宽度) | ('i',索引,宽度)]"""
        label = _T(self.spec["label"])
        parts = label.split("{}")
        segs = []
        content_w = 0.0
        for i, seg in enumerate(parts):
            if seg:
                sw = text_width(seg)
                segs.append(["t", seg, sw])
                content_w += sw
            if i < len(parts) - 1:
                iw = self._input_width(i)
                segs.append(["i", i, iw])
                content_w += iw
        self.segs = segs
        return content_w

    def _input_width(self, idx):
        default, itype, opts, w = self.spec["inputs"][idx]
        if itype == IN_COLOR:
            return 32
        if itype == IN_PATH:
            val = str(self.values[idx])
            shown = val.split("/")[-1].split("\\")[-1] or _T("选择文件…")
            return max(w, text_width(shown) + 22)
        if itype == IN_SELECT and opts:
            val = self.values[idx]
            shown = dict(opts).get(val, val)
            return max(w, text_width(_T(str(shown))) + 22)
        val = self.values[idx]
        need = text_width(str(val)) + 18
        return max(w, need)

    def _layout(self):
        content_w = self._seg_layout()
        self.head_w = PAD_L + content_w + PAD_R
        self.head_h = ROW_H

        if self.spec["shape"] == SHAPE_HAT:
            self.head_h = ROW_H + 6

        if self.spec["shape"] == SHAPE_CBLOCK:
            max_w = 0
            total_h = 0
            for c in self.children:
                total_h += c.total_height
                max_w = max(max_w, c.total_width)
            self.inner_h = max(36, total_h) + INNER_PAD_V
            self.inner_w = max(max_w, 50)
            self.w = max(self.head_w, INNER_INDENT + self.inner_w + INNER_INDENT)
            self.h = self.head_h + self.inner_h + 14
        else:
            self.w = self.head_w
            self.h = self.head_h

        self.total_width = self.w
        self.total_height = self.h
        n = self.next_block
        while n is not None:
            self.total_height += n.h
            self.total_width = max(self.total_width, n.w)
            n = n.next_block

    def relayout(self):
        self._layout()
        self.draw()
        self._place_stack()

    def relayout_all(self):
        """递归重算整条链（含 next 与 C 型子块）的布局并重绘。"""
        self._layout()
        n = self.next_block
        while n is not None:
            n.relayout_all()
            n = n.next_block
        for c in self.children:
            c.relayout_all()
        self.draw()
        self._place_stack()
        self._raise_bg_all()
        self._raise_fg_all()

    def _raise_items(self, block):
        """把一块积木及其子链的画布元素提到最上层（先背景后前景）。"""
        b = block
        while b is not None:
            for i in b.items:
                try:
                    self.canvas.tag_raise(i)
                except tk.TclError:
                    pass
            for c in b.children:
                self._raise_items(c)
            for i in b.fv_items:
                try:
                    self.canvas.tag_raise(i)
                except tk.TclError:
                    pass
            b = b.next_block

    def _raise_chain(self, block):
        self._raise_items(block)

    def _place_stack(self):
        n = self.next_block
        cy = self.y + self.h
        while n is not None:
            n._layout()
            n.x = self.x
            n.y = cy
            n.draw()
            n._place_stack()
            self._raise_items(n)
            cy += n.h
            n = n.next_block
        if self.spec["shape"] == SHAPE_CBLOCK:
            cy = self.y + self.head_h + INNER_PAD_V / 2
            for c in self.children:
                c._layout()
                c.x = self.x + INNER_INDENT
                c.y = cy
                c.draw()
                c._place_stack()
                self._raise_items(c)
                cy += c.total_height

    def _raise_fg_all(self):
        """整棵积木树：把所有前景（文字/输入框）提到所有背景之上，避免被邻块盖住。"""
        b = self
        while b is not None:
            for i in b.fv_items:
                try:
                    self.canvas.tag_raise(i)
                except tk.TclError:
                    pass
            for c in b.children:
                c._raise_fg_all()
            b = b.next_block

    def _raise_bg_all(self):
        """整棵积木树：背景元素按 上层块在下的顺序垒好。"""
        b = self
        while b is not None:
            for i in b.items:
                try:
                    self.canvas.tag_raise(i)
                except tk.TclError:
                    pass
            for c in b.children:
                c._raise_bg_all()
            b = b.next_block

    def _outline(self, x, y, w, h, top_notch=True, bottom_notch=True):
        shape = self.spec["shape"]
        pts = []
        if shape == SHAPE_HAT:
            cap_h = 10
            pts.append((x + 2, y + cap_h))
            steps = 16
            for i in range(steps + 1):
                t = i / steps
                px = x + 2 + (w - 4) * t
                py = y + cap_h * (1 - math.sin(math.pi * t) ** 0.7)
                pts.append((px, py))
            pts.append((x + w, y + cap_h + RADIUS))
        else:
            pts.append((x + RADIUS, y))
            if top_notch:
                nx0 = x + PAD_L - 2
                pts.append((nx0, y))
                pts.append((nx0 + NOTCH_D, y + NOTCH_D))
                pts.append((nx0 + NOTCH_D + NOTCH_W, y + NOTCH_D))
                pts.append((nx0 + NOTCH_W + NOTCH_D * 2, y))
            pts.append((x + w - RADIUS, y))
            pts.append((x + w, y + RADIUS))
        pts.append((x + w, y + h - RADIUS))
        if bottom_notch and shape != SHAPE_CAP:
            pts.append((x + w - RADIUS, y + h))
            nx0 = x + PAD_L - 2
            pts.append((nx0 + NOTCH_W + NOTCH_D * 2, y + h))
            pts.append((nx0 + NOTCH_W + NOTCH_D, y + h + NOTCH_D))
            pts.append((nx0 + NOTCH_D, y + h + NOTCH_D))
            pts.append((nx0, y + h))
            pts.append((x + RADIUS, y + h))
        else:
            pts.append((x + w - RADIUS, y + h))
            pts.append((x + RADIUS, y + h))
        pts.append((x, y + h - RADIUS))
        pts.append((x, y + RADIUS))
        return pts

    def draw(self):
        self.clear()
        name, color, dark = CAT_MAP[self.spec["cat"]]
        shape = self.spec["shape"]
        x, y, w, h = self.x, self.y, self.w, self.h
        c = self.canvas

        if shape == SHAPE_CBLOCK:
            self._draw_cblock_top(c, x, y, w, h, color, dark)
            self._draw_cblock_bottom(c, x, y, w, h, color, dark)
            self._bottom_items = list(self.items[-5:])
        else:
            self._bottom_items = []
            pts = self._outline(x, y, w, h,
                                top_notch=(shape != SHAPE_HAT),
                                bottom_notch=(shape != SHAPE_CAP))
            self.items.append(c.create_polygon(*self._flat(pts), fill=color,
                                               outline=dark, width=1, smooth=False))
            if shape != SHAPE_HAT:
                self.items.append(c.create_line(x + 8, y + 2.5, x + w - 8, y + 2.5,
                                                fill=lighten(color, 0.32), width=2,
                                                capstyle="round"))

        self._draw_content(c, x, y, color)

    def _draw_cblock_top(self, c, x, y, w, h, color, dark):
        """C 型：顶部横条 + 左侧竖臂。"""
        hh = self.head_h
        arm = INNER_INDENT
        top_pts = []
        top_pts.append((x + RADIUS, y))
        nx0 = x + PAD_L - 2
        top_pts.append((nx0, y))
        top_pts.append((nx0 + NOTCH_D, y + NOTCH_D))
        top_pts.append((nx0 + NOTCH_W + NOTCH_D, y + NOTCH_D))
        top_pts.append((nx0 + NOTCH_W + NOTCH_D * 2, y))
        top_pts.append((x + w - RADIUS, y))
        top_pts.append((x + w, y + RADIUS))
        top_pts.append((x + w, y + hh))
        top_pts.append((x + arm, y + hh))
        top_pts.append((x + arm, y + RADIUS))
        top_pts.append((x, y + RADIUS))
        self.items.append(c.create_polygon(*self._flat(top_pts), fill=color,
                                           outline=dark, width=1, smooth=False))
        self.items.append(c.create_line(x + 8, y + 2.5, x + w - 8, y + 2.5,
                                        fill=lighten(color, 0.32), width=2,
                                        capstyle="round"))
        arm_pts = [(x, y + RADIUS), (x + arm, y + RADIUS),
                   (x + arm, y + h - 8), (x, y + h - 8)]
        self.items.append(c.create_polygon(*self._flat(arm_pts), fill=color,
                                           outline=dark, width=1, smooth=False))

    def _draw_cblock_bottom(self, c, x, y, w, h, color, dark):
        """C 型：底部横条（含底部凸起）。"""
        bot = 8
        by = y + h - bot
        nx0 = x + PAD_L - 2
        bot_pts = [(x, by), (x + w, by), (x + w, y + h - RADIUS),
                   (x + w - RADIUS, y + h)]
        bot_pts.append((nx0 + NOTCH_W + NOTCH_D * 2, y + h))
        bot_pts.append((nx0 + NOTCH_W + NOTCH_D, y + h + NOTCH_D))
        bot_pts.append((nx0 + NOTCH_D, y + h + NOTCH_D))
        bot_pts.append((nx0, y + h))
        bot_pts.append((x + RADIUS, y + h))
        bot_pts.append((x, y + h - RADIUS))
        self.items.append(c.create_polygon(*self._flat(bot_pts), fill=color,
                                           outline=dark, width=1, smooth=False))
        self.items.append(c.create_line(x + 8, by + 2, x + w - 8, by + 2,
                                        fill=lighten(color, 0.32), width=1))

    def _flat(self, pts):
        out = []
        for p in pts:
            out.extend(p)
        return out

    def _draw_content(self, c, x, y, color):
        cx = x + PAD_L
        is_hat = self.spec["shape"] == SHAPE_HAT
        cy = y + (self.head_h + (6 if is_hat else 0)) / 2
        self.input_regions = []
        for seg in self.segs:
            if seg[0] == "t":
                self.fv_items.append(c.create_text(cx, cy, text=seg[1], anchor="w",
                                                   fill=TEXT_WHITE, font=FONT_CN))
                cx += seg[2]
            else:
                idx = seg[1]
                iw = seg[2]
                default, itype, opts, _ = self.spec["inputs"][idx]
                iy = cy - 10
                val = self.values[idx]
                if itype == IN_COLOR:
                    self._draw_color(c, cx, cy, iw, self.values[idx])
                    self.input_regions.append((cx, iy, cx + iw, iy + 20,
                                               idx, IN_COLOR, self.values[idx]))
                else:
                    if itype == IN_SELECT and opts:
                        shown = _T(str(dict(opts).get(val, val)))
                    elif itype == IN_PATH:
                        shown = str(val).split("/")[-1].split("\\")[-1] or _T("选择文件…")
                    else:
                        shown = str(val)
                    self._draw_round_box(c, cx, iy, iw, 20)
                    self.fv_items.append(c.create_text(
                        cx + iw / 2, cy, text=shown, anchor="center",
                        fill=TEXT_DARK, font=FONT_MONO if itype == IN_NUMBER else FONT_CN_S))
                    self.input_regions.append((cx, iy, cx + iw, iy + 20, idx, itype, str(val)))
                cx += iw + 4

    def _draw_round_box(self, c, x, y, w, h, fill="#FFFFFF"):
        self.fv_items.append(c.create_polygon(
            *self._flat([(x + 4, y), (x + w - 4, y), (x + w, y + 4),
                         (x + w, y + h - 4), (x + w - 4, y + h),
                         (x + 4, y + h), (x, y + h - 4), (x, y + 4)]),
            fill=fill, outline="", width=0))

    def _draw_color(self, c, cx, cy, iw, val):
        self._draw_round_box(c, cx, cy - 10, iw, 20)
        self.fv_items.append(c.create_rectangle(cx + 4, cy - 6, cx + iw - 4, cy + 6,
                                                fill=val, outline=darken(val, 0.8), width=1))

    def hit_input(self, cx, cy):
        """命中测试：返回 (idx, itype, val) 或 None。"""
        for (x1, y1, x2, y2, idx, itype, val) in self.input_regions:
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                return (idx, itype, val)
        return None

    def redraw_values(self):
        """只重绘前景（文字/输入框值），不动积木轮廓，用于输入后即时刷新。"""
        for i in self.fv_items:
            try:
                self.canvas.delete(i)
            except tk.TclError:
                pass
        self.fv_items = []
        color = CAT_MAP[self.spec["cat"]][1]
        self._draw_content(self.canvas, self.x, self.y, color)

    def clear(self):
        for i in list(self.items) + list(self.fv_items):
            try:
                self.canvas.delete(i)
            except tk.TclError:
                pass
        self.items = []
        self.fv_items = []
        self.input_regions = []

    def serialize(self):
        d = {"kind": self.kind, "values": [str(v) for v in self.values]}
        if self.cond is not None:
            d["cond"] = self.cond
        if self.children:
            d["children"] = [c.serialize() for c in self.children]
        if self.next_block:
            d["next"] = self.next_block.serialize()
        return d

    @staticmethod
    def _deserialize_chain(app, data, x, y, suppress=False):
        first = None
        prev = None
        cur_data = data
        while cur_data is not None:
            b = BlockView(app, cur_data["kind"], x, y,
                          values=cur_data.get("values"),
                          cond=cur_data.get("cond"))
            for cd in cur_data.get("children", []):
                b.children.append(BlockView._deserialize_chain(
                    app, cd, x + INNER_INDENT, y))
                b.children[-1].slot_owner = b
            if first is None:
                first = b
            if prev is not None:
                prev.next_block = b
                b.parent = prev
            prev = b
            cur_data = cur_data.get("next")
        if first:
            first.relayout_all()
        return first

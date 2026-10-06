# -*- coding: utf-8 -*-
"""AI Blocks —— 用积木搭建 AI 模型的生产力工具。

四大板块：神经网络 / 马尔可夫链 / 预设回答 / 经典 AI。
积木拖拽引擎复用自 TurtleBlocks，运行逻辑替换为真实 AI 训练。
"""

import os
import sys
import csv
import random
import tkinter as tk
from tkinter import filedialog, messagebox

from themes import (CATEGORIES, CAT_MAP, WORKSPACE_BG, PALETTE_BG,
                    PANEL_BORDER, TEXT_DARK, CANVAS_BG, DASH_BG, DASH_PANEL,
                    DASH_BORDER, DASH_TEXT, DASH_MUTED, ACCENT, ACCENT_2)
from blocks_def import (merged_blocks, BOARDS, BOARD_ORDER, board_text,
                        SHAPE_HAT, SHAPE_CBLOCK, SHAPE_CAP,
                        IN_NUMBER, IN_TEXT, IN_SELECT, IN_PATH)
import block_view as bv
from block_view import BlockView, INNER_INDENT
import ai_core as A
import visuals as V
import i18n

APP_NAME = "AI Blocks"
VERSION = "1.0"

INFER_SCRIPT = '''# -*- coding: utf-8 -*-
"""Standalone inference for an AI Blocks model. Only needs the standard library.

Usage:
    python model_infer.py            # interactive
    python model_infer.py 1.5 2.0    # predict directly
"""
import json
import math
import sys

def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _act(name, v):
    if name == "relu":
        return v if v > 0 else 0.0
    if name == "tanh":
        return math.tanh(v)
    return 1.0 / (1.0 + math.exp(-v)) if -60 < v < 60 else (0.0 if v <= -60 else 1.0)

def mlp_forward(model, x):
    a = list(x)
    W = model["weights"]
    b = model["biases"]
    hidden = model["hidden"]
    for li, (units, act) in enumerate(hidden):
        z = [sum(a[i] * W[li][i][j] for i in range(len(a))) + b[li][j]
             for j in range(units)]
        a = [_act(act, v) for v in z]
    li = len(hidden)
    z = [sum(a[i] * W[li][i][j] for i in range(len(a))) + b[li][j]
         for j in range(model["n_out"])]
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e) or 1e-9
    return [v / s for v in e]

def main():
    path = "model.json"
    payload = load(path)
    model = payload["model"]
    t = model.get("type")
    args = sys.argv[1:]

    def one(q):
        if t == "mlp":
            x = [float(v) for v in q.replace(",", " ").split()][:2]
            p = mlp_forward(model, x)
            cls = max(range(len(p)), key=lambda i: p[i])
            return "class %d  (confidence %.1f%%)" % (cls, p[cls] * 100)
        if t == "knn":
            x = [float(v) for v in q.replace(",", " ").split()][:2]
            d = sorted((math.hypot(px - x[0], py - x[1]), i)
                       for i, (px, py) in enumerate(model["X"]))
            votes = {}
            for _, i in d[:model["k"]]:
                y = model["y"][i]
                votes[y] = votes.get(y, 0) + 1
            return "class %d" % max(votes, key=votes.get)
        if t == "qabot":
            ql = q.lower()
            best = (None, 0.0)
            for r in model["rules"]:
                kws = r["keywords"]
                sc = sum(1 for k in kws if k and k in q) / (len(kws) or 1)
                if sc > best[1]:
                    best = (r["answer"], sc)
            return best[0] if best[0] is not None else "(no match)"
        if t == "markov":
            order = model["order"]
            trans = {tuple(k.split(" ")): v for k, v in model["transitions"].items()}
            import random
            seed = list(q)
            if len(seed) < order and trans:
                seed = list(next(iter(trans)))
            out = seed[:]
            for _ in range(40):
                opts = trans.get(tuple(out[-order:]))
                if not opts:
                    import random as _r
                    k = _r.choice(list(trans))
                    out = list(k)
                    continue
                toks = list(opts)
                ws = [opts[x] for x in toks]
                tot = sum(ws)
                r = random.random() * tot
                acc = 0.0
                pick = toks[-1]
                for x, w in zip(toks, ws):
                    acc += w
                    if r <= acc:
                        pick = x
                        break
                out.append(pick)
            return "".join(out)
        return "(unsupported model type: %s)" % t

    if args:
        print(one(" ".join(args)))
        return
    print("Loaded %s model. Type a question, Ctrl+C to quit." % t)
    while True:
        try:
            q = input("> ")
        except (EOFError, KeyboardInterrupt):
            break
        print(one(q))

if __name__ == "__main__":
    main()
'''

def T(s):
    """翻译 helper。"""
    return i18n.t(s)

PALETTE_W = 250
HEADER_H = 54
CATBAR_H = 34
DASH_W = 360

class DashSelect(tk.Frame):
    """首页：四个大板块卡片的入口。"""

    def __init__(self, master, on_pick, on_lang=None):
        super().__init__(master, bg="#0B1220")
        self.on_pick = on_pick
        self.on_lang = on_lang

        lang_bar = tk.Frame(self, bg="#0B1220")
        lang_bar.place(relx=1.0, rely=0.0, x=-24, y=20, anchor="ne")
        cur = i18n.get_lang()
        self._lang_chips = {}
        for code, label in ((i18n.ZH, "中文"), (i18n.EN, "English")):
            on = (code == cur)
            chip = tk.Label(lang_bar, text=label,
                            bg="#1B2A4A" if on else "#141E36",
                            fg=DASH_TEXT if on else DASH_MUTED,
                            font=("Microsoft YaHei UI", 10, "bold"),
                            padx=12, pady=5, cursor="hand2")
            chip.pack(side="left", padx=3)
            chip.bind("<Button-1>", lambda e, c=code: self._pick_lang(c))
            self._lang_chips[code] = chip

        tk.Label(self, text="AI Blocks", bg="#0B1220", fg=DASH_TEXT,
                 font=("Microsoft YaHei UI", 30, "bold")).pack(pady=(60, 4))
        tk.Label(self, text=T("用积木亲手搭出一个 AI，然后看着它从数据里学出来"),
                 bg="#0B1220", fg=DASH_MUTED,
                 font=("Microsoft YaHei UI", 12)).pack(pady=(0, 8))
        tk.Label(self, text=T("拖动积木 → 拼接模型 → 选择数据 → 运行 → 实时看它学习"),
                 bg="#0B1220", fg="#5A6B8C",
                 font=("Microsoft YaHei UI", 10)).pack(pady=(0, 34))

        wrap = tk.Frame(self, bg="#0B1220")
        wrap.pack()

        CARD_BG = "#141E36"

        for i, bid in enumerate(BOARD_ORDER):
            b = BOARDS[bid]
            card = tk.Frame(wrap, bg=CARD_BG, highlightbackground="#233153",
                            highlightthickness=1, cursor="hand2")
            card.grid(row=0, column=i, padx=8, sticky="nsew")
            card.configure(width=215, height=282)
            card.grid_propagate(False)

            strip = tk.Frame(card, bg=b["color"], height=6)
            strip.pack(fill="x")
            ico = tk.Label(card, text=b["icon"], bg=CARD_BG, fg=b["color"],
                           font=("Segoe UI Emoji", 30))
            ico.pack(pady=(20, 6))
            nm = tk.Label(card, text=T(b["name"]), bg=CARD_BG, fg=DASH_TEXT,
                          font=("Microsoft YaHei UI", 15, "bold"))
            nm.pack()
            sb = tk.Label(card, text=T(b["sub"]), bg=CARD_BG, fg=DASH_MUTED,
                          font=("Microsoft YaHei UI", 9), wraplength=182,
                          justify="center")
            sb.pack(pady=(8, 10), padx=12)
            go = tk.Label(card, text=T("进入 →"), bg=CARD_BG, fg=b["color"],
                          font=("Microsoft YaHei UI", 11, "bold"))
            go.pack(side="bottom", pady=14)

            for w in (card, strip, ico, nm, sb, go):
                w.bind("<Button-1>", lambda e, k=bid: self.on_pick(k))

        for c in range(len(BOARD_ORDER)):
            wrap.grid_columnconfigure(c, weight=1)

    def _pick_lang(self, code):
        """点了语言切换：交给上层重建界面，让文案立刻用新语言。"""
        if code == i18n.get_lang():
            return
        i18n.set_lang(code)
        i18n.save_pref()
        if self.on_lang:
            self.on_lang()

class App:
    def __init__(self, root):
        self.root = root
        self.root.title(f"{APP_NAME}  v{VERSION}")
        self.root.geometry("1440x900")
        self.root.minsize(1200, 760)
        self.root.configure(bg=WORKSPACE_BG)

        self.board = None
        self.palette_views = []
        self.workspace_blocks = []
        self.drag = None
        self.active_cat = None
        self.running = False
        self.script_gen = None

        self.ctx = {}
        self._training_job = None

        self._saved_script = None

        self.select_screen = DashSelect(self.root, self.enter_board,
                                        on_lang=self.on_lang_changed)
        self.select_screen.pack(fill="both", expand=True)

    def on_lang_changed(self):
        """语言切换回调：首页则重建首页；板块页则重建板块并保留积木。"""
        if self.board is None:
            self._rebuild_menu()
            return
        self._saved_script = self._serialize_workspace()
        bid = self.board
        self.board = None
        self._rebuild_board(bid)

    def _rebuild_menu(self):
        """重建首页（语言切换后用）。重建期间冻住窗口，避免闪烁。"""
        self.root.withdraw()
        try:
            for w in list(self.root.winfo_children()):
                w.destroy()
            self.select_screen = DashSelect(self.root, self.enter_board,
                                            on_lang=self.on_lang_changed)
            self.select_screen.pack(fill="both", expand=True)
        finally:
            self.root.deiconify()
            self.root.update_idletasks()

    def _rebuild_board(self, bid):
        """按当前语言重建板块界面，并恢复积木。
        重建期间冻结绘制，避免出现"空窗口闪一下"。"""
        bv.set_blocks(merged_blocks(bid))
        self.root.withdraw()
        try:
            for w in list(self.root.winfo_children()):
                w.destroy()
            self.board = bid
            self.palette_views = []
            self.workspace_blocks = []
            self._reset_ctx()
            self._build_ui()
            self._build_palette()
            self.select_category(BOARDS[bid]["cats"][0])
            if self._saved_script:
                self._restore_all(self._saved_script)
                self._saved_script = None
                for b in self.workspace_blocks:
                    b.relayout_all()
        finally:
            self.root.deiconify()
            self.root.update_idletasks()

    def enter_board(self, bid):
        try:
            self.on_stop()
        except Exception:
            pass
        if self._training_job:
            try:
                self.root.after_cancel(self._training_job)
            except Exception:
                pass
            self._training_job = None
        self.running = False
        self.script_gen = None
        self.board = bid
        bv.set_blocks(merged_blocks(bid))
        self.select_screen.pack_forget()
        self.select_screen.destroy()
        for w in list(self.root.winfo_children()):
            w.destroy()
        self._reset_ctx()
        self._build_ui()
        self._build_palette()
        self.select_category(BOARDS[bid]["cats"][0])
        if not hasattr(self, "_tick_started"):
            self._tick_started = True
            self.root.after(30, self._tick)

    def _reset_ctx(self):
        self._hint_kind = None
        self._chat_closed = True
        self._chat_bar = None
        self.ctx = {
            "dataset": None,
            "model": None,
            "layers": [],
            "lr": 0.1,
            "epochs": 200,
            "train_ratio": 0.8,
            "split_ratio": 0.8,
            "mm": A.Markov(),
            "qa": A.QABot(),
            "knn": A.KNN(),
            "km": A.KMeans(),
            "ga": A.GA(),
            "recog": None,
            "recog_kind": "",
            "qa_history": [],
            "log": [],
            "trained": False,
        }

    def _switch_lang(self, code):
        """板块页切换语言：保存积木 → 重建界面 → 恢复积木。"""
        if code == i18n.get_lang():
            return
        i18n.set_lang(code)
        i18n.save_pref()
        self.on_lang_changed()

    def _build_ui(self):
        b = BOARDS[self.board]
        self.log_text = None
        header = tk.Frame(self.root, bg="#0F172A", height=HEADER_H)
        header.pack(side="top", fill="x")
        header.pack_propagate(False)

        tk.Label(header, text=f"{b['icon']} {APP_NAME}", bg="#0F172A", fg=b["color"],
                 font=("Microsoft YaHei UI", 14, "bold")).pack(side="left", padx=(16, 10))
        tk.Label(header, text=T(b["name"]), bg="#0F172A", fg=DASH_TEXT,
                 font=("Microsoft YaHei UI", 12, "bold")).pack(side="left")

        def tbtn(text, cmd, bg=ACCENT, width=None):
            bb = tk.Button(header, text=text, command=cmd, bg=bg, fg="white",
                           activebackground=bv.darken(bg), activeforeground="white",
                           font=("Microsoft YaHei UI", 10), relief="flat",
                           padx=14, pady=6, cursor="hand2", bd=0)
            if width:
                bb.configure(width=width)
            bb.pack(side="left", padx=4, pady=10)
            return bb

        tk.Frame(header, bg="#0F172A", width=14).pack(side="left")
        tbtn("▶  " + T("运行"), self.on_run, bg="#22C55E")
        tbtn("■  " + T("停止"), self.on_stop, bg="#E24B4A")
        tk.Frame(header, bg="#233153", width=1).pack(side="left", fill="y", pady=12, padx=8)
        tbtn(T("清空") + " " + T("积木"), self.on_clear, bg="#475569")
        tbtn(T("示例"), self.on_demo, bg="#475569")
        tbtn(T("保存"), self.on_save, bg="#475569")
        tbtn(T("打开"), self.on_open, bg="#475569")
        tk.Frame(header, bg="#233153", width=1).pack(side="left", fill="y", pady=12, padx=8)
        tbtn("← " + T("返回"), self.back_to_menu, bg="#6D4AB8")

        cur = i18n.get_lang()
        lang_pill = tk.Frame(header, bg="#0F172A")
        lang_pill.pack(side="right", padx=(0, 14))
        for code, label in ((i18n.ZH, "中"), (i18n.EN, "EN")):
            on = (code == cur)
            chip = tk.Label(lang_pill, text=label,
                            bg="#1B2A4A" if on else "#141E36",
                            fg=DASH_TEXT if on else DASH_MUTED,
                            font=("Microsoft YaHei UI", 9, "bold"),
                            padx=8, pady=3, cursor="hand2")
            chip.pack(side="left", padx=2)
            chip.bind("<Button-1>", lambda e, c=code: self._switch_lang(c))

        self.status = tk.Label(header, text=T("就绪"), bg="#0F172A", fg=DASH_MUTED,
                               font=("Microsoft YaHei UI", 9))
        self.status.pack(side="right", padx=16)

        body = tk.Frame(self.root, bg=WORKSPACE_BG)
        body.pack(side="top", fill="both", expand=True)

        self.catbar = tk.Frame(body, bg="#FFFFFF", width=66,
                               highlightbackground=PANEL_BORDER, highlightthickness=1)
        self.catbar.pack(side="left", fill="y")
        self.catbar.pack_propagate(False)
        self.cat_buttons = {}
        for key in BOARDS[self.board]["cats"]:
            name, color, dark = CAT_MAP[key]
            btn = tk.Frame(self.catbar, bg="#FFFFFF", cursor="hand2")
            btn.pack(fill="x", pady=2, padx=3)
            dot = tk.Label(btn, text="●", bg="#FFFFFF", fg=color,
                           font=("Microsoft YaHei UI", 12))
            dot.pack(side="left", padx=(6, 2))
            lbl = tk.Label(btn, text=T(name), bg="#FFFFFF", fg=TEXT_DARK,
                           font=("Microsoft YaHei UI", 10))
            lbl.pack(side="left")
            for w in (btn, dot, lbl):
                w.bind("<Button-1>", lambda e, k=key: self.select_category(k))
            self.cat_buttons[key] = (btn, dot, lbl)

        pf = tk.Frame(body, bg=PALETTE_BG, width=PALETTE_W,
                      highlightbackground=PANEL_BORDER, highlightthickness=1)
        pf.pack(side="left", fill="y")
        pf.pack_propagate(False)

        self.hint_bar = tk.Label(pf, text=T("把积木拖到中间试试"),
                                 bg="#EEF4FF", fg="#4A5A78",
                                 font=("Microsoft YaHei UI", 9),
                                 wraplength=PALETTE_W - 26, justify="left",
                                 anchor="w", padx=10, pady=8)
        self.hint_bar.pack(side="bottom", fill="x")

        self.palette_canvas = tk.Canvas(pf, bg=PALETTE_BG, highlightthickness=0,
                                        width=PALETTE_W - 16)
        pf_scroll = tk.Scrollbar(pf, orient="vertical", command=self.palette_canvas.yview)
        self.palette_canvas.configure(yscrollcommand=pf_scroll.set)
        pf_scroll.pack(side="right", fill="y")
        self.palette_canvas.pack(side="left", fill="both", expand=True)
        self.palette_canvas.configure(scrollregion=(0, 0, PALETTE_W, 2000))
        self.palette_canvas.bind("<ButtonPress-1>", self.on_palette_press)
        self.palette_canvas.bind("<B1-Motion>", self.on_palette_drag)
        self.palette_canvas.bind("<ButtonRelease-1>", self.on_palette_release)
        self.palette_canvas.bind("<MouseWheel>", self._on_palette_wheel)
        self.palette_canvas.bind("<Motion>", self.on_palette_motion)
        self.palette_canvas.bind("<Leave>", lambda e: self._set_hint(None))

        mid = tk.Frame(body, bg=WORKSPACE_BG)
        mid.pack(side="left", fill="both", expand=True)
        self.ws_canvas = tk.Canvas(mid, bg=WORKSPACE_BG, highlightthickness=0)
        hscroll = tk.Scrollbar(mid, orient="horizontal", command=self.ws_canvas.xview)
        vscroll = tk.Scrollbar(mid, orient="vertical", command=self.ws_canvas.yview)
        self.ws_canvas.configure(xscrollcommand=hscroll.set, yscrollcommand=vscroll.set,
                                 scrollregion=(0, 0, 2600, 2000))
        hscroll.pack(side="bottom", fill="x")
        vscroll.pack(side="right", fill="y")
        self.ws_canvas.pack(side="left", fill="both", expand=True)
        self._draw_grid(self.ws_canvas, 2600, 2000)

        self.ws_canvas.bind("<ButtonPress-1>", self.on_ws_press)
        self.ws_canvas.bind("<B1-Motion>", self.on_ws_drag)
        self.ws_canvas.bind("<ButtonRelease-1>", self.on_ws_release)
        self.ws_canvas.bind("<MouseWheel>", self._on_wheel)
        self.ws_canvas.bind("<Button-3>", self.on_ws_right)

        dash = tk.Frame(body, bg=DASH_BG, width=DASH_W,
                        highlightbackground="#233153", highlightthickness=1)
        dash.pack(side="right", fill="y")
        dash.pack_propagate(False)
        tk.Label(dash, text=T("可视化"), bg=DASH_BG, fg=DASH_TEXT,
                 font=("Microsoft YaHei UI", 11, "bold")).pack(anchor="w", padx=12, pady=(10, 4))
        self.dash_canvas = tk.Canvas(dash, bg=DASH_BG, highlightthickness=0,
                                     width=DASH_W - 24, height=430)
        self.dash_canvas.pack(padx=12, fill="both", expand=True)
        self.dash = self._make_dash()

        out_head = tk.Frame(dash, bg=DASH_BG)
        out_head.pack(fill="x", padx=12, pady=(10, 3))
        tk.Label(out_head, text=T("输出"), bg=DASH_BG, fg=DASH_TEXT,
                 font=("Microsoft YaHei UI", 11, "bold")).pack(side="left")
        clr = tk.Label(out_head, text=T("清空"), bg="#2A3A5A", fg=DASH_MUTED,
                       font=("Microsoft YaHei UI", 8), padx=7, pady=1, cursor="hand2")
        clr.pack(side="right")
        clr.bind("<Button-1>", lambda e: self._clear_log())

        log_wrap = tk.Frame(dash, bg=DASH_PANEL,
                            highlightbackground="#233153", highlightthickness=1)
        log_wrap.pack(fill="both", expand=False, padx=12, pady=(0, 12))
        self.log_text = tk.Text(log_wrap, height=12, bg=DASH_PANEL, fg=DASH_TEXT,
                                font=("Consolas", 8), relief="flat", wrap="word",
                                insertbackground=DASH_TEXT, padx=7, pady=5,
                                selectbackground="#2A3A5A")
        log_scroll = tk.Scrollbar(log_wrap, orient="vertical",
                                  command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        log_scroll.pack(side="right", fill="y")
        self.log_text.pack(side="left", fill="both", expand=True)
        self.log_text.tag_config("ok", foreground="#22C55E")
        self.log_text.tag_config("warn", foreground="#FBBF24")
        self.log_text.tag_config("ask", foreground="#93B4FF")
        self.log_text.tag_config("ans", foreground="#E8EEFF")
        self.log_text.configure(state="disabled")

    def _make_dash(self):
        """按板块创建可视化视图集合。"""
        c = self.dash_canvas
        W = DASH_W - 28
        bid = self.board
        views = {}
        if bid == "nn":
            views["net"] = V.NetView(c, 0, 0, W, 210, T("网络结构（亮度=激活值）"))
            views["loss"] = V.LineChart(c, 0, 220, W, 400, T("损失曲线 Loss"), 0, 2.5)
            views["acc"] = V.LineChart(c, 0, 410, W, 560, T("准确率 Accuracy"), 0, 1,
                                       y_fmt="{:.2f}")
            views["scatter"] = V.ScatterView(c, 0, 570, W, 730, T("决策边界 + 数据点"))
        elif bid == "mm":
            views["heat"] = V.HeatmapView(c, 0, 0, W, 330, T("转移概率矩阵（行→列）"))
            views["gen"] = V.LineChart(c, 0, 340, W, 520, T("生成进度"), 0, 1)
        elif bid == "qa":
            views["qa"] = V.QAPanel(c, 0, 0, W, 340, T("回答"))
        elif bid == "classic":
            views["scatter"] = V.ScatterView(c, 0, 0, W, 380, T("数据 / 决策边界"))
            views["curve"] = V.LineChart(c, 0, 390, W, 620, T("指标曲线"), 0, 1)
        elif bid in ("audio", "image"):
            title = (T("各类别识别概率") if bid == "audio"
                     else T("各类别识别概率"))
            views["recog"] = V.RecogBars(c, 0, 0, W, 320, title)
        return views

    GRID = 26

    def _draw_grid(self, canvas, w, h, step=GRID):
        for x in range(step, w, step):
            for y in range(step, h, step):
                canvas.create_oval(x - 1.5, y - 1.5, x + 1.5, y + 1.5,
                                   fill="#DCE3EC", outline="", tags="grid")
        canvas.tag_lower("grid")

    def select_category(self, key):
        self.active_cat = key
        for k, (btn, dot, lbl) in self.cat_buttons.items():
            if k == key:
                for w in (btn, dot, lbl):
                    w.configure(bg="#EEF4FF")
                lbl.configure(fg=CAT_MAP[k][1])
            else:
                for w in (btn, dot, lbl):
                    w.configure(bg="#FFFFFF")
                lbl.configure(fg=TEXT_DARK)
        self._build_palette()

    def _build_palette(self):
        c = self.palette_canvas
        for v in self.palette_views:
            v.clear()
        self.palette_views = []
        c.delete("all")
        from blocks_def import BOARDS as _B
        kinds = [k for k, sp in merged_blocks(self.board).items()
                 if sp["cat"] == self.active_cat]
        y = 16
        for k in kinds:
            v = BlockView(self, k, 14, y, palette=True, suppress_widgets=True)
            self.palette_views.append(v)
            y += v.h + 20
        c.configure(scrollregion=(0, 0, PALETTE_W - 16, max(y + 20, 500)))
        c.yview_moveto(0)

    def _on_palette_wheel(self, e):
        self.palette_canvas.yview_scroll(int(-e.delta / 120), "units")

    def _set_hint(self, view):
        """更新调色板底部说明条。view=None 时显示默认提示。"""
        try:
            if view is None:
                self.hint_bar.configure(text=T("把积木拖到中间试试"), fg="#4A5A78")
                return
            hint = view.spec.get("hint") or T("（暂无说明）")
            self.hint_bar.configure(text=T(hint), fg="#2C3E5C")
        except tk.TclError:
            pass

    def on_palette_motion(self, event):
        """悬停在调色板积木上时，在底部显示它的用途说明。"""
        cx = self.palette_canvas.canvasx(event.x)
        cy = self.palette_canvas.canvasy(event.y)
        for v in self.palette_views:
            if v.x <= cx <= v.x + v.w and v.y <= cy <= v.y + v.h:
                if getattr(self, "_hint_kind", None) != v.kind:
                    self._hint_kind = v.kind
                    self._set_hint(v)
                return
        if getattr(self, "_hint_kind", None) is not None:
            self._hint_kind = None
            self._set_hint(None)

    def on_palette_press(self, event):
        if self.running:
            return
        cx = self.palette_canvas.canvasx(event.x)
        cy = self.palette_canvas.canvasy(event.y)
        for v in self.palette_views:
            if v.x <= cx <= v.x + v.w and v.y <= cy <= v.y + v.h:
                self.drag = {"kind": v.kind, "from": "palette", "x": cx, "y": cy,
                             "ghost": None}
                return

    def on_palette_drag(self, event):
        if not self.drag or self.drag.get("from") != "palette":
            return
        cx = self.palette_canvas.canvasx(event.x)
        cy = self.palette_canvas.canvasy(event.y)
        self.palette_canvas.delete("ghost")
        self.palette_canvas.create_rectangle(cx - 40, cy - 10, cx + 40, cy + 10,
                                             fill="#C9D6EA", outline="#8FA5C8",
                                             tags="ghost")

    def on_palette_release(self, event):
        if not self.drag or self.drag.get("from") != "palette":
            return
        kind = self.drag["kind"]
        self.palette_canvas.delete("ghost")
        self.drag = None
        vx = self.ws_canvas.canvasx(self.ws_canvas.winfo_width() / 2)
        vy = self.ws_canvas.canvasy(60)
        gx, gy = self._grid_snap(vx - 60, vy)
        self._spawn_block(kind, gx, gy)

    def _spawn_block(self, kind, x, y, values=None):
        b = BlockView(self, kind, x, y, values=values)
        self.workspace_blocks.append(b)
        self._raise_chain(b)
        return b

    def _raise_chain(self, block):
        """把整棵子树提到最上层：先背景，再前景（保证文字在最顶）。"""
        nodes = list(self._iter_all(block))
        for bb in nodes:
            for i in bb.items:
                try:
                    self.ws_canvas.tag_raise(i)
                except tk.TclError:
                    pass
        for bb in nodes:
            for i in bb.fv_items:
                try:
                    self.ws_canvas.tag_raise(i)
                except tk.TclError:
                    pass

    def _iter_all(self, b):
        """遍历一块积木及其整棵子树。"""
        yield b
        for c in b.children:
            yield from self._iter_all(c)
        if b.next_block is not None:
            yield from self._iter_all(b.next_block)

    def _poly_of(self, v):
        xs, ys = [], []
        for i in v.items:
            bb = self.ws_canvas.bbox(i)
            if bb:
                xs += [bb[0], bb[2]]
                ys += [bb[1], bb[3]]
        if not xs:
            return []
        return [(min(xs), min(ys)), (max(xs), min(ys)),
                (max(xs), max(ys)), (min(xs), max(ys))]

    def _point_in_poly(self, x, y, poly):
        if not poly:
            return False
        for i in range(len(poly)):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % len(poly)]
            if ((y1 > y) != (y2 > y)) and \
               (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-9) + x1):
                return True
        return False

    def _hit_block(self, cx, cy):
        for b in reversed(self.workspace_blocks):
            for bb in self._iter_all(b):
                if bb.x <= cx <= bb.x + bb.w and bb.y <= cy <= bb.y + bb.h:
                    if self._point_in_poly(cx, cy, self._poly_of(bb)):
                        return bb
        return None

    def _is_descendant(self, node, ancestor):
        if node is None or ancestor is None:
            return False
        p = node
        while p is not None:
            if p is ancestor:
                return True
            p = p.parent or p.slot_owner
        return False

    def on_ws_press(self, event):
        if self.running:
            return
        cx = self.ws_canvas.canvasx(event.x)
        cy = self.ws_canvas.canvasy(event.y)
        blk = self._hit_block(cx, cy)
        if blk is None:
            self.drag = {"mode": "pan", "x": event.x, "y": event.y}
            self.ws_canvas.scan_mark(event.x, event.y)
            return
        hit = blk.hit_input(cx, cy)
        if hit is not None:
            self._begin_inline_edit(blk, hit, cx, cy)
            return
        self._detach(blk)
        subtree = set()
        for x in self._iter_all(blk):
            subtree.add(id(x))
        self.workspace_blocks = [w for w in self.workspace_blocks
                                 if w is not blk and id(w) not in subtree]
        self.workspace_blocks.append(blk)
        self.drag = {"mode": "move", "block": blk, "off_x": cx - blk.x,
                     "off_y": cy - blk.y, "target": None}
        self._raise_chain(blk)
        self.ws_canvas.configure(cursor="fleur")

    def on_ws_drag(self, event):
        if not self.drag:
            return
        if self.drag["mode"] == "pan":
            self.ws_canvas.scan_dragto(event.x, event.y, gain=1)
            return
        cx = self.ws_canvas.canvasx(event.x)
        cy = self.ws_canvas.canvasy(event.y)
        blk = self.drag["block"]
        nx = cx - self.drag["off_x"]
        ny = cy - self.drag["off_y"]
        t = self._find_snap(blk, nx, ny)
        self.drag["target"] = t
        if t is not None:
            _k, _tg, _v, _m, sx, sy = t
            self._move_chain_to(blk, sx, sy)
        else:
            gx, gy = self._grid_snap(nx, ny)
            self._move_chain_to(blk, gx, gy)

    def on_ws_release(self, event):
        if not self.drag:
            return
        if self.drag.get("mode") == "pan":
            self.drag = None
            return
        self._apply_snap(self.drag.get("target"))
        self.drag = None
        self.ws_canvas.configure(cursor="")

    def on_ws_right(self, event):
        cx = self.ws_canvas.canvasx(event.x)
        cy = self.ws_canvas.canvasy(event.y)
        blk = self._hit_block(cx, cy)
        if blk:
            self._detach(blk)
            if blk in self.workspace_blocks:
                self.workspace_blocks.remove(blk)
            for bb in self._iter_all(blk):
                bb.clear()
            self._normalize_roots()

    def _move_chain(self, block, dx, dy):
        b = block
        while b is not None:
            b.x += dx
            b.y += dy
            for i in b.items + b.fv_items:
                try:
                    self.ws_canvas.move(i, dx, dy)
                except tk.TclError:
                    pass
            for c in b.children:
                self._move_chain(c, dx, dy)
            b = b.next_block

    def _move_chain_to(self, block, tx, ty):
        self._move_chain(block, tx - block.x, ty - block.y)

    def _grid_snap(self, x, y):
        g = self.GRID
        return round(x / g) * g, round(y / g) * g

    def _find_snap(self, blk, nx, ny):
        SNAP = 30
        best = None
        for w in self.workspace_blocks:
            for cand in self._iter_all(w):
                if cand is blk or self._is_descendant(cand, blk):
                    continue
                bx, by = cand.x, cand.y + cand.h
                d = ((nx - bx) ** 2 + (ny - by) ** 2) ** 0.5
                if d < SNAP and (best is None or d < best[0]):
                    best = (d, "block", cand, None, "below", bx, by)
                if cand.spec["shape"] == SHAPE_CBLOCK:
                    ix = cand.x + INNER_INDENT
                    last = self._last_child(cand)
                    iy = (last.y + last.h) if last else (cand.y + cand.head_h + 5)
                    d = ((nx - ix) ** 2 + (ny - iy) ** 2) ** 0.5
                    if d < SNAP and (best is None or d < best[0]):
                        best = (d, "block", cand, None, "inside", ix, iy)
        if best is None:
            return None
        _d, kind, target, val, mode, sx, sy = best
        return (kind, target, val, mode, sx, sy)

    def _last_child(self, cblock):
        if not cblock.children:
            return None
        b = cblock.children[-1]
        while b.next_block is not None:
            b = b.next_block
        return b

    def _apply_snap(self, t):
        if t is None:
            self._normalize_roots()
            return
        _kind, target, _val, mode, sx, sy = t
        blk = self.drag["block"] if self.drag else None
        if blk is None:
            self._normalize_roots()
            return
        self._move_chain_to(blk, sx, sy)
        if mode == "below":
            old_next = target.next_block
            target.next_block = blk
            blk.parent = target
            if old_next is not None and old_next is not blk:
                tail = self._chain_tail(blk)
                if tail is not None and tail is not old_next:
                    tail.next_block = old_next
                    old_next.parent = tail
        else:
            blk.slot_owner = target
            target.children.append(blk)
            blk.parent = target
        top = blk
        guard = 0
        while (top.parent is not None or top.slot_owner is not None) and guard < 1000:
            top = top.parent or top.slot_owner
            guard += 1
        top.relayout_all()
        self._normalize_roots()

    def _chain_tail(self, b):
        guard = 0
        while b is not None and b.next_block is not None and guard < 1000:
            b = b.next_block
            guard += 1
        return b

    def _detach(self, block):
        for owner in (block.parent, block.slot_owner):
            if owner is None:
                continue
            if getattr(owner, "next_block", None) is block:
                owner.next_block = None
                owner.relayout_all()
            elif block in getattr(owner, "children", []):
                owner.children.remove(block)
                owner.relayout_all()
        block.parent = None
        block.slot_owner = None

    def _all_blocks(self):
        out = []
        for w in self.workspace_blocks:
            out.extend(self._iter_all(w))
        return out

    def _normalize_roots(self):
        roots = []
        for w in list(self.workspace_blocks):
            r = w
            guard = 0
            while guard < 1000:
                p = r.parent or r.slot_owner
                if p is None:
                    break
                attached = (getattr(p, "next_block", None) is r) or \
                           (r in getattr(p, "children", []))
                if not attached:
                    r.parent = None
                    r.slot_owner = None
                    break
                r = p
                guard += 1
            if r not in roots:
                roots.append(r)
        self.workspace_blocks = roots

    def _on_wheel(self, event):
        self.ws_canvas.yview_scroll(int(-event.delta / 120), "units")

    def _begin_inline_edit(self, blk, hit, cx, cy):
        idx, itype, val = hit
        if itype == IN_PATH:
            path = filedialog.askopenfilename(title=T("选择数据文件"),
                                              filetypes=[(T("数据文件"), "*.csv *.txt *.tsv"),
                                                         (T("所有文件"), "*.*")])
            if path:
                blk.values[idx] = path
                blk.redraw_values()
                self.status.configure(text=T("已选择：") + os.path.basename(path))
            return
        if itype == IN_SELECT:
            opts = blk.spec["inputs"][idx][2] or []
            if opts:
                cur = blk.values[idx]
                keys = [o[0] for o in opts]
                nxt = keys[(keys.index(cur) + 1) % len(keys)] if cur in keys else keys[0]
                blk.values[idx] = nxt
                blk.redraw_values()
            return
        ent = tk.Entry(self.ws_canvas, font=("Consolas", 11), justify="center",
                       relief="flat", bg="#FFFDE7", highlightthickness=1,
                       highlightbackground=ACCENT)
        ent.place(x=cx - 40, y=cy - 11, width=80, height=22)
        ent.insert(0, str(val))
        ent.select_range(0, "end")
        ent.focus_set()

        def _alive():
            try:
                return bool(ent.winfo_exists())
            except tk.TclError:
                return False

        def commit(e=None):
            if not _alive():
                return
            try:
                blk.values[idx] = ent.get()
            except tk.TclError:
                pass
            try:
                ent.destroy()
            except tk.TclError:
                pass
            blk.redraw_values()

        def cancel(e=None):
            if not _alive():
                return
            try:
                ent.destroy()
            except tk.TclError:
                pass

        ent.bind("<Return>", commit)
        ent.bind("<FocusOut>", commit)
        ent.bind("<Escape>", cancel)

    def on_run(self):
        if self.running:
            return
        hats = [b for b in self.workspace_blocks if b.spec["shape"] == SHAPE_HAT]
        if not hats:
            messagebox.showinfo(APP_NAME, T("工作区里没有「当点击运行时」积木。\n"
                                           "先从左侧拖一个出来放在最上面。"))
            return
        self._reset_ctx()
        self._clear_log()
        self.running = True
        self.status.configure(text=T("运行中…"), fg="#22C55E")
        self.script_gen = self._all_scripts(hats)

    def _all_scripts(self, hats):
        progs = [self._run_chain(h) for h in hats]
        while progs:
            alive = []
            for p in progs:
                try:
                    next(p)
                    alive.append(p)
                except StopIteration:
                    pass
                except Exception as ex:
                    self.status.configure(text=T("错误") + f": {ex}", fg="#E24B4A")
                    self._log("⚠ " + T("错误") + f": {ex}")
            progs = alive
            yield

    def _run_chain(self, blk):
        b = blk.next_block
        while b is not None:
            yield from self._exec_block(b)
            b = b.next_block

    def _exec_block(self, b):
        op = b.spec["opcode"]
        vals = b.values
        c = self.ctx

        def colval(i):
            try:
                return vals[i]
            except IndexError:
                return ""

        if op == "load_dataset":
            p = colval(0)
            if p and os.path.exists(p):
                c["dataset"] = A.load_csv_dataset(p)
            else:
                c["dataset"] = A.make_blobs(240, 2)
            self._log(T("载入数据集：") + f"{c['dataset'].source} ({len(c['dataset'])} " + T("条样本") + ")")
            self._refresh_dash()

        elif op == "set_split":
            c["train_ratio"] = float(colval(0)) / 100.0

        elif op == "mm_load":
            p = colval(0)
            if p and os.path.exists(p):
                with open(p, "r", encoding="utf-8") as f:
                    txt = f.read()
            else:
                txt = ("从前有座山，山里有座庙，庙里有个老和尚，"
                       "老和尚在给小和尚讲故事：从前有座山，山里有座庙……")
            c["mm"].order = int(colval(0)) if False else c["mm"].order
            c["mm"].load_text(txt)
            self._log(T("载入序列：") + f"{c['mm'].source}")

        elif op == "qa_load":
            p = colval(0)
            if p and os.path.exists(p):
                n = c["qa"].load_csv(p)
                self._log(T("载入问答库：") + f"{c['qa'].source}")
            else:
                c["qa"].strategy = "contains"
                for kw, ans in (("你好$%hi$%hello", "你好！我是规则机器人。"),
                                ("天气$%气温", "今天晴转多云，18~25 度。"),
                                ("名字$%你是谁", "我是预设回答机器人，靠关键词匹配。"),
                                ("几点$%时间$%现在", "现在是下午三点。")):
                    c["qa"].add_rule(T(kw), T(ans))
                self._log(T("载入内置问答库（4条）"))

        elif op == "qa_add":
            c["qa"].add_rule(colval(0), colval(1))
            self._log(T("添加规则：") + f"{colval(0)} → {colval(1)}")

        elif op == "knn_load" or op == "km_load":
            p = colval(0)
            if p and os.path.exists(p):
                c["dataset"] = A.load_csv_dataset(p)
            else:
                c["dataset"] = A.make_blobs(240, 3 if op == "km_load" else 2, seed=9)
            self._log(T("载入数据：") + f"{c['dataset'].source}")
            self._refresh_dash()

        elif op == "ga_load":
            top = colval(0)

        elif op == "add_layer":
            n = int(float(colval(0)))
            act = colval(1)
            c["layers"].append((n, act))
            self._log(T("添加隐藏层：") + f"{n} / {act}")
            self._update_net_view()

        elif op == "set_lr":
            c["lr"] = float(colval(0))

        elif op == "set_epochs":
            c["epochs"] = int(float(colval(0)))

        elif op == "init_model":
            if c["dataset"] is None:
                c["dataset"] = A.make_blobs(240, 2)
            m = A.MLP()
            for (n, act) in c["layers"]:
                m.add_layer(n, act)
            m.configure(c["dataset"], c["lr"], c["epochs"], c["train_ratio"])
            c["model"] = m
            self._log(T("构建模型：") + f"{[m.n_in] + [u for u,_ in m.hidden] + [m.n_out]}")
            self._update_net_view()

        elif op == "mm_order":
            c["mm"].order = int(float(colval(0)))

        elif op == "mm_unit":
            c["mm"].unit = colval(0)

        elif op == "knn_k":
            c["knn"].k = int(float(colval(0)))

        elif op == "km_k":
            c["km"].k = int(float(colval(0)))

        elif op == "ga_params":
            c["ga_params"] = (int(float(colval(0))), int(float(colval(1))), float(colval(2)))
            c["ga_func"] = c.get("ga_func", "quad")

        elif op == "train_model":
            m = c.get("model")
            if m is None:
                self._log("⚠ " + T("请先构建模型"))
            else:
                yield from self._drive_training(m, "nn")

        elif op == "mm_train":
            k = c["mm"].train()
            self._log(T("统计完成：") + f"{k}")
            self._refresh_dash()

        elif op == "knn_train":
            if c["dataset"] is not None:
                n = c["knn"].train(c["dataset"])
                self._log(T("KNN 记录 ") + f"{n} " + T("条样本") + f" (K={c['knn'].k})")
                self._refresh_dash()

        elif op == "km_train":
            if c["dataset"] is None:
                c["dataset"] = A.make_blobs(240, 3, seed=9)
            c["km"].configure(c["dataset"], c["km"].k)
            while True:
                s = c["km"].step()
                if s is None:
                    break
                self._update_km_view()
                yield
            self._log(T("K-Means 收敛：") + f"{c['km'].iter} iter, inertia={c['km'].inertia:.1f}")
            self._update_km_view()

        elif op == "ga_run":
            f = c.get("ga_func", "quad")
            pop, gens, mut = c.get("ga_params", (30, 60, 0.1))
            c["ga"].configure(f, pop, gens, mut)
            while True:
                s = c["ga"].step()
                if s is None:
                    break
                self._update_ga_view()
                yield
            self._log(T("GA 最优") + f" x={c['ga'].best_x:.3f}  f={c['ga'].best_f:.3f}")
            self._update_ga_view(done=True)

        elif op == "predict":
            m = c.get("model")
            if m is None:
                self._log("⚠ " + T("请先训练模型"))
            else:
                txt = colval(0)
                try:
                    xy = [float(t) for t in str(txt).replace(",", " ").split()][:2]
                    if len(xy) < 2:
                        raise ValueError
                    p = m.predict_proba(xy)
                    cls = max(range(len(p)), key=lambda i: p[i])
                    name = (c["dataset"].class_names[cls]
                            if c["dataset"] and cls < len(c["dataset"].class_names)
                            else str(cls))
                    conf = p[cls]
                    self._log(T("预测") + f" {xy} → 「{name}」 " + T("置信度") + f" {conf:.1%}")
                    self._update_predict_marker(xy, cls)
                except Exception:
                    self._log("⚠ " + T("预测输入应形如「1.5 2.0」") + f"，「{txt}」")
                self._refresh_dash()

        elif op == "mm_generate":
            n = int(float(colval(0)))
            temp = float(colval(1))
            seed_txt = c.get("mm_seed", "")
            txt, trace = c["mm"].generate(seed_txt, n, temp)
            self._log(T("生成：") + f"{txt}")
            self._refresh_dash()

        elif op == "mm_seed":
            c["mm_seed"] = colval(0)

        elif op == "qa_ask":
            q = colval(0)
            ans, hit, score, ok = c["qa"].answer(q)
            mark = T("✓命中") if ok else T("✗未命中")
            self._log(T("问：") + f"{q}")
            self._log(T("答：") + f"{ans}")
            self._log("     " + f"{mark}   " + T("分=") + f"{score:.2f}   "
                      + T("命中词=") + f"{hit}")
            self._update_qa_view(q, ans, score, ok, hit)

        elif op == "knn_query":
            if c["dataset"] is None:
                self._log("⚠ " + T("请先载入数据并训练"))
            else:
                px, py = float(colval(0)), float(colval(1))
                winner, neigh, dists = c["knn"].query(px, py)
                name = (c["dataset"].class_names[winner]
                        if winner < len(c["dataset"].class_names) else str(winner))
                self._log(T("查询") + f"({px},{py}) → 「{name}」, {len(neigh)} " + T("个邻居") + f" {[round(d,2) for d in dists]}")
                self._update_knn_view(px, py, neigh)

        elif op == "print":
            self._log(str(colval(0)))

        elif op == "show_metric":
            self._log(f"{colval(0)} = {colval(1)}")

        elif op == "save_model":
            self._op_save_model(c, colval(0), colval(1))

        elif op == "load_model":
            self._op_load_model(c, colval(0))

        elif op == "chat":
            yield from self._op_chat(c)

        elif op in ("au_folder", "im_folder"):
            c["recog_folder"] = colval(0)

        elif op in ("au_hidden", "im_hidden"):
            c["recog_hidden"] = int(float(colval(0)))

        elif op in ("au_epochs", "im_epochs"):
            c["recog_epochs"] = int(float(colval(0)))

        elif op == "recog_train":
            yield from self._op_recog_train(c)

        elif op == "recog_predict":
            self._op_recog_predict(c, colval(0))

        elif op == "recog_accuracy":
            rec = c.get("recog")
            if rec is None:
                self._log("⚠ " + T("还没训练，先跑一下训练积木"))
            else:
                self._log(T("识别准确率") + f": {rec.accuracy():.1%} "
                          + f"({len(rec.test)} " + T("个测试样本") + ")")

        elif op == "im_draw":
            self._op_image_draw(c)

        elif op == "wait":
            secs = float(colval(0))
            frames = max(1, int(secs / 0.03))
            for _ in range(frames):
                yield

        elif op == "repeat":
            n = int(float(colval(0)))
            body = b.children[0] if b.children else None
            for _ in range(n):
                if body is not None:
                    yield from self._run_chain(body)
                yield
            return

        elif op == "stop":
            raise StopIteration

        yield

    def _drive_training(self, m, kind):
        steps = 0
        while True:
            s = m.step()
            steps += 1
            if s is None:
                break
            if steps % 3 == 0:
                self._refresh_dash()
                yield
        self.ctx["trained"] = True
        self._log(T("训练完成：") + f"{m.epoch} " + T("轮") + f"  loss={m.loss:.4f}  "
                  + T("准确率=") + f"{m.acc:.1%}  " + T("测试集=") + f"{m.test_accuracy():.1%}")
        self._refresh_dash()

    def _update_net_view(self):
        v = self.dash.get("net")
        if not v:
            return
        c = self.ctx
        layers = [len(c["dataset"].X[0]) if c["dataset"] else 2]
        layers += [n for n, _ in c["layers"]]
        layers += [max(2, c["dataset"].n_classes if c["dataset"] else 2)]
        v.draw(layers)

    def _refresh_dash(self):
        c = self.ctx
        v = self.dash
        if self.board == "nn" and "loss" in v:
            m = c.get("model")
            if m:
                v["loss"].draw([(m.loss_history or [1.0], ACCENT, "loss")])
                v["acc"].draw([(m.acc_history or [0.0], ACCENT_2, "acc")])
                bounds = c["dataset"].bounds if c["dataset"] else None
                grid = m.decision_grid(bounds, 26) if bounds else None
                v["scatter"].draw(c["dataset"], grid, bounds)
                self._update_net_view()
        elif self.board == "mm" and "heat" in v:
            mm = c["mm"]
            if mm.trained:
                labels, mat = mm.matrix()
                v["heat"].draw(labels, mat)
        elif self.board == "qa" and "bar" in v:
            pass
        elif self.board == "classic" and "scatter" in v:
            ds = c["dataset"]
            if ds:
                v["scatter"].draw(ds, None, ds.bounds)

    def _update_km_view(self):
        v = self.dash.get("scatter")
        c = self.ctx
        if not v or not c["dataset"]:
            return
        grid = c["km"].decision_grid(c["dataset"].bounds, 24)
        v.draw(c["dataset"], grid, c["dataset"].bounds, centroids=c["km"].centroids)
        cur = self.dash.get("curve")
        if cur:
            cur.draw([(c["km"].history or [1], "#F25C54", "inertia")])

    def _update_ga_view(self, done=False):
        v = self.dash.get("scatter")
        c = self.ctx
        ga = c["ga"]
        if not v:
            return
        lo, hi = ga.domain
        f = ga.fn
        pts = []
        N = 160
        for i in range(N):
            x = lo + (hi - lo) * i / (N - 1)
            pts.append((x, f(x)))
        fmax = max(p[1] for p in pts)
        fmin = min(p[1] for p in pts)
        class FakeDS:
            X = []
            y = []
            bounds = (lo, hi, fmin - 0.1, fmax + 0.1)
        v.draw(FakeDS(), None, FakeDS.bounds, curves=[(pts, "#4C97FF")])
        v.draw(FakeDS(), None, FakeDS.bounds, curves=[(pts, "#4C97FF")])
        cur = self.dash.get("curve")
        if cur:
            cur.draw([(ga.history or [0], ACCENT_2, T("最优适应度"))])

    def _update_knn_view(self, px, py, neigh):
        v = self.dash.get("scatter")
        c = self.ctx
        if not v or not c["dataset"]:
            return
        bounds = c["dataset"].bounds
        bx0, bx1, by0, by1 = bounds
        W = DASH_W - 28
        sx = self._dm(px, bx0, bx1, 8, W - 8)
        sy = self._dm(py, by0, by1, 370, 30)
        grid = c["knn"].decision_grid(bounds, 22)
        hl = [(sx, sy, "query")]
        for i in neigh:
            p = c["dataset"].X[i]
            hx = self._dm(p[0], bx0, bx1, 8, W - 8)
            hy = self._dm(p[1], by0, by1, 370, 30)
            hl.append((hx, hy, "neighbor"))
        v.draw(c["dataset"], grid, bounds, highlight=hl)

    def _dm(self, v, a, b, c, d):
        if b - a == 0:
            return (c + d) / 2
        return c + (v - a) / (b - a) * (d - c)

    def _update_predict_marker(self, xy, cls):
        v = self.dash.get("scatter")
        if not v or not self.ctx["dataset"]:
            return
        ds = self.ctx["dataset"]
        bounds = ds.bounds
        bx0, bx1, by0, by1 = bounds
        W = DASH_W - 28
        sx = self._dm(xy[0], bx0, bx1, 8, W - 8)
        sy = self._dm(xy[1], by0, by1, 722, 592)
        m = self.ctx.get("model")
        grid = m.decision_grid(bounds, 26) if m else None
        v.draw(ds, grid, bounds, highlight=[(sx, sy, "query")])

    def _update_qa_view(self, q, ans, score, ok, hit):
        h = self.ctx.setdefault("qa_history", [])
        h.insert(0, (q, ans, score, ok, hit))
        del h[6:]
        v = self.dash.get("qa")
        if not v:
            return
        try:
            v.draw(h)
        except Exception:
            pass

    def _log(self, text):
        """往右侧输出区写一行，同时更新底部状态栏。"""
        self.ctx["log"].append(text)
        try:
            self.status.configure(text=str(text)[:70], fg=DASH_MUTED)
        except Exception:
            pass
        self._append_log(text)

    def _append_log(self, text):
        t = getattr(self, "log_text", None)
        if t is None:
            return
        s = str(text)
        if s.startswith("✓"):
            tag = "ok"
        elif s.startswith("⚠"):
            tag = "warn"
        elif s.startswith(T("问：")) or s.startswith(T("你：")):
            tag = "ask"
        elif s.startswith(T("答：")) or s.startswith(T("它：")):
            tag = "ans"
        else:
            tag = ""
        try:
            t.configure(state="normal")
            t.insert("end", s + "\n", tag)
            lines = int(t.index("end-1c").split(".")[0])
            if lines > 400:
                t.delete("1.0", "100.0")
            t.see("end")
            t.configure(state="disabled")
        except tk.TclError:
            pass

    def _clear_log(self):
        t = getattr(self, "log_text", None)
        if t is None:
            return
        try:
            t.configure(state="normal")
            t.delete("1.0", "end")
            t.configure(state="disabled")
        except tk.TclError:
            pass

    def _current_model(self, c, name):
        """按名字取当前板块要导出的模型对象。"""
        name = str(name).strip().lower()
        if name in ("", "model", "模型", "nn", "mlp"):
            if c.get("model") is not None:
                return c["model"]
        if name in ("qa", "qabot") or (name in ("", "model", "模型") and self.board == "qa"):
            return c["qa"]
        if name in ("mm", "markov") or (name in ("", "model", "模型") and self.board == "mm"):
            return c["mm"]
        if name in ("knn",) or (name in ("", "model", "模型") and self.board == "classic"):
            if c["knn"].X:
                return c["knn"]
        if name in ("kmeans", "km") and c["km"].centroids:
            return c["km"]
        if name in ("ga",) and c["ga"].gen > 0:
            return c["ga"]
        if name in ("recog", "识别", "识别器") or self.board in ("audio", "image"):
            if c.get("recog") is not None:
                return c["recog"]
        if self.board == "classic":
            for k in ("knn", "km", "ga"):
                if k == "knn" and c["knn"].X:
                    return c["knn"]
                if k == "km" and c["km"].centroids:
                    return c["km"]
                if k == "ga" and c["ga"].gen > 0:
                    return c["ga"]
        return None

    def _op_save_model(self, c, which, path):
        """导出模型到 JSON。path 为空时弹保存框。"""
        obj = self._current_model(c, which)
        if obj is None:
            self._log("⚠ " + T("还没有可导出的模型，先训练一下"))
            return
        if not path:
            path = filedialog.asksaveasfilename(
                title=T("导出模型"),
                defaultextension=".json",
                filetypes=[(T("模型文件"), "*.json"), (T("所有文件"), "*.*")],
                initialfile=f"{self.board}-model.json")
        if not path:
            return
        try:
            _ok, msg = A.export_model(obj, path, {"board": self.board,
                                                  "lang": i18n.get_lang()})
            self._log("✓ " + msg)
            self._write_infer_script(path, obj)
        except Exception as ex:
            self._log("⚠ " + T("导出失败") + f": {ex}")

    def _write_infer_script(self, model_path, obj):
        """在模型文件旁生成一个零依赖的推理脚本。"""
        try:
            base = os.path.splitext(model_path)[0]
            out = base + "_infer.py"
            with open(out, "w", encoding="utf-8") as f:
                f.write(INFER_SCRIPT)
            self._log(T("已附带推理脚本") + f": {os.path.basename(out)}")
        except Exception:
            pass

    def _op_load_model(self, c, path):
        """从 JSON 读回模型。"""
        if not path:
            path = filedialog.askopenfilename(
                title=T("载入模型"),
                filetypes=[(T("模型文件"), "*.json"), (T("所有文件"), "*.*")])
        if not path:
            return
        try:
            obj, board_hint, raw = A.import_model(path)
        except Exception as ex:
            self._log("⚠ " + T("载入失败") + f": {ex}")
            return
        t = raw.get("type")
        c["model"] = obj if t == "mlp" else c.get("model")
        if t == "markov":
            c["mm"] = obj
        elif t == "qabot":
            c["qa"] = obj
        elif t == "knn":
            c["knn"] = obj
        elif t == "kmeans":
            c["km"] = obj
        elif t == "ga":
            c["ga"] = obj
        elif t == "recog":
            c["recog"] = obj
            c["recog_kind"] = obj.kind
            c["recog_classes"] = list(obj.class_names)
            c["recog_folder"] = obj.folder
            self._refresh_recog_view(obj)
        meta = raw.get("meta", {})
        self._log("✓ " + T("已载入模型") + f" [{t}] "
                  + (T("准确率") + f" {meta['acc']:.1%}" if "acc" in meta else ""))
        if t == "mlp":
            c["trained"] = True
            self._update_net_view()
        self._refresh_dash()

    def _op_chat(self, c):
        """进入对话模式：底部出现输入框，可以一直聊。"""
        bot = self._make_chat_fn(c)
        if bot is None:
            self._log("⚠ " + T("还没有可对话的模型，先训练或载入一个"))
            return
        self._log(T("对话模式已开启 —— 在底部输入框里说话"))
        if not self._open_chat_bar(bot):
            return
        while not self._chat_closed:
            yield

    def _make_chat_fn(self, c):
        """返回一个 question -> (answer, detail) 的函数，取决于当前板块。"""
        if self.board == "nn" and c.get("model") is not None:
            m = c["model"]
            names = c["dataset"].class_names if c["dataset"] else []

            def f(q):
                try:
                    xy = [float(t) for t in str(q).replace(",", " ").split()][:2]
                    if len(xy) < 2:
                        return T("请输入两个数，如 1.5 2.0"), ""
                    p = m.predict_proba(xy)
                    cls = max(range(len(p)), key=lambda i: p[i])
                    nm = names[cls] if cls < len(names) else str(cls)
                    return nm, T("置信度") + f" {p[cls]:.1%}"
                except Exception:
                    return T("请输入两个数，如 1.5 2.0"), ""
            return f
        if self.board == "mm" and c["mm"].probs:
            mk = c["mm"]

            def f(q):
                txt, _ = mk.generate(str(q), 60, 1.0)
                return txt, ""
            return f
        if self.board == "qa" and c["qa"].rules:
            bot = c["qa"]

            def f(q):
                ans, hit, score, ok = bot.answer(str(q))
                return ans, (T("命中") if ok else T("未命中")) + f" {score:.2f} {hit}"
            return f
        if self.board == "classic" and c["knn"].X:
            knn = c["knn"]
            names = c["dataset"].class_names if c["dataset"] else []

            def f(q):
                try:
                    xy = [float(t) for t in str(q).replace(",", " ").split()][:2]
                    if len(xy) < 2:
                        return T("请输入两个数，如 1.5 2.0"), ""
                    w, neigh, _ = knn.query(xy[0], xy[1])
                    nm = names[w] if w < len(names) else str(w)
                    return nm, T("邻居") + f" {len(neigh)}"
                except Exception:
                    return T("请输入两个数，如 1.5 2.0"), ""
            return f
        if self.board in ("audio", "image") and c.get("recog") is not None:
            rec = c["recog"]

            def f(q):
                p = str(q).strip().strip('"')
                if not p:
                    return T("把要识别的文件路径贴进来"), ""
                if not os.path.exists(p):
                    return T("找不到这个文件") + f": {p}", ""
                try:
                    name, conf, _ = rec.predict_path(p)
                    return "「" + name + "」", T("置信度") + f" {conf:.1%}"
                except Exception as ex:
                    return T("识别失败") + f": {ex}", ""
            return f
        return None

    def _open_chat_bar(self, bot):
        """在窗口底部插入一个对话条。返回是否成功。"""
        try:
            bar = tk.Frame(self.root, bg="#0F172A")
            bar.pack(side="bottom", fill="x")
            tk.Label(bar, text=T("你说："), bg="#0F172A", fg=DASH_MUTED,
                     font=("Microsoft YaHei UI", 10)).pack(side="left", padx=(12, 4), pady=8)
            ent = tk.Entry(bar, font=("Microsoft YaHei UI", 11), relief="flat",
                           bg="#1B2A4A", fg=DASH_TEXT, insertbackground=DASH_TEXT)
            ent.pack(side="left", fill="x", expand=True, padx=4, pady=8)
            ent.focus_set()
            close = tk.Label(bar, text=T("结束"), bg="#E24B4A", fg="#FFFFFF",
                             font=("Microsoft YaHei UI", 9, "bold"),
                             padx=12, pady=4, cursor="hand2")
            close.pack(side="right", padx=10)

            self._chat_closed = False
            self._chat_bar = bar

            def send(e=None):
                q = ent.get().strip()
                if not q:
                    return
                ans, detail = bot(q)
                self._log(T("你：") + f" {q}")
                self._log(T("它：") + f" {ans}  {detail}")
                ent.delete(0, "end")

            def stop(e=None):
                self._chat_closed = True

            ent.bind("<Return>", send)
            close.bind("<Button-1>", stop)
            return True
        except tk.TclError:
            return False

    def _op_recog_train(self, c):
        """用文件夹里的样本训练识别器（分帧跑，界面能看到进度）。"""
        kind = "audio" if self.board == "audio" else "image"
        folder = c.get("recog_folder", "")
        if not folder:
            folder = filedialog.askdirectory(title=T("选择样本文件夹"))
            if not folder:
                self._log("⚠ " + T("没有选文件夹"))
                return
        rec = A.RecogModel()
        rec.kind = kind
        hidden = c.get("recog_hidden")
        epochs = c.get("recog_epochs")

        try:
            names, items, _ = rec.scan(folder)
        except Exception as ex:
            self._log("⚠ " + T("读文件夹失败") + f": {ex}")
            return
        if len(names) < 2:
            self._log("⚠ " + T("至少要有两个类别文件夹，每个装一类的样本"))
            return
        self._log(T("发现") + f" {len(names)} " + T("个类别") + f" {names}, "
                  + f"{len(items)} " + T("个样本，正在提取特征…"))
        X, y, ok_names = [], [], []
        for path, lab in items:
            try:
                X.append(rec.feats(path))
            except Exception:
                continue
            y.append(lab)
        if len(X) < 2:
            self._log("⚠ " + T("能读出来的样本太少，检查一下文件格式"))
            return
        rec.class_names = names
        rec.n_features = len(X[0])

        self._log(T("开始训练识别器") + f"（{len(X)} " + T("个样本") +
                  f", {len(names)} " + T("类") + "）…")
        try:
            rec.prepare(X, y, names, folder, hidden, epochs)
        except Exception as ex:
            self._log("⚠ " + T("训练失败") + f": {ex}")
            return
        c["recog"] = rec
        c["recog_kind"] = kind
        c["recog_classes"] = names

        steps = 0
        while True:
            s = rec.mlp.step()
            if s is None:
                break
            steps += 1
            if steps % 4 == 0:
                self._refresh_recog_view(rec, self._recog_partial_probs(rec))
                yield
        rec.finish()
        try:
            self._refresh_recog_view(rec)
        except Exception:
            pass
        self._log("✓ " + T("训练完成") + f"  " + T("训练集") + f" {rec.acc:.1%}  "
                  + T("测试集") + f" {rec.test_acc:.1%}")

    def _recog_partial_probs(self, rec):
        """训练途中给个粗略进度条（还没考完，就用当前样本的平均预测）。"""
        try:
            if not rec.samples:
                return None
            n = min(6, len(rec.samples))
            acc = [0.0] * len(rec.class_names)
            for i in range(n):
                x, lab = rec.samples[i]
                try:
                    p = rec.mlp.predict_proba(x)
                except Exception:
                    break
                for j in range(min(len(acc), len(p))):
                    acc[j] += p[j]
            return [v / n for v in acc]
        except Exception:
            return None

    def _op_recog_predict(self, c, path):
        rec = c.get("recog")
        if rec is None:
            self._log("⚠ " + T("还没训练，先跑一下训练积木"))
            return
        if not path:
            ft = ([(T("音频文件"), "*.wav")] if self.board == "audio"
                  else [(T("图片文件"), "*.png *.gif *.bmp *.ppm")])
            path = filedialog.askopenfilename(title=T("选择要识别的文件"),
                                              filetypes=ft + [(T("所有文件"), "*.*")])
        if not path:
            return
        try:
            name, conf, probs = rec.predict_path(path)
        except Exception as ex:
            self._log("⚠ " + T("识别失败") + f": {ex}")
            return
        self._log(T("识别结果") + f": 「{name}」  " + T("置信度") + f" {conf:.1%}  "
                  + f"({os.path.basename(path)})")
        self._refresh_recog_view(rec, probs, name)

    def _op_image_draw(self, c):
        """打开一个画板，鼠标画图后直接识别。"""
        rec = c.get("recog")
        if rec is None:
            self._log("⚠ " + T("还没训练，先跑一下训练积木"))
            return
        try:
            dlg = tk.Toplevel(self.root)
        except tk.TclError:
            return
        dlg.title(T("画板 —— 画个图形让它认"))
        dlg.configure(bg="#0F172A")
        dlg.geometry("360x460")
        GRID = 8
        CELL = 28
        cv = tk.Canvas(dlg, width=GRID * CELL, height=GRID * CELL,
                       bg="#111A2E", highlightthickness=0)
        cv.pack(padx=16, pady=(16, 8))
        cells = [[0] * GRID for _ in range(GRID)]
        rects = {}
        for gy in range(GRID):
            for gx in range(GRID):
                rid = cv.create_rectangle(gx * CELL, gy * CELL,
                                          (gx + 1) * CELL, (gy + 1) * CELL,
                                          fill="#1B2A4A", outline="#243656")
                rects[(gx, gy)] = rid

        def paint(ev):
            gx, gy = ev.x // CELL, ev.y // CELL
            if 0 <= gx < GRID and 0 <= gy < GRID:
                cells[gy][gx] = 1
                cv.itemconfigure(rects[(gx, gy)], fill="#E8EEFF")

        cv.bind("<B1-Motion>", paint)
        cv.bind("<Button-1>", paint)

        res = tk.Label(dlg, text=T("画点什么，然后点识别"), bg="#0F172A",
                       fg=DASH_MUTED, font=("Microsoft YaHei UI", 10))
        res.pack(pady=(0, 6))

        def clear():
            for gy in range(GRID):
                for gx in range(GRID):
                    cells[gy][gx] = 0
                    cv.itemconfigure(rects[(gx, gy)], fill="#1B2A4A")
            res.configure(text=T("已清空"), fg=DASH_MUTED)

        def do_predict():
            flat = [cells[gy][gx] for gy in range(GRID) for gx in range(GRID)]
            try:
                name, conf, probs = rec.predict_vec(flat)
            except Exception as ex:
                res.configure(text=T("识别失败") + f": {ex}", fg="#E24B4A")
                self._log("⚠ " + T("识别失败") + f": {ex}")
                return
            res.configure(text=T("我猜是") + f"「{name}」  {conf:.1%}", fg="#22C55E")
            self._log(T("画板识别") + f": 「{name}」  " + T("置信度") + f" {conf:.1%}")
            self._refresh_recog_view(rec, probs, name)

        btns = tk.Frame(dlg, bg="#0F172A")
        btns.pack(pady=4)
        tk.Button(btns, text=T("识别"), command=do_predict, bg=ACCENT, fg="#FFFFFF",
                  relief="flat", font=("Microsoft YaHei UI", 10, "bold"),
                  padx=18, pady=6, cursor="hand2").pack(side="left", padx=6)
        tk.Button(btns, text=T("清空"), command=clear, bg="#2A3A5A", fg=DASH_TEXT,
                  relief="flat", font=("Microsoft YaHei UI", 10),
                  padx=18, pady=6, cursor="hand2").pack(side="left", padx=6)
        try:
            self.root.wait_window(dlg)
        except tk.TclError:
            pass

    def _refresh_recog_view(self, rec, probs=None, best=None):
        """在右侧面板列出各类别概率条形。"""
        v = self.dash.get("recog")
        if v is None:
            return
        if probs is None:
            probs = [1.0 if i == 0 else 0.0 for i in range(len(rec.class_names))]
        rows = [(rec.class_names[i], probs[i])
                for i in range(min(len(rec.class_names), len(probs)))]
        rows.sort(key=lambda kv: -kv[1])
        try:
            v.draw(rows, best)
        except Exception:
            pass

    def on_stop(self):
        self.running = False
        self.status.configure(text=T("已停止"), fg=DASH_MUTED)

    def on_clear(self):
        for b in list(self.workspace_blocks):
            for bb in self._iter_all(b):
                bb.clear()
        self.workspace_blocks = []
        self._reset_ctx()
        self._clear_dash()
        self._clear_log()
        self.status.configure(text=T("已清空"), fg=DASH_MUTED)

    def _clear_dash(self):
        self.dash_canvas.delete("all")
        self.dash = self._make_dash()

    def on_demo(self):
        self.on_clear()
        bid = self.board
        if bid == "nn":
            seq = [("when_start", ()), ("load_dataset", ("",)),
                   ("add_layer", ("8", "relu")), ("add_layer", ("8", "tanh")),
                   ("set_lr", ("0.3",)), ("set_epochs", ("120",)),
                   ("init_model", ()), ("train_model", ()),
                   ("predict", ("1.5 2.0",))]
        elif bid == "mm":
            seq = [("when_start", ()), ("mm_load", ("",)), ("mm_order", ("2",)),
                   ("mm_unit", ("char",)), ("mm_train", ()),
                   ("mm_seed", ("从前",)), ("mm_generate", ("60", "1.0"))]
        elif bid == "qa":
            seq = [("when_start", ()), ("qa_load", ("",)),
                   ("qa_ask", ("今天天气怎么样",)), ("qa_ask", ("你是谁",))]
        elif bid == "audio":
            seq = [("when_start", ()), ("au_folder", ("",)),
                   ("au_hidden", ("16",)), ("au_epochs", ("150",)),
                   ("recog_train", ()), ("recog_accuracy", ())]
        elif bid == "image":
            seq = [("when_start", ()), ("im_folder", ("",)),
                   ("im_hidden", ("24",)), ("im_epochs", ("200",)),
                   ("recog_train", ()), ("im_draw", ())]
        else:
            seq = [("when_start", ()), ("knn_load", ("",)), ("knn_k", ("5",)),
                   ("knn_train", ()), ("knn_query", ("0", "0"))]
        prev = None
        y = 60
        for kind, vals in seq:
            b = self._spawn_block(kind, 80, y, values=list(vals) if vals else None)
            if prev is None:
                pass
            else:
                prev.next_block = b
                b.parent = prev
            prev = b
            y += b.h
        top = self.workspace_blocks[0] if self.workspace_blocks else None
        if top:
            top.relayout_all()
        self._normalize_roots()
        self.status.configure(text=T("已载入示例，点「运行」开始"), fg="#22C55E")

    def on_save(self):
        path = filedialog.asksaveasfilename(defaultextension=".aiblocks",
                                            filetypes=[("AI Blocks", "*.aiblocks")])
        if not path:
            return
        import json
        data = {"board": self.board, "blocks": self._serialize_all()}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        self.status.configure(text=T("已保存") + f" {os.path.basename(path)}", fg="#22C55E")

    def _serialize_all(self):
        out = []
        for b in self.workspace_blocks:
            out.append(self._serialize_tree(b))
        return out

    def _serialize_workspace(self):
        """语言切换前保存工作区（与 _serialize_all 同义）。"""
        return self._serialize_all()

    def _restore_all(self, items):
        """把 _serialize_workspace 的结果恢复到当前工作区。"""
        for item in items or []:
            b = self._rebuild_tree(item, None, None)
            if b:
                self.workspace_blocks.append(b)

    def _serialize_tree(self, b):
        d = {"kind": b.kind, "x": b.x, "y": b.y, "values": [str(v) for v in b.values],
             "next": None, "children": []}
        if b.next_block:
            d["next"] = self._serialize_tree(b.next_block)
        for c in b.children:
            d["children"].append(self._serialize_tree(c))
        return d

    def on_open(self):
        path = filedialog.askopenfilename(filetypes=[("AI Blocks", "*.aiblocks")])
        if not path:
            return
        import json
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.on_clear()
        for item in data.get("blocks", []):
            b = self._rebuild_tree(item, None, None)
            if b:
                self.workspace_blocks.append(b)
        self.status.configure(text=T("已打开") + f" {os.path.basename(path)}", fg="#22C55E")

    def _rebuild_tree(self, d, parent, owner):
        kind = d.get("kind")
        active = bv._ACTIVE_BLOCKS or bv.BLOCKS
        if kind not in active:
            nxt = d.get("next")
            if not nxt:
                return None
            return self._rebuild_tree(nxt, parent, owner)
        b = BlockView(self, kind, d["x"], d["y"], values=d["values"])
        b.parent = parent
        b.slot_owner = owner
        if parent is not None:
            parent.next_block = b
        for cd in d.get("children", []):
            c = self._rebuild_tree(cd, None, b)
            if c:
                b.children.append(c)
                c.slot_owner = b
        if d.get("next"):
            self._rebuild_tree(d["next"], b, None)
        return b

    def back_to_menu(self):
        self.on_stop()
        if self._training_job:
            try:
                self.root.after_cancel(self._training_job)
            except Exception:
                pass
        self.root.withdraw()
        try:
            for w in list(self.root.winfo_children()):
                w.destroy()
            self._tick_started = True
            self.board = None
            self.workspace_blocks = []
            self.palette_views = []
            self.drag = None
            self.running = False
            self.script_gen = None
            self.select_screen = DashSelect(self.root, self.enter_board,
                                            on_lang=self.on_lang_changed)
            self.select_screen.pack(fill="both", expand=True)
        finally:
            self.root.deiconify()
            self.root.update_idletasks()

    def _tick(self):
        self._tick_job = None
        if not self.root.winfo_exists():
            return
        try:
            status_alive = self.status is not None and self.status.winfo_exists()
        except tk.TclError:
            status_alive = False
        if self.running and self.script_gen is not None:
            try:
                next(self.script_gen)
            except StopIteration:
                self.running = False
                if status_alive:
                    self.status.configure(text=T("运行完成 ✓"), fg="#22C55E")
            except Exception as ex:
                self.running = False
                if status_alive:
                    self.status.configure(text=T("错误") + f": {ex}", fg="#E24B4A")
        try:
            self._tick_job = self.root.after(30, self._tick)
        except tk.TclError:
            self._tick_job = None

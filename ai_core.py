# -*- coding: utf-8 -*-
"""AI 内核 —— 纯 Python，零依赖，可单步执行以便界面做动画。

所有模型都实现统一的「分步」接口：
    step() -> dict   # 执行一小步，返回本步的指标快照；返回 None 表示训练结束
    reset()          # 回到初始状态
这样界面可以在 Tkinter 主循环里每 30ms 调一次 step()，实现「看着它学习」。
"""

import math
import random
import os
import csv

import i18n as _i18n

def _T(s):
    """翻译 helper（内核里少量用户可见文案）。"""
    return _i18n.t(s) if isinstance(s, str) else s

def sigmoid(x):
    if x < -60:
        return 0.0
    if x > 60:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))

def relu(x):
    return x if x > 0 else 0.0

def d_relu(x):
    return 1.0 if x > 0 else 0.0

def tanh(x):
    return math.tanh(x)

def randn():
    """近似标准正态分布（Box-Muller）。"""
    u1 = random.random() or 1e-9
    u2 = random.random()
    return math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)

class Dataset:
    """二维特征 + 多个类别的数据集。"""

    def __init__(self):
        self.X = []
        self.y = []
        self.class_names = []
        self.n_features = 2
        self.n_classes = 2
        self.source = ""

    def __len__(self):
        return len(self.X)

    @property
    def bounds(self):
        if not self.X:
            return (0, 1, 0, 1)
        xs = [p[0] for p in self.X]
        ys = [p[1] for p in self.X]
        pad = 0.1 * max(max(xs) - min(xs), max(ys) - min(ys), 1e-6)
        return (min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad)

    def split(self, train_ratio=0.8, seed=42):
        idx = list(range(len(self.X)))
        random.Random(seed).shuffle(idx)
        cut = int(len(idx) * train_ratio)
        tr, te = idx[:cut], idx[cut:]
        return (([self.X[i] for i in tr], [self.y[i] for i in tr]),
                ([self.X[i] for i in te], [self.y[i] for i in te]))

def make_blobs(n=240, centers=2, spread=0.9, seed=1):
    """生成高斯团簇数据集（默认 2 类，方便 2D 决策边界可视化）。"""
    rng = random.Random(seed)
    ds = Dataset()
    ds.n_classes = centers
    ds.class_names = [chr(ord("A") + i) for i in range(centers)]
    cs = []
    for i in range(centers):
        a = 2 * math.pi * i / centers
        cs.append((math.cos(a) * 2.2, math.sin(a) * 2.2))
    for i in range(n):
        c = rng.randrange(centers)
        cx, cy = cs[c]
        ds.X.append([cx + rng.gauss(0, spread), cy + rng.gauss(0, spread)])
        ds.y.append(c)
    ds.source = _T("内置团簇") + f" ({centers} " + _T("类") + f" × {n//centers})"
    return ds

def make_xor(n=320, spread=0.35, seed=2):
    """XOR 螺旋状不可线性分割数据：用来证明「隐藏层」的必要性。"""
    rng = random.Random(seed)
    ds = Dataset()
    ds.n_classes = 2
    ds.class_names = ["0", "1"]
    for i in range(n):
        x = rng.uniform(-3, 3)
        y = rng.uniform(-3, 3)
        label = 1 if (x * y) > 0 else 0
        ds.X.append([x + rng.gauss(0, spread), y + rng.gauss(0, spread)])
        ds.y.append(label)
    ds.source = _T("内置 XOR（非线性可分）") + f" ({n})"
    return ds

def make_rings(n=300, seed=3):
    """同心圆环：经典的非线性分类测试。"""
    rng = random.Random(seed)
    ds = Dataset()
    ds.n_classes = 2
    ds.class_names = [_T("内环"), _T("外环")]
    for i in range(n):
        label = i % 2
        r = rng.uniform(0.6, 1.2) if label == 0 else rng.uniform(2.2, 3.0)
        a = rng.uniform(0, 2 * math.pi)
        ds.X.append([r * math.cos(a) + rng.gauss(0, .08),
                     r * math.sin(a) + rng.gauss(0, .08)])
        ds.y.append(label)
    ds.source = _T("内置同心圆") + f" ({n})"
    return ds

BUILTIN_DATASETS = {
    "blobs2": ("团簇·2类", lambda: make_blobs(240, 2)),
    "blobs3": ("团簇·3类", lambda: make_blobs(300, 3)),
    "xor": ("XOR·非线性", lambda: make_xor(320)),
    "rings": ("同心圆", lambda: make_rings(300)),
}

def builtin_dataset_label(key):
    """内置数据集名称（本地化）。"""
    return _T(BUILTIN_DATASETS[key][0]) if key in BUILTIN_DATASETS else key

def _split_line(line):
    """把一行拆成单元格。优先按逗号，其次制表符 / 分号 / 多个空格。
    这样 .csv、.tsv、.txt 甚至手打的「1.2 3.4 A」都能直接读。"""
    line = line.strip()
    if not line:
        return []
    if "," in line:
        return [c.strip() for c in line.split(",")]
    if "\t" in line:
        return [c.strip() for c in line.split("\t")]
    if ";" in line:
        return [c.strip() for c in line.split(";")]
    return line.split()

def load_csv_dataset(path):
    """从 CSV / TSV / TXT 载入二维数据。

    宽容规则（尽量不让人踩坑）：
      - 分隔符：逗号 / 制表符 / 分号 / 空格 都行
      - 表头：可有可无，自动识别
      - 标签：最后一列不是数字时当作标签；全部是数字时视为无标签
      - 无标签：全部标为同一类（适合 K-Means 聚类）
      - 列不足两列：用 0 补齐
    """
    ds = Dataset()
    rows = []
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        for line in f:
            cells = _split_line(line)
            if cells:
                rows.append(cells)
    if not rows:
        raise ValueError(_T("文件为空"))

    def is_number(s):
        try:
            float(s)
            return True
        except (TypeError, ValueError):
            return False

    first = rows[0]
    if len(first) >= 2 and not all(is_number(c) for c in first[:2]):
        data = rows[1:]
    elif len(first) == 1 and not is_number(first[0]):
        data = rows[1:]
    else:
        data = rows

    labels = {}
    for r in data:
        nums = []
        lab = None
        for c in r:
            if is_number(c):
                nums.append(float(c))
            else:
                lab = c.strip()
        if len(nums) < 2:
            if len(nums) == 1:
                nums.append(0.0)
            else:
                continue
        feats = nums[:2]
        if lab is None:
            lab = _T("未标注")
        if lab not in labels:
            labels[lab] = len(labels)
        ds.X.append(feats)
        ds.y.append(labels[lab])

    if not ds.X:
        raise ValueError(_T("没有读到有效数据行"))

    ds.n_classes = max(1, len(labels))
    ds.class_names = [k for k, _ in sorted(labels.items(), key=lambda kv: kv[1])]
    ds.n_features = len(ds.X[0])
    n_ul = sum(1 for k in labels if k == _T("未标注"))
    tag = ("" if len(labels) > 1 or not n_ul
           else ", " + _T("无标签"))
    ds.source = (os.path.basename(path) + f" ({len(ds.X)} " + _T("条样本")
                 + f", {ds.n_classes} " + _T("类") + tag + ")")
    return ds

AUDIO_FEAT_DIM = 24

def _read_wav_frames(path):
    """读 wav，返回 (采样率, 单声道 float 列表，范围 -1~1)。
    只依赖标准库 wave；多声道自动混成单声道。"""
    import wave
    import array
    with wave.open(path, "rb") as w:
        nch = w.getnchannels()
        sw = w.getsampwidth()
        sr = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)
    if sw == 1:
        a = array.array("B", raw)
        vals = [(v - 128) / 128.0 for v in a]
    elif sw == 2:
        a = array.array("h", raw)
        vals = [v / 32768.0 for v in a]
    elif sw == 3:
        vals = []
        for i in range(0, len(raw) - 2, 3):
            b0, b1, b2 = raw[i], raw[i + 1], raw[i + 2]
            v = b0 | (b1 << 8) | (b2 << 16)
            if v & 0x800000:
                v -= 0x1000000
            vals.append(v / 8388608.0)
    else:
        a = array.array("i", raw)
        vals = [v / 2147483648.0 for v in a]
    if nch > 1:
        vals = [sum(vals[i:i + nch]) / nch for i in range(0, len(vals), nch)]
    return sr, vals

def audio_features(path, n=AUDIO_FEAT_DIM):
    """把 wav 文件变成 n 个数字。每个数字是一小段音频的特征。

    每段算三个量：能量（多大声）、过零率（多刺耳）、频谱质心（音调高低），
    再把它们均匀铺成 n 个数字。这样「喂数据」就是拖一个 wav 文件夹进来。
    """
    sr, vals = _read_wav_frames(path)
    if not vals:
        raise ValueError(_T("音频是空的"))
    cap = sr * 2
    if len(vals) > cap:
        vals = vals[:cap]
    seg = max(1, len(vals) // (n // 3))
    feats = []
    for s in range(0, len(vals), seg):
        chunk = vals[s:s + seg]
        if not chunk:
            break
        rms = math.sqrt(sum(v * v for v in chunk) / len(chunk))
        zc = sum(1 for i in range(1, len(chunk))
                 if (chunk[i - 1] < 0) != (chunk[i] < 0)) / max(1, len(chunk))
        dif = [abs(chunk[i] - chunk[i - 1]) for i in range(1, len(chunk))]
        centroid = (sum(dif) / len(dif)) if dif else 0.0
        feats += [rms, zc, centroid]
    if len(feats) < n:
        feats += [0.0] * (n - len(feats))
    feats = feats[:n]
    mx = max(feats) or 1.0
    return [f / mx for f in feats]

IMAGE_FEAT_DIM = 64

def image_features(path, grid=8):
    """把图片读成 grid×grid 的灰度向量（0~1）。只用标准库。

    PNG / GIF / PPM / PGM / BMP 都能读 —— 纯手写解析，不依赖 Pillow。
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".ppm" or ext == ".pgm":
        w, h, px = _read_pnm(path)
    elif ext == ".bmp":
        w, h, px = _read_bmp(path)
    elif ext == ".png":
        w, h, px = _read_png(path)
    elif ext == ".gif":
        w, h, px = _read_gif(path)
    else:
        raise ValueError(_T("暂不支持这种图片格式，请用 png / gif / bmp / ppm"))
    out = []
    for gy in range(grid):
        for gx in range(grid):
            sx = min(w - 1, int((gx + 0.5) * w / grid))
            sy = min(h - 1, int((gy + 0.5) * h / grid))
            v = px[sy * w + sx]
            out.append(v / 255.0)
    return out

def _read_pnm(path):
    """读 P2/P3(ASCII) / P5/P6(二进制) 的 PGM/PPM。"""
    with open(path, "rb") as f:
        data = f.read()
    pos = 0
    tokens = []

    def next_tok():
        nonlocal pos
        while pos < len(data):
            c = data[pos:pos + 1]
            if c.isspace():
                pos += 1
            elif c == b"#":
                while pos < len(data) and data[pos:pos + 1] not in (b"\n", b"\r"):
                    pos += 1
            else:
                break
        start = pos
        while pos < len(data) and not data[pos:pos + 1].isspace():
            pos += 1
        return data[start:pos]

    magic = next_tok()
    w = int(next_tok())
    h = int(next_tok())
    mx = int(next_tok())
    pos += 1
    px = [0] * (w * h)
    if magic in (b"P5", b"P6"):
        nch = 3 if magic == b"P6" else 1
        step = 1 if mx < 256 else 2
        for i in range(w * h):
            if step == 1:
                ch = data[pos:pos + nch]
                pos += nch
                px[i] = sum(ch) // nch if nch == 3 else ch[0]
            else:
                ch = [data[pos + 2 * k] * 256 + data[pos + 2 * k + 1] for k in range(nch)]
                pos += 2 * nch
                px[i] = sum(ch) // nch if nch == 3 else ch[0] // 256
    else:
        nch = 3 if magic == b"P3" else 1
        vals = []
        while pos < len(data) and len(vals) < w * h * nch:
            t = next_tok()
            if not t:
                break
            try:
                vals.append(int(t))
            except ValueError:
                break
        for i in range(w * h):
            if nch == 3:
                px[i] = sum(vals[i * 3:i * 3 + 3]) // 3
            else:
                px[i] = vals[i] if i < len(vals) else 0
    return w, h, px

def _read_bmp(path):
    """读 24/32 位未压缩 BMP。"""
    with open(path, "rb") as f:
        d = f.read()
    if d[:2] != b"BM":
        raise ValueError(_T("不是有效的 BMP 文件"))
    off = int.from_bytes(d[10:14], "little")
    w = int.from_bytes(d[18:22], "little")
    h = int.from_bytes(d[22:26], "little", signed=True)
    bpp = int.from_bytes(d[28:30], "little")
    if bpp not in (24, 32):
        raise ValueError(_T("只支持 24/32 位 BMP"))
    step = bpp // 8
    row = ((w * step + 3) // 4) * 4
    flip = h > 0
    h = abs(h)
    px = [0] * (w * h)
    for y in range(h):
        sy = (h - 1 - y) if flip else y
        base = off + sy * row
        for x in range(w):
            i = base + x * step
            b, g, r = d[i], d[i + 1], d[i + 2]
            px[y * w + x] = (r * 299 + g * 587 + b * 114) // 1000
    return w, h, px

def _read_png(path):
    """手写 PNG 解码：支持 8 位灰度/RGB/RGBA/调色板（够画板用）。"""
    import zlib
    with open(path, "rb") as f:
        d = f.read()
    if d[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(_T("不是有效的 PNG 文件"))
    pos = 8
    w = h = bitd = ctype = 0
    idat = b""
    palette = None
    while pos < len(d):
        ln = int.from_bytes(d[pos:pos + 4], "big")
        typ = d[pos + 4:pos + 8]
        body = d[pos + 8:pos + 8 + ln]
        pos += 12 + ln
        if typ == b"IHDR":
            w = int.from_bytes(body[0:4], "big")
            h = int.from_bytes(body[4:8], "big")
            bitd = body[8]
            ctype = body[9]
        elif typ == b"PLTE":
            palette = [(body[i], body[i + 1], body[i + 2])
                       for i in range(0, len(body), 3)]
        elif typ == b"IDAT":
            idat += body
        elif typ == b"IEND":
            break
    if bitd != 8:
        raise ValueError(_T("只支持 8 位 PNG 图片"))
    raw = zlib.decompress(idat)
    nch = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[ctype]
    stride = w * nch
    out = bytearray(stride * h)
    prev = bytearray(stride)
    p = 0
    for y in range(h):
        ft = raw[p]
        p += 1
        line = bytearray(raw[p:p + stride])
        p += stride
        if ft == 1:
            for i in range(nch, stride):
                line[i] = (line[i] + line[i - nch]) & 255
        elif ft == 2:
            for i in range(stride):
                line[i] = (line[i] + prev[i]) & 255
        elif ft == 3:
            for i in range(stride):
                a = line[i - nch] if i >= nch else 0
                line[i] = (line[i] + ((a + prev[i]) >> 1)) & 255
        elif ft == 4:
            for i in range(stride):
                a = line[i - nch] if i >= nch else 0
                b = prev[i]
                c = prev[i - nch] if i >= nch else 0
                pp = a + b - c
                pa, pb, pc = abs(pp - a), abs(pp - b), abs(pp - c)
                pr = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                line[i] = (line[i] + pr) & 255
        out[y * stride:(y + 1) * stride] = line
        prev = line
    px = [0] * (w * h)
    for i in range(w * h):
        base = i * nch
        if ctype in (0, 4):
            v = out[base]
        elif ctype == 2:
            r, g, b = out[base], out[base + 1], out[base + 2]
            v = (r * 299 + g * 587 + b * 114) // 1000
        elif ctype == 3:
            idx = out[base]
            if palette and idx < len(palette):
                r, g, b = palette[idx]
                v = (r * 299 + g * 587 + b * 114) // 1000
            else:
                v = idx
        else:
            v = out[base]
        px[i] = v
    return w, h, px

def _read_gif(path):
    """手写 GIF 解码：读第一帧即可。"""
    with open(path, "rb") as f:
        d = f.read()
    if d[:6] not in (b"GIF87a", b"GIF89a"):
        raise ValueError(_T("不是有效的 GIF 文件"))
    w = int.from_bytes(d[6:8], "little")
    h = int.from_bytes(d[8:10], "little")
    flags = d[10]
    pos = 13
    gct = None
    if flags & 0x80:
        n = 2 ** ((flags & 0x07) + 1)
        gct = [(d[pos + i * 3], d[pos + i * 3 + 1], d[pos + i * 3 + 2])
               for i in range(n)]
        pos += n * 3
    while pos < len(d) and d[pos] != 0x2C:
        if d[pos] == 0x21:
            pos += 2
            while pos < len(d) and d[pos] != 0:
                pos += d[pos] + 1
            pos += 1
        else:
            pos += 1
    pos += 1
    iw = int.from_bytes(d[pos:pos + 2], "little")
    ih = int.from_bytes(d[pos + 2:pos + 4], "little")
    lflags = d[pos + 8]
    pos += 9
    lct = gct
    if lflags & 0x80:
        n = 2 ** ((lflags & 0x07) + 1)
        lct = [(d[pos + i * 3], d[pos + i * 3 + 1], d[pos + i * 3 + 2])
               for i in range(n)]
        pos += n * 3
    min_code = d[pos]
    pos += 1
    data = bytearray()
    while pos < len(d) and d[pos] != 0:
        ln = d[pos]
        data += d[pos + 1:pos + 1 + ln]
        pos += ln + 1
    pix = _lzw_decode(bytes(data), min_code, iw * ih)
    px = [0] * (w * h)
    for y in range(h):
        for x in range(w):
            if x < iw and y < ih:
                idx = pix[y * iw + x] if (y * iw + x) < len(pix) else 0
                if lct and idx < len(lct):
                    r, g, b = lct[idx]
                    px[y * w + x] = (r * 299 + g * 587 + b * 114) // 1000
    return w, h, px

def _lzw_decode(data, min_code, expected):
    """GIF 的 LZW 解压。"""
    clear = 1 << min_code
    eoi = clear + 1
    code_size = min_code + 1
    table = {i: [i] for i in range(clear)}
    nxt = eoi + 1
    out = []
    prev = None
    bitbuf = 0
    bitcnt = 0
    for byte in data:
        bitbuf |= byte << bitcnt
        bitcnt += 8
        while bitcnt >= code_size:
            code = bitbuf & ((1 << code_size) - 1)
            bitbuf >>= code_size
            bitcnt -= code_size
            if code == clear:
                table = {i: [i] for i in range(clear)}
                nxt = eoi + 1
                code_size = min_code + 1
                prev = None
                continue
            if code == eoi:
                return out

            if code in table:
                entry = table[code]
            elif prev is not None:
                entry = prev + [prev[0]]
            else:
                entry = [0]
            out += entry
            if prev is not None:
                if nxt < 4096:
                    table[nxt] = prev + [entry[0]]
                    nxt += 1
                    if nxt == (1 << code_size) and code_size < 12:
                        code_size += 1
            prev = entry
    return out

class MLP:
    """多层感知机，从零手写前向 + 反向传播。支持单步训练以便动画。"""

    def __init__(self):
        self.hidden = []
        self.lr = 0.1
        self.epochs = 200
        self.W = []
        self.b = []
        self.n_in = 2
        self.n_out = 2
        self.dataset = None
        self.train_X, self.train_y = [], []
        self.test_X, self.test_y = [], []
        self.epoch = 0
        self.loss = 0.0
        self.acc = 0.0
        self.loss_history = []
        self.acc_history = []
        self._order = []
        self._ptr = 0
        self._cur_loss = 0.0
        self._cur_correct = 0
        self._seen = 0
        self.batch_size = 8
        self.activation_last = []

    def add_layer(self, units, act="relu"):
        self.hidden.append((int(units), act))

    def configure(self, dataset, lr, epochs, train_ratio=0.8, batch_size=8):
        self.dataset = dataset
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        (self.train_X, self.train_y), (self.test_X, self.test_y) = \
            dataset.split(train_ratio)
        self.n_in = len(self.train_X[0]) if self.train_X else 2
        self.n_out = max(2, dataset.n_classes)
        self._build()

    def _build(self):
        self.W, self.b = [], []
        dims = [self.n_in] + [u for u, _ in self.hidden] + [self.n_out]
        for i in range(len(dims) - 1):
            fan_in = dims[i]
            std = math.sqrt(2.0 / fan_in)
            self.W.append([[randn() * std for _ in range(dims[i + 1])]
                           for _ in range(fan_in)])
            self.b.append([0.0] * dims[i + 1])
        self.reset()

    def reset(self):
        self.epoch = 0
        self.loss = 0.0
        self.acc = 0.0
        self.loss_history = []
        self.acc_history = []
        self._order = list(range(len(self.train_X)))
        random.shuffle(self._order)
        self._ptr = 0
        self._cur_loss = 0.0
        self._cur_correct = 0
        self._seen = 0

    def forward(self, x):
        """返回各层激活（含输入层），用于可视化。"""
        acts = [list(x)]
        a = x
        for li, (units, act) in enumerate(self.hidden):
            z = [sum(a[i] * self.W[li][i][j] for i in range(len(a))) + self.b[li][j]
                 for j in range(units)]
            a = [relu(v) if act == "relu" else tanh(v) if act == "tanh" else sigmoid(v)
                 for v in z]
            acts.append(a)
        li = len(self.hidden)
        z = [sum(a[i] * self.W[li][i][j] for i in range(len(a))) + self.b[li][j]
             for j in range(self.n_out)]
        m = max(z)
        e = [math.exp(v - m) for v in z]
        s = sum(e) or 1e-9
        acts.append([v / s for v in e])
        return acts

    def predict_proba(self, x):
        return self.forward(x)[-1]

    def predict(self, x):
        p = self.predict_proba(x)
        return max(range(len(p)), key=lambda i: p[i])

    def _backprop(self, x, y):
        acts = self.forward(x)
        n_layers = len(self.W)
        delta = list(acts[-1])
        delta[y] -= 1.0
        grads_W = [None] * n_layers
        grads_b = [None] * n_layers
        for li in range(n_layers - 1, -1, -1):
            a_prev = acts[li]
            gw = [[a_prev[i] * delta[j] for j in range(len(delta))]
                  for i in range(len(a_prev))]
            grads_W[li] = gw
            grads_b[li] = list(delta)
            if li > 0:
                wa = [sum(self.W[li][i][j] * delta[j] for j in range(len(delta)))
                      for i in range(len(a_prev))]
                units, act = self.hidden[li - 1]
                if act == "relu":
                    wa = [wa[i] * d_relu(acts[li][i]) for i in range(len(wa))]
                elif act == "tanh":
                    wa = [wa[i] * (1 - acts[li][i] ** 2) for i in range(len(wa))]
                elif act == "sigmoid":
                    wa = [wa[i] * acts[li][i] * (1 - acts[li][i]) for i in range(len(wa))]
                delta = wa
        return grads_W, grads_b

    def step(self):
        """执行一个小批次的训练，返回指标快照。训练结束返回 None。"""
        if self.epoch >= self.epochs:
            return None
        if self._ptr >= len(self._order):
            self._finish_epoch()
            if self.epoch >= self.epochs:
                return None
            self._order = list(range(len(self.train_X)))
            random.shuffle(self._order)
            self._ptr = 0
        n = min(self.batch_size, len(self._order) - self._ptr)
        accW, accb = None, None
        batch_loss = 0.0
        for _ in range(n):
            i = self._order[self._ptr]
            self._ptr += 1
            x, y = self.train_X[i], self.train_y[i]
            gw, gb = self._backprop(x, y)
            p = self.predict_proba(x)
            batch_loss += -math.log(max(p[y], 1e-12))
            self._cur_correct += (max(range(len(p)), key=lambda k: p[k]) == y)
            self._seen += 1
            if accW is None:
                accW = [[[v for v in col] for col in layer] for layer in gw]
                accb = [[v for v in layer] for layer in gb]
            else:
                for li in range(len(gw)):
                    for a in range(len(gw[li])):
                        for b2 in range(len(gw[li][a])):
                            accW[li][a][b2] += gw[li][a][b2]
                    for a in range(len(accb[li])):
                        accb[li][a] += gb[li][a]
        for li in range(len(self.W)):
            for a in range(len(self.W[li])):
                for b2 in range(len(self.W[li][a])):
                    self.W[li][a][b2] -= self.lr * accW[li][a][b2] / n
            for a in range(len(self.b[li])):
                self.b[li][a] -= self.lr * accb[li][a] / n
        self._cur_loss += batch_loss
        return self._snapshot(partial=True)

    def _finish_epoch(self):
        self.epoch += 1
        self.loss = self._cur_loss / max(1, self._seen)
        self.acc = self._cur_correct / max(1, self._seen)
        self.loss_history.append(self.loss)
        self.acc_history.append(self.acc)
        self._cur_loss = 0.0
        self._cur_correct = 0
        self._seen = 0

    def _snapshot(self, partial=False):
        return {
            "epoch": self.epoch,
            "loss": self.loss,
            "acc": self.acc,
            "progress": self.epoch / max(1, self.epochs),
        }

    def test_accuracy(self):
        if not self.test_X:
            return 0.0
        ok = sum(1 for x, y in zip(self.test_X, self.test_y)
                 if self.predict(x) == y)
        return ok / len(self.test_X)

    def decision_grid(self, bounds, res=28):
        """在给定范围内采样预测，用于画决策边界。返回 (res,res) 的类别网格。"""
        x0, x1, y0, y1 = bounds
        grid = []
        for gy in range(res):
            row = []
            yy = y0 + (y1 - y0) * gy / (res - 1)
            for gx in range(res):
                xx = x0 + (x1 - x0) * gx / (res - 1)
                row.append(self.predict([xx, yy]))
            grid.append(row)
        return grid

    def to_dict(self):
        return {
            "type": "mlp",
            "n_in": self.n_in,
            "n_out": self.n_out,
            "hidden": [[int(u), a] for u, a in self.hidden],
            "weights": self.W,
            "biases": self.b,
            "meta": {
                "lr": self.lr,
                "epochs": self.epochs,
                "trained_epochs": self.epoch,
                "loss": round(self.loss, 6),
                "acc": round(self.acc, 6),
                "test_acc": round(self.test_accuracy(), 6),
                "n_train": len(self.train_X),
                "n_test": len(self.test_X),
            },
        }

    @staticmethod
    def from_dict(d):
        m = MLP()
        m.n_in = int(d["n_in"])
        m.n_out = int(d["n_out"])
        m.hidden = [(int(u), a) for u, a in d["hidden"]]
        m.W = [[[float(v) for v in col] for col in layer] for layer in d["weights"]]
        m.b = [[float(v) for v in layer] for layer in d["biases"]]
        meta = d.get("meta", {})
        m.lr = float(meta.get("lr", 0.1))
        m.epochs = int(meta.get("epochs", 0))
        m.epoch = int(meta.get("trained_epochs", 0))
        m.loss = float(meta.get("loss", 0.0))
        m.acc = float(meta.get("acc", 0.0))
        return m

class Markov:
    """n 阶马尔可夫链：统计转移概率 + 温度采样生成。"""

    def __init__(self):
        self.order = 2
        self.unit = "char"
        self.tokens = []
        self.trans = {}
        self.probs = {}
        self.vocab = []
        self.source = ""
        self.trained = False

    def load_text(self, text):
        if self.unit == "char":
            self.tokens = [c for c in text if c != "\r"]
        else:
            import re
            self.tokens = re.findall(r"[\u4e00-\u9fff]|[A-Za-z]+|\d+|\S", text)
        self.source = f"{len(self.tokens)} " + _T("个单元")

    def train(self):
        self.trans = {}
        n = self.order
        L = self.tokens
        for i in range(len(L) - n):
            key = tuple(L[i:i + n])
            nxt = L[i + n]
            self.trans.setdefault(key, {})
            self.trans[key][nxt] = self.trans[key].get(nxt, 0) + 1
        self.probs = {}
        for key, cnt in self.trans.items():
            tot = sum(cnt.values())
            self.probs[key] = sorted(((t, c / tot) for t, c in cnt.items()),
                                     key=lambda kv: -kv[1])
        self.vocab = sorted(set(self.tokens))
        self.trained = True
        return len(self.probs)

    def generate(self, seed_text, length=60, temperature=1.0):
        """按概率采样生成。temperature 越高越随机。"""
        if not self.probs:
            return "", []
        n = self.order
        if self.unit == "char":
            seed = [c for c in seed_text if c != "\r"]
        else:
            import re
            seed = re.findall(r"[\u4e00-\u9fff]|[A-Za-z]+|\d+|\S", seed_text)
        if len(seed) < n:
            if self.probs:
                seed = list(next(iter(self.probs)))
            else:
                return "", []
        out = seed[:]
        trace = []
        for _ in range(length):
            key = tuple(out[-n:])
            options = self.probs.get(key)
            if not options:
                key = random.choice(list(self.probs))
                out = list(key)
                options = self.probs[key]
                trace.append(("⟲重启", 1.0))
            toks = [t for t, _ in options]
            ws = [(p ** (1.0 / max(1e-3, temperature))) for _, p in options]
            s = sum(ws) or 1.0
            ws = [w / s for w in ws]
            r = random.random()
            acc = 0.0
            picked = toks[-1]
            for t, w in zip(toks, ws):
                acc += w
                if r <= acc:
                    picked = t
                    break
            trace.append((picked, dict(options).get(picked, 0)))
            out.append(picked)
        joiner = "" if self.unit == "char" else " "
        return joiner.join(out), trace

    def matrix(self, limit=14):
        """返回 (labels, 概率矩阵) 供热力图绘制。"""
        keys = list(self.probs.keys())[:limit]
        labels = ["".join(k) for k in keys]
        rows = []
        for k in keys:
            row = []
            for k2 in keys:
                row.append(self.trans.get(k, {}).get(k2[-1], 0))
            rows.append(row)
        norm = []
        for row in rows:
            tot = sum(row) or 1
            norm.append([v / tot for v in row])
        return labels, norm

    def to_dict(self):
        return {
            "type": "markov",
            "order": self.order,
            "unit": self.unit,
            "vocab": self.vocab,
            "transitions": {" ".join(k): v for k, v in self.trans.items()},
            "meta": {
                "n_tokens": len(self.tokens),
                "n_states": len(self.probs),
                "trained": self.trained,
                "source": self.source,
            },
        }

    @staticmethod
    def from_dict(d):
        m = Markov()
        m.order = int(d["order"])
        m.unit = d.get("unit", "char")
        m.vocab = list(d.get("vocab", []))
        m.trans = {}
        for k, v in d.get("transitions", {}).items():
            m.trans[tuple(k.split(" "))] = {t: int(c) for t, c in v.items()}
        m.probs = {}
        for key, cnt in m.trans.items():
            tot = sum(cnt.values())
            m.probs[key] = sorted(((t, c / tot) for t, c in cnt.items()),
                                  key=lambda kv: -kv[1])
        m.trained = bool(d.get("meta", {}).get("trained", True))
        m.source = d.get("meta", {}).get("source", "")
        return m

KW_SEP = "$%"
KW_SEP_ALIASES = ("$%",)

def split_keywords(kw):
    """把一行关键词拆成列表。主分隔符 $%，兼容 , 和 、 。"""
    text = str(kw)
    for alias in KW_SEP_ALIASES:
        text = text.replace(alias, "\x00")
    text = text.replace("，", "\x00").replace("、", "\x00").replace(",", "\x00")
    return [w.strip() for w in text.split("\x00") if w.strip()]

class QABot:
    """规则/检索式问答：关键词匹配查表。刻意做得「笨」，用于对比。"""

    def __init__(self):
        self.rules = []
        self.strategy = "contains"
        self.source = ""

    def add_rule(self, kw, answer):
        words = split_keywords(kw)
        self.rules.append((words, answer, kw))

    def load_csv(self, path):
        """载入问答库。宽容读法，尽量不让人踩坑：
          - 分隔符：逗号 / 制表符 / 分号 都行
          - 表头：自动跳过（关键词/keyword/question/问题…）
          - 也支持 `问题 = 回答`、`问题: 回答` 这种一行一条的写法
          - 超过两列时，从第二列起用 $% 拼成答案
        """
        n = 0
        HEADERS = ("关键词", "keyword", "keywords", "question", "问题", "问")
        with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                kw = ans = None
                if "=" in line:
                    kw, _, ans = line.partition("=")
                elif "：" in line:
                    kw, _, ans = line.partition("：")
                elif ":" in line and "," not in line and "\t" not in line:
                    kw, _, ans = line.partition(":")
                else:
                    cells = _split_line(line)
                    if len(cells) < 2:
                        continue
                    kw = cells[0]
                    ans = KW_SEP.join(cells[1:])
                kw, ans = (kw or "").strip(), (ans or "").strip()
                if not kw or not ans:
                    continue
                if kw.lower() in HEADERS:
                    continue
                self.add_rule(kw, ans)
                n += 1
        self.source = f"{os.path.basename(path)} ({n} " + _T("条规则") + ")"
        return n

    def answer(self, question):
        """返回 (回答, 命中关键词, 得分, 是否命中)。"""
        if not self.rules:
            return (_T("（规则库为空）"), [], 0.0, False)
        q = question.strip()
        best = None
        for kws, ans, raw in self.rules:
            score = 0.0
            hit = []
            if self.strategy == "exact":
                for kw in kws:
                    if q == kw:
                        score = 1.0
                        hit = [kw]
            elif self.strategy == "contains":
                for kw in kws:
                    if kw and kw in q:
                        score += 1.0
                        hit.append(kw)
                if kws:
                    score /= len(kws)
            else:
                for kw in kws:
                    if not kw:
                        continue
                    common = len(set(q) & set(kw))
                    denom = len(set(q) | set(kw)) or 1
                    sc = common / denom
                    if sc > score:
                        score = sc
                        hit = [kw]
            if best is None or score > best[2]:
                best = (ans, hit, score)
        ans, hit, score = best
        if self.strategy == "similar":
            hit_ok = score >= 0.34
        else:
            hit_ok = score > 0
        return (ans, hit, score, hit_ok)

    def to_dict(self):
        return {
            "type": "qabot",
            "strategy": self.strategy,
            "rules": [{"keywords": kws, "answer": ans, "raw": raw}
                      for kws, ans, raw in self.rules],
            "meta": {"n_rules": len(self.rules), "source": self.source},
        }

    @staticmethod
    def from_dict(d):
        m = QABot()
        m.strategy = d.get("strategy", "contains")
        for r in d.get("rules", []):
            m.rules.append((list(r.get("keywords", [])), r.get("answer", ""),
                            r.get("raw", "")))
        m.source = d.get("meta", {}).get("source", "")
        return m

class KNN:
    def __init__(self):
        self.k = 3
        self.X, self.y = [], []
        self.source = ""

    def train(self, dataset):
        self.X, self.y = list(dataset.X), list(dataset.y)
        return len(self.X)

    def query(self, px, py):
        d = []
        for i, p in enumerate(self.X):
            d.append((math.hypot(p[0] - px, p[1] - py), i))
        d.sort()
        neigh = d[:self.k]
        votes = {}
        for _, i in neigh:
            votes[self.y[i]] = votes.get(self.y[i], 0) + 1
        winner = max(votes, key=votes.get) if votes else 0
        return winner, [i for _, i in neigh], [dd for dd, _ in neigh]

    def decision_grid(self, bounds, res=28, sample=120):
        x0, x1, y0, y1 = bounds
        if len(self.X) > sample:
            idx = random.sample(range(len(self.X)), sample)
            X, y = [self.X[i] for i in idx], [self.y[i] for i in idx]
        else:
            X, y = self.X, self.y
        grid = []
        for gy in range(res):
            row = []
            yy = y0 + (y1 - y0) * gy / (res - 1)
            for gx in range(res):
                xx = x0 + (x1 - x0) * gx / (res - 1)
                d = sorted((math.hypot(p[0] - xx, p[1] - yy), j) for j, p in enumerate(X))[:self.k]
                votes = {}
                for _, j in d:
                    votes[y[j]] = votes.get(y[j], 0) + 1
                row.append(max(votes, key=votes.get) if votes else 0)
            grid.append(row)
        return grid

    def to_dict(self):
        return {
            "type": "knn",
            "k": self.k,
            "X": self.X,
            "y": self.y,
            "meta": {"n_samples": len(self.X), "source": self.source},
        }

    @staticmethod
    def from_dict(d):
        m = KNN()
        m.k = int(d["k"])
        m.X = [[float(v) for v in p] for p in d.get("X", [])]
        m.y = [int(v) for v in d.get("y", [])]
        m.source = d.get("meta", {}).get("source", "")
        return m

class KMeans:
    def __init__(self):
        self.k = 3
        self.X = []
        self.centroids = []
        self.assign = []
        self.iter = 0
        self.inertia = 0.0
        self.history = []
        self.source = ""
        self._done = False

    def configure(self, dataset, k):
        self.k = int(k)
        self.X = list(dataset.X)
        self.source = dataset.source
        self.reset()

    def reset(self):
        xs = [p[0] for p in self.X]
        ys = [p[1] for p in self.X]
        self.centroids = [[random.uniform(min(xs), max(xs)),
                           random.uniform(min(ys), max(ys))] for _ in range(self.k)]
        self.assign = [0] * len(self.X)
        self.iter = 0
        self.inertia = 0.0
        self.history = []
        self._done = False

    def step(self):
        """一次迭代：分配 + 更新簇心。返回快照，收敛后返回 None。"""
        if self._done or not self.X:
            return None
        changed = 0
        inertia = 0.0
        for i, p in enumerate(self.X):
            d = [math.hypot(p[0] - c[0], p[1] - c[1]) for c in self.centroids]
            best = min(range(len(d)), key=lambda j: d[j])
            if best != self.assign[i]:
                changed += 1
            self.assign[i] = best
            inertia += d[best] ** 2
        for j in range(self.k):
            pts = [self.X[i] for i in range(len(self.X)) if self.assign[i] == j]
            if pts:
                self.centroids[j] = [sum(p[0] for p in pts) / len(pts),
                                     sum(p[1] for p in pts) / len(pts)]
        self.iter += 1
        self.inertia = inertia
        self.history.append(inertia)
        if changed == 0 or self.iter >= 60:
            self._done = True
        return {"iter": self.iter, "inertia": self.inertia, "changed": changed}

    def decision_grid(self, bounds, res=28):
        x0, x1, y0, y1 = bounds
        grid = []
        for gy in range(res):
            row = []
            yy = y0 + (y1 - y0) * gy / (res - 1)
            for gx in range(res):
                xx = x0 + (x1 - x0) * gx / (res - 1)
                d = [math.hypot(xx - c[0], yy - c[1]) for c in self.centroids]
                row.append(min(range(len(d)), key=lambda j: d[j]))
            grid.append(row)
        return grid

    def to_dict(self):
        return {
            "type": "kmeans",
            "k": self.k,
            "centroids": self.centroids,
            "meta": {
                "n_samples": len(self.X),
                "iterations": self.iter,
                "inertia": round(self.inertia, 6),
                "source": self.source,
            },
        }

    @staticmethod
    def from_dict(d):
        m = KMeans()
        m.k = int(d["k"])
        m.centroids = [[float(v) for v in c] for c in d.get("centroids", [])]
        meta = d.get("meta", {})
        m.iter = int(meta.get("iterations", 0))
        m.inertia = float(meta.get("inertia", 0.0))
        m.source = meta.get("source", "")
        m._done = True
        return m

GA_FUNCS = {
    "quad": ("f(x)=-(x-3)²+9", lambda x: -(x - 3) ** 2 + 9, -6, 10),
    "sine": ("f(x)=sin(x)·x/3", lambda x: math.sin(x) * x / 3.0, -8, 8),
    "bimodal": ("bimodal f(x)=sin(x)+sin(2.3x)", lambda x: math.sin(x) + math.sin(2.3 * x), -10, 10),
}

class GA:
    """一维函数的遗传算法求最大。种群在实数区间上进化。"""

    def __init__(self):
        self.func = "quad"
        self.pop_size = 30
        self.generations = 60
        self.mut_rate = 0.1
        self.pop = []
        self.gen = 0
        self.best_x = 0.0
        self.best_f = -1e9
        self.history = []
        self._done = False

    def configure(self, func, pop, gens, mut):
        self.func = func if func in GA_FUNCS else "quad"
        self.pop_size = int(pop)
        self.generations = int(gens)
        self.mut_rate = float(mut)
        self.reset()

    @property
    def domain(self):
        return GA_FUNCS[self.func][2], GA_FUNCS[self.func][3]

    @property
    def fn(self):
        return GA_FUNCS[self.func][1]

    @property
    def label(self):
        return GA_FUNCS[self.func][0]

    def reset(self):
        lo, hi = self.domain
        self.pop = [random.uniform(lo, hi) for _ in range(self.pop_size)]
        self.gen = 0
        self.best_x = 0.0
        self.best_f = -1e9
        self.history = []
        self._done = False

    def step(self):
        if self._done or self.gen >= self.generations:
            return None
        lo, hi = self.domain
        fits = [self.fn(x) for x in self.pop]
        bi = max(range(len(self.pop)), key=lambda i: fits[i])
        if fits[bi] > self.best_f:
            self.best_f = fits[bi]
            self.best_x = self.pop[bi]
        self.history.append(self.best_f)
        fmin = min(fits)
        w = [f - fmin + 1e-6 for f in fits]
        tot = sum(w)
        def pick():
            r = random.random() * tot
            acc = 0
            for i, wi in enumerate(w):
                acc += wi
                if r <= acc:
                    return self.pop[i]
            return self.pop[-1]
        new = [self.pop[bi]]
        while len(new) < self.pop_size:
            a, b = pick(), pick()
            child = (a + b) / 2 + random.gauss(0, 0.1 * (hi - lo))
            if random.random() < self.mut_rate:
                child += random.gauss(0, 0.15 * (hi - lo))
            new.append(max(lo, min(hi, child)))
        self.pop = new
        self.gen += 1
        if self.gen >= self.generations:
            self._done = True
        return {"gen": self.gen, "best_x": self.best_x, "best_f": self.best_f}

    def to_dict(self):
        return {
            "type": "ga",
            "func": self.func,
            "best_x": round(self.best_x, 6),
            "best_f": round(self.best_f, 6),
            "meta": {
                "pop_size": self.pop_size,
                "generations": self.generations,
                "mut_rate": self.mut_rate,
                "ran_generations": self.gen,
                "domain": list(self.domain),
            },
        }

    @staticmethod
    def from_dict(d):
        m = GA()
        m.func = d.get("func", "quad")
        meta = d.get("meta", {})
        m.pop_size = int(meta.get("pop_size", 30))
        m.generations = int(meta.get("generations", 60))
        m.mut_rate = float(meta.get("mut_rate", 0.1))
        m.gen = int(meta.get("ran_generations", 0))
        m.best_x = float(d.get("best_x", 0.0))
        m.best_f = float(d.get("best_f", -1e9))
        m._done = True
        return m

MODEL_FORMAT = "aiblocks-model"
MODEL_VERSION = 1

class RecogModel:
    """音频/图片通用识别器：扫描样本文件夹 → 提特征 → 训练 MLP。

    样本组织方式（傻瓜式）：
        样本文件夹/
            猫/  a.wav b.wav ...
            狗/  c.wav d.wav ...
    子文件夹名 = 类别名。音频用 wav，图片用 png/gif/bmp/ppm。
    """

    def __init__(self):
        self.kind = "audio"
        self.class_names = []
        self.n_features = AUDIO_FEAT_DIM
        self.mlp = None
        self.folder = ""
        self.samples = []
        self.test = []
        self.hidden = 16
        self.epochs = 150
        self.lr = 0.3
        self.n_train = 0
        self.acc = 0.0
        self.test_acc = 0.0

    def feats(self, path):
        return (audio_features(path) if self.kind == "audio"
                else image_features(path))

    def _exts(self):
        if self.kind == "audio":
            return (".wav",)
        return (".png", ".gif", ".bmp", ".ppm", ".pgm", ".jpg", ".jpeg")

    def scan(self, folder):
        """返回 (类别名列表, [(path, label)], 跳过数)。"""
        names = []
        items = []
        skipped = 0
        exts = self._exts()
        for entry in sorted(os.listdir(folder)):
            sub = os.path.join(folder, entry)
            if not os.path.isdir(sub):
                continue
            names.append(entry)
            for fn in sorted(os.listdir(sub)):
                if not fn.lower().endswith(exts):
                    continue
                items.append((os.path.join(sub, fn), len(names) - 1))
        if self.kind == "image":
            for e in list(exts):
                if e in (".jpg", ".jpeg"):
                    continue
        return names, items, skipped

    def prepare(self, X, y, names, folder, hidden=None, epochs=None, lr=0.3):
        """把「扫到的特征」准备成可训练的 MLP。真正跑训练交给 step()。"""
        self.folder = folder
        if hidden is not None:
            self.hidden = int(hidden)
        if epochs is not None:
            self.epochs = int(epochs)
        self.lr = float(lr)
        self.class_names = list(names)
        self.n_features = len(X[0])
        self.samples = list(zip(X, y))

        n_cls = len(names)
        rng = random.Random(42)
        idx = list(range(len(X)))
        rng.shuffle(idx)
        cut = max(1, int(len(idx) * 0.75))
        if cut >= len(idx):
            cut = max(1, len(idx) - 1)
        tr, te = idx[:cut], idx[cut:]
        self.n_train = len(tr)
        ds = Dataset()
        ds.X = [X[i] for i in tr]
        ds.y = [y[i] for i in tr]
        ds.n_classes = n_cls
        ds.class_names = list(names)
        self.test = [(X[i], y[i]) for i in te]

        m = MLP()
        m.hidden = [(self.hidden, "relu"), (max(4, self.hidden // 2), "tanh")]
        m.configure(ds, self.lr, self.epochs)
        self.mlp = m
        return self

    def finish(self):
        """训练步跑完后收尾，算最终准确率。"""
        if self.mlp:
            self.acc = self.mlp.acc
        self.test_acc = self.accuracy()
        return self.acc, self.test_acc

    def train(self, folder, hidden=None, epochs=None, lr=0.3, progress=None):
        """同步训练（一步到位）。纯逻辑调用用这个，界面用 prepare+step 分帧。"""
        names, items, _ = self.scan(folder)
        if not names:
            raise ValueError(_T("没找到类别文件夹。请在样本文件夹里建几个子文件夹，"
                               "每个子文件夹放一类的样本"))
        if len(items) < 2:
            raise ValueError(_T("样本太少，至少要有两个文件"))
        X, y = [], []
        for path, lab in items:
            try:
                X.append(self.feats(path))
            except Exception:
                continue
            y.append(lab)
        if len(X) < 2:
            raise ValueError(_T("能读出来的样本太少，检查一下文件格式"))
        self.prepare(X, y, names, folder, hidden, epochs, lr)
        while True:
            s = self.mlp.step()
            if s is None:
                break
            if progress is not None:
                progress(self.mlp)
        self.finish()
        return len(X), len(names)

    def stepwise(self):
        """返回 MLP 的 step，供界面分帧驱动。"""
        return self.mlp.step() if self.mlp else None

    def accuracy(self):
        if not self.mlp or not self.test:
            return 0.0
        ok = sum(1 for x, lab in self.test if self.mlp.predict(x) == lab)
        return ok / len(self.test)

    def predict_path(self, path):
        """返回 (类别名, 置信度, 所有类别的概率列表)。"""
        if not self.mlp:
            raise ValueError(_T("还没训练，先跑一下训练积木"))
        f = self.feats(path)
        if len(f) < self.n_features:
            f = f + [0.0] * (self.n_features - len(f))
        f = f[:self.n_features]
        p = self.mlp.predict_proba(f)
        cls = max(range(len(p)), key=lambda i: p[i])
        name = self.class_names[cls] if cls < len(self.class_names) else str(cls)
        return name, p[cls], p

    def predict_vec(self, f):
        """从原始特征向量直接预测（画板用）。"""
        if not self.mlp:
            raise ValueError(_T("还没训练，先跑一下训练积木"))
        if len(f) < self.n_features:
            f = f + [0.0] * (self.n_features - len(f))
        f = f[:self.n_features]
        p = self.mlp.predict_proba(f)
        cls = max(range(len(p)), key=lambda i: p[i])
        name = self.class_names[cls] if cls < len(self.class_names) else str(cls)
        return name, p[cls], p

    def to_dict(self):
        return {
            "type": "recog",
            "kind": self.kind,
            "class_names": self.class_names,
            "n_features": self.n_features,
            "hidden": self.hidden,
            "epochs": self.epochs,
            "lr": self.lr,
            "mlp": self.mlp.to_dict() if self.mlp else None,
            "meta": {
                "n_train": self.n_train,
                "n_classes": len(self.class_names),
                "acc": round(self.acc, 4),
                "test_acc": round(self.test_acc, 4),
                "folder": self.folder,
            },
        }

    @staticmethod
    def from_dict(d):
        m = RecogModel()
        m.kind = d.get("kind", "audio")
        m.class_names = list(d.get("class_names", []))
        m.n_features = int(d.get("n_features", AUDIO_FEAT_DIM))
        m.hidden = int(d.get("hidden", 16))
        m.epochs = int(d.get("epochs", 150))
        m.lr = float(d.get("lr", 0.3))
        if d.get("mlp"):
            m.mlp = MLP.from_dict(d["mlp"])
        meta = d.get("meta", {})
        m.n_train = int(meta.get("n_train", 0))
        m.acc = float(meta.get("acc", 0.0))
        m.test_acc = float(meta.get("test_acc", 0.0))
        m.folder = meta.get("folder", "")
        return m

def export_model(obj, path, extra=None):
    """把任意模型导出成通用 JSON。返回 (成功, 消息)。

    JSON 结构：
        { "format": "aiblocks-model", "version": 1,
          "board": "nn", "saved_at": "...", "model": { ... } }
    任何语言都能解析 —— 权重就是普通的嵌套数组。
    """
    import json
    import datetime
    payload = {
        "format": MODEL_FORMAT,
        "version": MODEL_VERSION,
        "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "model": obj.to_dict(),
    }
    if extra:
        payload.update(extra)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    return True, _T("已导出") + f": {os.path.basename(path)}"

def import_model(path):
    """读回导出的模型。返回 (模型对象, board 提示, 原始 dict)。"""
    import json
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("format") != MODEL_FORMAT:
        raise ValueError(_T("不是本程序导出的模型文件"))
    d = payload.get("model", {})
    t = d.get("type")
    if t == "mlp":
        return MLP.from_dict(d), "nn", d
    if t == "markov":
        return Markov.from_dict(d), "mm", d
    if t == "qabot":
        return QABot.from_dict(d), "qa", d
    if t == "knn":
        return KNN.from_dict(d), "classic", d
    if t == "kmeans":
        return KMeans.from_dict(d), "classic", d
    if t == "ga":
        return GA.from_dict(d), "classic", d
    if t == "recog":
        m = RecogModel.from_dict(d)
        return m, ("audio" if m.kind == "audio" else "image"), d
    raise ValueError(_T("无法识别的模型类型") + f": {t}")

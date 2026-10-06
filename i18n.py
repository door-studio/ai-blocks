# -*- coding: utf-8 -*-
"""AI Blocks 国际化 —— 中英双语。

设计：
  * 所有界面文案用「中文原文」作为 key，查表得到当前语言的译文。
  * 找不到 key 时原样返回（方便渐进式覆盖，也保证不会崩）。
  * 积木 label 里用 {} 占位，翻译时保持占位顺序一致。
  * 语言选择持久化到用户目录，下次启动沿用。
  * 首次启动按系统语言自动选择：非中文 → 英文。
"""

import json
import os
import sys

ZH = "zh"
EN = "en"
LANGS = (ZH, EN)

_state = {"lang": ZH}

def _config_dir():
    if sys.platform.startswith("win"):
        base = os.environ.get("APPDATA") or os.path.expanduser("~")
    elif sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Application Support")
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    return os.path.join(base, "AIBlocks")

def _config_path():
    return os.path.join(_config_dir(), "settings.json")

EN_MAP = {
    "用积木亲手搭出一个 AI，然后看着它从数据里学出来":
        "Build an AI with blocks, then watch it actually learn from data",
    "拖动积木 → 拼接模型 → 选择数据 → 运行 → 实时看它学习":
        "Drag blocks → snap a model → pick data → run → watch it learn live",
    "进入 →": "Enter →",
    "语言": "Language",
    "中文": "中文",
    "English": "English",

    "神经网络": "Neural Network",
    "搭网络 → 喂数据 → 看它自己变聪明 → 做预测":
        "Build a net → feed data → watch it improve → predict",
    "真正会「学习」的那种 AI。你亲手搭出网络结构，"
    "看着它一次次调错、损失一路下降、分类边界逐渐弯折，"
    "最后用学好的模型去预测新数据。这就是大模型的原理骨架。":
        "The kind of AI that genuinely \"learns\". You build the network structure "
        "by hand, watch it correct its mistakes again and again as the loss falls "
        "and the boundary bends, then predict new data with the trained model. "
        "This is the skeleton of how large models work.",

    "马尔可夫链": "Markov Chain",
    "数一数下一个字最可能是什么 → 照着往下编":
        "Count what character likely comes next → keep writing",
    "简单粗暴的「接话」模型：统计每个字后面最常跟哪个字，"
    "然后按概率一个字一个字往下接。它是 GPT 的极简前身 —— "
    "大语言模型做的也是「猜下一个词」，只是规模大得离谱。":
        "A plain \"continue the sentence\" model: it counts which character most "
        "often follows each character, then appends one character at a time by "
        "probability. This is the minimalist ancestor of GPT — large language "
        "models also \"guess the next word\", just at an absurd scale.",

    "预设回答": "Canned Replies",
    "人工写死规则 → 关键词匹配 → 查表回复":
        "Hand-written rules → keyword match → look up the reply",
    "最「笨」的 AI：答案全靠人提前写好，程序只会拿你的话去比对关键词。"
    "它存在的意义是当反面教材 —— 让你一眼看出「查字典」和「真学习」"
    "差在哪。跟神经网络对比着玩，感受最明显。":
        "The \"dumbest\" AI: every answer is written by hand ahead of time, and "
        "the program only compares your words against keywords. Its purpose is "
        "to be a counter-example — so you can see at a glance how \"looking it up\" "
        "differs from \"really learning\". Play it side by side with the neural "
        "network and the contrast is obvious.",

    "经典算法": "Classic Algorithms",
    "KNN 投票 / K-Means 分堆 / 遗传算法进化":
        "KNN votes / K-Means clusters / GA evolves",
    "三个好懂又好看的经典算法：KNN 看邻居投票决定类别、"
    "K-Means 让几个簇心慢慢挪到最合适的位置、"
    "遗传算法模拟自然选择，让一群候选解一代代进化出最优答案。":
        "Three classic algorithms that are both easy and fun to watch: KNN lets "
        "neighbors vote on a class, K-Means slides its centroids until they settle "
        "in the best spots, and a genetic algorithm mimics natural selection so a "
        "population of candidate solutions evolves toward the best answer.",

    "声音识别": "Sound Recognition",
    "把声音变成数字 → 喂给神经网络 → 学会听声辨类":
        "Turn sound into numbers -> feed a neural network -> learn to tell sounds apart",
    "声音在电脑里就是一串数字。这里把一段 wav 压成 24 个特征数字"
    "（多大声、多刺耳、音调多高），再丢给神经网络去学。"
    "你只要按类别建好文件夹放样本，它就能学会区分不同的声音。":
        "Sound is just a string of numbers inside a computer. Here a wav clip is "
        "compressed into 24 feature numbers (how loud, how noisy, how high), then "
        "handed to a neural network. You just create one folder per class, drop "
        "samples in, and it learns to tell the sounds apart.",

    "图片识别": "Image Recognition",
    "把图片变成数字 → 喂给神经网络 → 学会认图":
        "Turn pictures into numbers -> feed a neural network -> learn to recognize",
    "一张图在电脑里就是一堆像素数字。这里把它缩成 8×8 的灰度格子"
    "（64 个数字），再丢给神经网络去学。"
    "按类别建文件夹放图片，或者直接用画板手写，看它能不能认出来。":
        "A picture is just a pile of pixel numbers inside a computer. Here it is "
        "shrunk into an 8x8 grid of grayscale values (64 numbers) and handed to a "
        "neural network. Create one folder per class full of images, or just draw "
        "on the sketchpad, and see whether it can recognize it.",

    "训练识别器（用文件夹里的样本）": "Train recognizer (from folder samples)",
    "选一个文件夹，里面每个子文件夹的名字就是类别名，装该类的样本文件":
        "Pick a folder. Each subfolder is named after a class and holds that class's samples",
    "识别 {}": "Recognize {}",
    "挑一个文件，看训练好的识别器把它判成哪一类":
        "Pick a file and see which class the trained recognizer assigns it to",
    "报告识别准确率": "Report recognition accuracy",
    "拿留出来的样本考一考，看看学得怎么样":
        "Test it on held-out samples to see how well it learned",
    "样本文件夹 {}": "Samples folder {}",
    "每个子文件夹 = 一个声音类别，文件夹名就是类别名，里面放 wav":
        "Each subfolder = one sound class; the folder name is the class name, put wav files inside",
    "每个子文件夹 = 一个图片类别，文件夹名就是类别名，里面放 png/jpg":
        "Each subfolder = one image class; the folder name is the class name, put png/jpg inside",
    "隐藏层 {} 个神经元": "Hidden layer {} neurons",
    "识别器的脑容量。样本多、类别杂就调大一点":
        "The recognizer's brain size. Turn it up if you have many samples or messy classes",
    "识别器的脑容量。图越复杂、类别越多就调大一点":
        "The recognizer's brain size. Turn it up for more complex or more numerous classes",
    "训练 {} 轮": "Train {} epochs",
    "把样本反复看几遍。少了学不透，多了容易背答案":
        "Go through the samples again and again. Too few and it never learns; too many and it memorizes",
    "打开画板手写识别": "Open sketchpad to draw and recognize",
    "弹出一个小画板，用鼠标画一个图形，点识别看它认不认得":
        "Opens a small sketchpad; draw a shape with the mouse and click Recognize",

    "各类别识别概率": "Class probabilities",
    "训练后这里会显示每个类别的概率":
        "After training, the probability of each class shows up here",
    "选择样本文件夹": "Choose samples folder",
    "没有选文件夹": "No folder selected",
    "读文件夹失败": "Failed to read folder",
    "至少要有两个类别文件夹，每个装一类的样本":
        "You need at least two class folders, each holding one class of samples",
    "发现": "Found",
    "个类别": "classes",
    "个样本，正在提取特征…": "samples, extracting features…",
    "能读出来的样本太少，检查一下文件格式":
        "Too few samples could be read, check the file formats",
    "开始训练识别器": "Training recognizer",
    "个样本": "samples",
    "训练失败": "Training failed",
    "训练完成": "Training finished",
    "训练集": "Train",
    "测试集": "Test",
    "选择要识别的文件": "Choose a file to recognize",
    "音频文件": "Audio files",
    "图片文件": "Image files",
    "识别结果": "Result",
    "识别失败": "Recognition failed",
    "置信度": "Confidence",
    "还没训练，先跑一下训练积木":
        "No model yet. Run the training block first",
    "没找到类别文件夹。请在样本文件夹里建几个子文件夹，"
    "每个子文件夹放一类的样本":
        "No class folders found. Create a few subfolders inside the samples "
        "folder, each holding one class of samples",
    "样本太少，至少要有两个文件": "Too few samples, need at least two files",
    "音频是空的": "The audio is empty",
    "暂不支持这种图片格式，请用 png / gif / bmp / ppm":
        "This image format is not supported yet; use png / gif / bmp / ppm",
    "不是有效的 BMP 文件": "Not a valid BMP file",
    "只支持 24/32 位 BMP": "Only 24/32-bit BMP is supported",
    "不是有效的 PNG 文件": "Not a valid PNG file",
    "只支持 8 位 PNG 图片": "Only 8-bit PNG images are supported",
    "不是有效的 GIF 文件": "Not a valid GIF file",
    "文件为空": "The file is empty",
    "识别准确率": "Recognition accuracy",
    "个测试样本": "test samples",
    "画板 —— 画个图形让它认": "Sketchpad - draw a shape and let it guess",
    "画点什么，然后点识别": "Draw something, then click Recognize",
    "已清空": "Cleared",
    "我猜是": "I guess",
    "画板识别": "Sketchpad guess",
    "把要识别的文件路径贴进来": "Paste the path of the file to recognize",
    "找不到这个文件": "File not found",

    "返回": "Back",
    "运行": "Run",
    "停止": "Stop",
    "清空": "Clear",
    "示例": "Example",
    "就绪": "Ready",
    "运行完成 ✓": "Finished ✓",
    "正在运行…": "Running…",
    "运行中": "Running",
    "错误": "Error",
    "积木": "Blocks",
    "调色板": "Palette",
    "工作区": "Workspace",
    "数据": "Data",
    "模型": "Model",
    "训练": "Train",
    "推理": "Inference",
    "预测": "Predict",
    "流程": "Control",
    "输出": "Output",
    "可视化": "Visualization",
    "日志": "Log",
    "帮助": "Help",

    "网络结构（亮度=激活值）": "Network (brightness = activation)",
    "损失曲线 Loss": "Loss curve",
    "准确率 Accuracy": "Accuracy",
    "决策边界 + 数据点": "Decision boundary + points",
    "转移概率矩阵（行→列）": "Transition matrix (row → col)",
    "生成进度": "Generation progress",
    "匹配得分": "Match score",
    "数据 / 决策边界": "Data / decision boundary",
    "指标曲线": "Metric curve",
    "最优适应度": "best fitness",

    "当点击运行时": "when run clicked",
    "重复 {} 次": "repeat {} times",
    "等待 {} 秒": "wait {} seconds",
    "停止运行": "stop",
    "输出日志 {}": "log {}",
    "显示指标 {} = {}": "show metric {} = {}",

    "载入数据集 {}": "load dataset {}",
    "训练集比例 {}%": "train split {}%",
    "训练 {} 轮": "train {} epochs",
    "学习率 {}": "learning rate {}",
    "添加隐藏层 {} 个神经元 激活 {}": "add hidden layer {} units, activation {}",
    "构建模型": "build model",
    "开始训练（实时可见）": "start training (live)",
    "预测 {} 的结果": "predict {}",

    "载入序列文本 {}": "load sequence text {}",
    "模型阶数 n = {}": "model order n = {}",
    "切分单位 {}": "token unit {}",
    "按字": "char",
    "按词": "word",
    "统计转移概率": "count transitions",
    "生成 {} 个单元 温度 {}": "generate {} units, temp {}",
    "以 {} 开头": "start with {}",

    "载入问答库 {}": "load Q&A base {}",
    "添加规则 关键词 {} → 回答 {}": "add rule keywords {} → answer {}",
    "提问 {}": "ask {}",

    "载入散点数据 {}": "load scatter data {}",
    "邻居个数 K = {}": "neighbors K = {}",
    "记录样本（惰性训练）": "store samples (lazy train)",
    "查询点 x:{} y:{}": "query point x:{} y:{}",
    "载入聚类数据 {}": "load cluster data {}",
    "簇数量 K = {}": "clusters K = {}",
    "迭代聚类（看簇心移动）": "iterate clustering (watch centroids)",
    "目标函数 {}": "objective {}",
    "抛物线": "parabola",
    "正弦": "sine",
    "双峰": "bimodal",
    "种群 {} 代数 {} 变异率 {}": "population {} generations {} mutation {}",
    "进化求解（看种群收敛）": "evolve (watch convergence)",

    "选择文件…": "choose file…",
    "选择数据文件": "Choose data file",
    "CSV 文件": "CSV files",
    "文本文件": "Text files",
    "所有文件": "All files",

    "关键词之间用 $% 隔开": "separate keywords with $%",
    "首个关键词之间用 $% 隔开": "separate keywords with $%",

    "把积木拖到中间试试": "Drag a block into the middle to try it",
    "（暂无说明）": "(no description)",
    "CSV 格式：第一列问题关键词，第二列回答":
        "CSV format: column 1 = question keywords, column 2 = answer",
    "同一问题的多个说法用 $% 隔开，如：你好$%hi":
        "Separate several ways to ask the same thing with $%, e.g. hi$%hello",
    "问一句话，看它能不能匹配到规则":
        "Ask a sentence and see whether it matches a rule",

    "程序起点。点「运行」后从这里开始往下走":
        "The starting point. After you click Run, execution begins here",
    "把夹在中间的积木重复执行若干次":
        "Runs the blocks nested inside it a set number of times",
    "暂停一会儿再往下走，用来放慢节奏看清过程":
        "Pauses briefly before continuing — slow things down to see them clearly",
    "到此结束，后面的积木都不执行":
        "Ends here; any blocks after it won't run",
    "在右侧日志区打印一句话": "Prints a line into the log panel on the right",
    "把一个数字显示到面板上，方便观察变化":
        "Shows a number on the panel so you can watch it change",

    "留空 = 用内置数据（二团簇）。CSV 只需两列：x,y 和标签列":
        "Leave empty to use built-in data (two blobs). CSV needs just x, y and a label column",
    "拿多少数据用来学习，剩下的留着考试（测准确率）":
        "How much data is used to learn; the rest is kept aside as a test",
    "把所有数据反复看几遍。轮数越多学得越透，也越慢":
        "How many times it goes through all the data. More epochs = better learning, but slower",
    "每次调整的步子大小。太大容易冲过头，太小走得慢":
        "How big each adjustment step is. Too big overshoots; too small crawls",
    "多堆一层、多放几个神经元，模型就能学更复杂的形状":
        "Stack more layers and units so the model can learn more complex shapes",
    "按上面的层数和参数把网络搭起来（初始化权重）":
        "Builds the network from the settings above (initializes the weights)",
    "启动训练，右边会实时画出损失下降和决策边界变化":
        "Starts training; the right panel draws the loss and boundary changing live",
    "填两个数（用逗号隔开），看模型把它判成哪一类":
        "Type two numbers separated by a comma and see which class the model picks",

    "留空 = 用内置文本。也可以选一个 txt 文件当语料":
        "Leave empty for built-in text, or pick a .txt file as your corpus",
    "看前面几个字来猜下一个。n 越大越像原文，也越容易照抄":
        "Looks at the previous few characters to guess the next. Larger n copies the source more closely",
    "按「字」还是一整个「词」来统计和生成":
        "Count and generate by single character or by whole word",
    "数一遍：每个字后面最常跟哪个字，把概率记下来":
        "Counts which character most often follows each character and records the odds",
    "温度低 = 保守常选高频字；温度高 = 更随机、更敢发挥":
        "Low temperature = safely picks frequent characters; high = more random and creative",
    "给几个字当开头，让它接着往下写":
        "Give it a few characters to start from and let it continue",

    "留空 = 内置数据。CSV 只需两列：x,y 和标签列":
        "Leave empty for built-in data. CSV needs just x, y and a label column",
    "看最近的几个邻居投票决定答案。K 太小怕噪声，太大怕糊":
        "Nearest neighbors vote. Small K is noisy; large K blurs the boundary",
    "KNN 不用真训练，把样本记住就行，查的时候现算":
        "KNN doesn't really train — just store the samples and compute on demand",
    "给一个坐标，看它周围邻居是什么类别":
        "Give a coordinate and see what class its neighbors are",
    "留空 = 内置数据。聚类不需要标签，只要 x,y 两列":
        "Leave empty for built-in data. Clustering needs only x, y — no labels",
    "想把数据分成几堆": "How many clusters you want",
    "让几个簇心一遍遍挪位，直到找到最合适的位置":
        "Slides the centroids again and again until they settle in the best spots",
    "想找最大值的那个函数，选一个试试":
        "Pick one of these functions to search for its maximum",
    "种群 = 一次养多少个候选解；代数 = 繁殖几轮；变异率 = 突变的概率":
        "Population = how many candidates at once; generations = breeding rounds; mutation = chance of random change",
    "开始一场自然选择，看种群朝最优解聚拢":
        "Starts natural selection and watches the population converge on the best answer",

    "请先载入数据": "Please load data first",
    "请先构建模型": "Please build the model first",
    "请先建立索引": "Please build the index first",
    "请先统计转移概率": "Please count transitions first",
    "数据为空": "Data is empty",
    "已载入": "Loaded",
    "条样本": "samples",
    "条规则": "rules",
    "无匹配": "no match",
    "未知": "unknown",
    "（规则库为空）": "(rule base is empty)",
    "未标注": "unlabeled",
    "无标签": "no labels",
    "没有读到有效数据行": "no valid data rows found",

    "内置团簇": "built-in blobs",
    "内置 XOR（非线性可分）": "built-in XOR (non-linearly separable)",
    "内置同心圆": "built-in concentric rings",
    "类": "classes",
    "个单元": "units",
    "文件为空": "file is empty",
    "内环": "inner ring",
    "外环": "outer ring",
    "团簇·2类": "blobs · 2 classes",
    "团簇·3类": "blobs · 3 classes",
    "XOR·非线性": "XOR · non-linear",
    "同心圆": "concentric rings",

    "已停止": "Stopped",
    "已清空": "Cleared",
    "已保存": "Saved",
    "已打开": "Opened",
    "已选择：": "Selected: ",
    "已载入示例，点「运行」开始": "Example loaded — click Run to start",
    "工作区里没有「当点击运行时」积木。\n先从左侧拖一个出来放在最上面。":
        "No \"when run clicked\" block in the workspace.\nDrag one out from the "
        "left and put it at the top.",

    "载入数据集：": "Loaded dataset: ",
    "载入序列：": "Loaded sequence: ",
    "载入问答库：": "Loaded Q&A base: ",
    "载入数据：": "Loaded data: ",
    "载入内置问答库（3条）": "Loaded built-in Q&A base (3 rules)",
    "添加规则：": "Added rule: ",
    "添加隐藏层：": "Added hidden layer: ",
    "构建模型：": "Built model: ",
    "索引就绪，共 ": "Index ready, ",
    "统计完成：": "Counted transitions: ",
    "KNN 记录 ": "KNN stored ",
    "K-Means 收敛：": "K-Means converged: ",
    "GA 最优": "GA best",
    "置信度": "confidence",
    "预测输入应形如「1.5 2.0」": "Prediction input should look like \"1.5 2.0\"",
    "生成：": "Generated: ",
    "问：": "Q: ",
    "答：": "A: ",
    "分=": "score=",
    "命中词=": "hits=",
    "✓命中": "✓hit",
    "✗未命中": "✗miss",
    "请先载入数据并训练": "Please load data and train first",
    "查询": "Query",
    "个邻居": "neighbors",
    "训练完成：": "Training done: ",
    "轮": "epochs",
    "准确率=": "acc=",
    "测试集=": "test=",

    "你好$%hi": "hello$%hi",
    "你好！我是规则机器人。": "Hi! I'm a rule-based bot.",
    "天气$%下雨": "weather$%rain",
    "今天晴转多云。": "Sunny turning cloudy today.",
    "名字$%你是谁": "name$%who are you",
    "我是预设回答机器人。": "I'm a canned-reply bot.",

    "尚未训练": "not trained yet",
    "输入": "in",
    "隐藏": "hidden",
    "输出层": "out",

    "拖动积木搭出你的 AI · 数据全程本地运行": 
        "Drag blocks to build your AI · everything runs locally",
}

def get_lang():
    return _state["lang"]

def set_lang(lang):
    if lang in LANGS:
        _state["lang"] = lang

def t(zh_text):
    """把中文原文翻译成当前语言。找不到就用原文。"""
    if _state["lang"] == ZH:
        return zh_text
    return EN_MAP.get(zh_text, zh_text)

def detect_system_lang():
    """检测系统语言：中文环境返回 zh，其余返回 en。"""
    for var in ("LC_ALL", "LC_MESSAGES", "LANG"):
        v = os.environ.get(var, "")
        if v:
            low = v.lower()
            if low.startswith("zh"):
                return ZH
            if low[:2] in ("en",):
                return EN
    if sys.platform.startswith("win"):
        try:
            import ctypes
            langid = ctypes.windll.kernel32.GetUserDefaultUILanguage()
            if langid in (0x0804, 0x0404, 0x0C04, 0x1004):
                return ZH
            if (langid & 0x3FF) == 0x04:
                return ZH
            return EN
        except Exception:
            pass
    try:
        import locale
        loc = locale.getdefaultlocale()[0] or ""
        if loc.lower().startswith("zh"):
            return ZH
    except Exception:
        pass
    return EN

def load_pref():
    """读取保存的语言选择；没有则自动检测系统语言。"""
    try:
        with open(_config_path(), "r", encoding="utf-8") as f:
            data = json.load(f)
        lang = data.get("lang")
        if lang in LANGS:
            _state["lang"] = lang
            return lang
    except Exception:
        pass
    lang = detect_system_lang()
    _state["lang"] = lang
    return lang

def save_pref():
    try:
        os.makedirs(_config_dir(), exist_ok=True)
        with open(_config_path(), "w", encoding="utf-8") as f:
            json.dump({"lang": _state["lang"]}, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False

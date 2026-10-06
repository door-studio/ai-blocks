# -*- coding: utf-8 -*-
"""AI Blocks 积木定义 —— 四类通用积木 + 各板块专属积木。"""

SHAPE_HAT = "hat"
SHAPE_STACK = "stack"
SHAPE_CBLOCK = "cblock"
SHAPE_CAP = "cap"
SHAPE_BOOL = "bool"

IN_NUMBER = "number"
IN_TEXT = "text"
IN_COLOR = "color"
IN_SELECT = "select"
IN_PATH = "path"

COMMON_BLOCKS = {
    "when_start": {
        "cat": "control", "shape": SHAPE_HAT,
        "opcode": "when_start", "label": "当点击运行时", "inputs": [],
        "hint": "程序起点。点「运行」后从这里开始往下走",
    },
    "repeat": {
        "cat": "control", "shape": SHAPE_CBLOCK,
        "opcode": "repeat", "label": "重复 {} 次", "inputs": [(10, IN_NUMBER, None, 44)],
        "hint": "把夹在中间的积木重复执行若干次",
    },
    "wait": {
        "cat": "control", "shape": SHAPE_STACK,
        "opcode": "wait", "label": "等待 {} 秒", "inputs": [(0.2, IN_NUMBER, None, 46)],
        "hint": "暂停一会儿再往下走，用来放慢节奏看清过程",
    },
    "stop": {
        "cat": "control", "shape": SHAPE_CAP,
        "opcode": "stop", "label": "停止运行", "inputs": [],
        "hint": "到此结束，后面的积木都不执行",
    },

    "print": {
        "cat": "output", "shape": SHAPE_STACK,
        "opcode": "print", "label": "输出日志 {}", "inputs": [("hello", IN_TEXT, None, 110)],
        "hint": "在右侧日志区打印一句话",
    },
    "show_metric": {
        "cat": "output", "shape": SHAPE_STACK,
        "opcode": "show_metric",
        "label": "显示指标 {} = {}", "inputs": [("loss", IN_TEXT, None, 70), (0, IN_NUMBER, None, 50)],
        "hint": "把一个数字显示到面板上，方便观察变化",
    },

    "save_model": {
        "cat": "output", "shape": SHAPE_STACK,
        "opcode": "save_model",
        "label": "导出模型 {} 到 {}",
        "inputs": [("model", IN_TEXT, None, 70), ("", IN_PATH, None, 140)],
        "hint": "把训好的模型存成一个通用 JSON 文件，别的语言也能读",
    },
    "load_model": {
        "cat": "data", "shape": SHAPE_STACK,
        "opcode": "load_model",
        "label": "载入模型 {}", "inputs": [("", IN_PATH, None, 150)],
        "hint": "读回之前导出的模型，不用重新训练就能直接用",
    },
    "chat": {
        "cat": "predict", "shape": SHAPE_STACK,
        "opcode": "chat",
        "label": "开始对话", "inputs": [],
        "hint": "运行后底部出现输入框，可以连续对话，边聊边看它怎么回答",
    },

    "recog_train": {
        "cat": "train", "shape": SHAPE_STACK,
        "opcode": "recog_train",
        "label": "训练识别器（用文件夹里的样本）", "inputs": [],
        "hint": "选一个文件夹，里面每个子文件夹的名字就是类别名，装该类的样本文件",
    },
    "recog_predict": {
        "cat": "predict", "shape": SHAPE_STACK,
        "opcode": "recog_predict",
        "label": "识别 {}", "inputs": [("", IN_PATH, None, 150)],
        "hint": "挑一个文件，看训练好的识别器把它判成哪一类",
    },
    "recog_accuracy": {
        "cat": "predict", "shape": SHAPE_STACK,
        "opcode": "recog_accuracy",
        "label": "报告识别准确率", "inputs": [],
        "hint": "拿留出来的样本考一考，看看学得怎么样",
    },
}

NN_BLOCKS = {
    "load_dataset": {
        "cat": "data", "shape": SHAPE_STACK,
        "opcode": "load_dataset",
        "label": "载入数据集 {}", "inputs": [("", IN_PATH, None, 150)],
        "hint": "留空 = 用内置数据（二团簇）。CSV 只需两列：x,y 和标签列",
    },
    "set_split": {
        "cat": "data", "shape": SHAPE_STACK,
        "opcode": "set_split",
        "label": "训练集比例 {}%", "inputs": [(80, IN_NUMBER, None, 44)],
        "hint": "拿多少数据用来学习，剩下的留着考试（测准确率）",
    },
    "set_epochs": {
        "cat": "train", "shape": SHAPE_STACK,
        "opcode": "set_epochs",
        "label": "训练 {} 轮", "inputs": [(200, IN_NUMBER, None, 50)],
        "hint": "把所有数据反复看几遍。轮数越多学得越透，也越慢",
    },
    "set_lr": {
        "cat": "train", "shape": SHAPE_STACK,
        "opcode": "set_lr",
        "label": "学习率 {}", "inputs": [(0.1, IN_NUMBER, None, 56)],
        "hint": "每次调整的步子大小。太大容易冲过头，太小走得慢",
    },
    "add_layer": {
        "cat": "model", "shape": SHAPE_STACK,
        "opcode": "add_layer",
        "label": "添加隐藏层 {} 个神经元 激活 {}",
        "inputs": [(8, IN_NUMBER, None, 44),
                   ("relu", IN_SELECT, [("relu", "ReLU"), ("tanh", "Tanh"), ("sigmoid", "Sigmoid")], 76)],
        "hint": "多堆一层、多放几个神经元，模型就能学更复杂的形状",
    },
    "init_model": {
        "cat": "model", "shape": SHAPE_STACK,
        "opcode": "init_model", "label": "构建模型", "inputs": [],
        "hint": "按上面的层数和参数把网络搭起来（初始化权重）",
    },
    "train_model": {
        "cat": "train", "shape": SHAPE_STACK,
        "opcode": "train_model", "label": "开始训练（实时可见）", "inputs": [],
        "hint": "启动训练，右边会实时画出损失下降和决策边界变化",
    },
    "predict": {
        "cat": "predict", "shape": SHAPE_STACK,
        "opcode": "predict",
        "label": "预测 {} 的结果", "inputs": [("", IN_TEXT, None, 120)],
        "hint": "填两个数（用逗号隔开），看模型把它判成哪一类",
    },
}

MM_BLOCKS = {
    "mm_load": {
        "cat": "data", "shape": SHAPE_STACK,
        "opcode": "mm_load",
        "label": "载入序列文本 {}", "inputs": [("", IN_PATH, None, 150)],
        "hint": "留空 = 用内置文本。也可以选一个 txt 文件当语料",
    },
    "mm_order": {
        "cat": "model", "shape": SHAPE_STACK,
        "opcode": "mm_order",
        "label": "模型阶数 n = {}", "inputs": [(2, IN_NUMBER, None, 44)],
        "hint": "看前面几个字来猜下一个。n 越大越像原文，也越容易照抄",
    },
    "mm_unit": {
        "cat": "model", "shape": SHAPE_STACK,
        "opcode": "mm_unit",
        "label": "切分单位 {}",
        "inputs": [("char", IN_SELECT, [("char", "按字"), ("word", "按词")], 74)],
        "hint": "按「字」还是一整个「词」来统计和生成",
    },
    "mm_train": {
        "cat": "train", "shape": SHAPE_STACK,
        "opcode": "mm_train", "label": "统计转移概率", "inputs": [],
        "hint": "数一遍：每个字后面最常跟哪个字，把概率记下来",
    },
    "mm_generate": {
        "cat": "predict", "shape": SHAPE_STACK,
        "opcode": "mm_generate",
        "label": "生成 {} 个单元 温度 {}",
        "inputs": [(60, IN_NUMBER, None, 48), (1.0, IN_NUMBER, None, 44)],
        "hint": "温度低 = 保守常选高频字；温度高 = 更随机、更敢发挥",
    },
    "mm_seed": {
        "cat": "predict", "shape": SHAPE_STACK,
        "opcode": "mm_seed",
        "label": "以 {} 开头", "inputs": [("", IN_TEXT, None, 100)],
        "hint": "给几个字当开头，让它接着往下写",
    },
}

QA_BLOCKS = {
    "qa_load": {
        "cat": "data", "shape": SHAPE_STACK,
        "opcode": "qa_load",
        "label": "载入问答库 {}",
        "inputs": [("", IN_PATH, None, 150)],
        "hint": "CSV 格式：第一列问题关键词，第二列回答",
    },
    "qa_add": {
        "cat": "data", "shape": SHAPE_STACK,
        "opcode": "qa_add",
        "label": "添加规则 关键词 {} → 回答 {}",
        "inputs": [("", IN_TEXT, None, 90), ("", IN_TEXT, None, 110)],
        "hint": "同一问题的多个说法用 $% 隔开，如：你好$%hi",
    },
    "qa_ask": {
        "cat": "predict", "shape": SHAPE_STACK,
        "opcode": "qa_ask",
        "label": "提问 {}", "inputs": [("", IN_TEXT, None, 140)],
        "hint": "问一句话，看它能不能匹配到规则",
    },
}

CLASSIC_BLOCKS = {
    "knn_load": {
        "cat": "data", "shape": SHAPE_STACK,
        "opcode": "knn_load",
        "label": "载入散点数据 {}", "inputs": [("", IN_PATH, None, 150)],
        "hint": "留空 = 内置数据。CSV 只需两列：x,y 和标签列",
    },
    "knn_k": {
        "cat": "model", "shape": SHAPE_STACK,
        "opcode": "knn_k",
        "label": "邻居个数 K = {}", "inputs": [(3, IN_NUMBER, None, 44)],
        "hint": "看最近的几个邻居投票决定答案。K 太小怕噪声，太大怕糊",
    },
    "knn_train": {
        "cat": "train", "shape": SHAPE_STACK,
        "opcode": "knn_train", "label": "记录样本（惰性训练）", "inputs": [],
        "hint": "KNN 不用真训练，把样本记住就行，查的时候现算",
    },
    "knn_query": {
        "cat": "predict", "shape": SHAPE_STACK,
        "opcode": "knn_query",
        "label": "查询点 x:{} y:{}", "inputs": [(0, IN_NUMBER, None, 44), (0, IN_NUMBER, None, 44)],
        "hint": "给一个坐标，看它周围邻居是什么类别",
    },
    "km_load": {
        "cat": "data", "shape": SHAPE_STACK,
        "opcode": "km_load",
        "label": "载入聚类数据 {}", "inputs": [("", IN_PATH, None, 150)],
        "hint": "留空 = 内置数据。聚类不需要标签，只要 x,y 两列",
    },
    "km_k": {
        "cat": "model", "shape": SHAPE_STACK,
        "opcode": "km_k",
        "label": "簇数量 K = {}", "inputs": [(3, IN_NUMBER, None, 44)],
        "hint": "想把数据分成几堆",
    },
    "km_train": {
        "cat": "train", "shape": SHAPE_STACK,
        "opcode": "km_train", "label": "迭代聚类（看簇心移动）", "inputs": [],
        "hint": "让几个簇心一遍遍挪位，直到找到最合适的位置",
    },
    "ga_load": {
        "cat": "data", "shape": SHAPE_STACK,
        "opcode": "ga_load",
        "label": "目标函数 {}", "inputs": [("quad", IN_SELECT, [("quad", "抛物线"), ("sine", "正弦"), ("bimodal", "双峰")], 74)],
        "hint": "想找最大值的那个函数，选一个试试",
    },
    "ga_params": {
        "cat": "train", "shape": SHAPE_STACK,
        "opcode": "ga_params",
        "label": "种群 {} 代数 {} 变异率 {}",
        "inputs": [(30, IN_NUMBER, None, 42), (60, IN_NUMBER, None, 42), (0.1, IN_NUMBER, None, 42)],
        "hint": "种群 = 一次养多少个候选解；代数 = 繁殖几轮；变异率 = 突变的概率",
    },
    "ga_run": {
        "cat": "predict", "shape": SHAPE_STACK,
        "opcode": "ga_run", "label": "进化求解（看种群收敛）", "inputs": [],
        "hint": "开始一场自然选择，看种群朝最优解聚拢",
    },
}

AUDIO_BLOCKS = {
    "au_folder": {
        "cat": "data", "shape": SHAPE_STACK,
        "opcode": "au_folder",
        "label": "样本文件夹 {}", "inputs": [("", IN_PATH, None, 150)],
        "hint": "每个子文件夹 = 一个声音类别，文件夹名就是类别名，里面放 wav",
    },
    "au_hidden": {
        "cat": "model", "shape": SHAPE_STACK,
        "opcode": "au_hidden",
        "label": "隐藏层 {} 个神经元", "inputs": [(16, IN_NUMBER, None, 44)],
        "hint": "识别器的脑容量。样本多、类别杂就调大一点",
    },
    "au_epochs": {
        "cat": "train", "shape": SHAPE_STACK,
        "opcode": "au_epochs",
        "label": "训练 {} 轮", "inputs": [(150, IN_NUMBER, None, 50)],
        "hint": "把样本反复看几遍。少了学不透，多了容易背答案",
    },
}

IMAGE_BLOCKS = {
    "im_folder": {
        "cat": "data", "shape": SHAPE_STACK,
        "opcode": "im_folder",
        "label": "样本文件夹 {}", "inputs": [("", IN_PATH, None, 150)],
        "hint": "每个子文件夹 = 一个图片类别，文件夹名就是类别名，里面放 png/jpg",
    },
    "im_hidden": {
        "cat": "model", "shape": SHAPE_STACK,
        "opcode": "im_hidden",
        "label": "隐藏层 {} 个神经元", "inputs": [(24, IN_NUMBER, None, 44)],
        "hint": "识别器的脑容量。图越复杂、类别越多就调大一点",
    },
    "im_epochs": {
        "cat": "train", "shape": SHAPE_STACK,
        "opcode": "im_epochs",
        "label": "训练 {} 轮", "inputs": [(200, IN_NUMBER, None, 50)],
        "hint": "把样本反复看几遍。少了学不透，多了容易背答案",
    },
    "im_draw": {
        "cat": "predict", "shape": SHAPE_STACK,
        "opcode": "im_draw",
        "label": "打开画板手写识别", "inputs": [],
        "hint": "弹出一个小画板，用鼠标画一个图形，点识别看它认不认得",
    },
}

BOARDS = {
    "nn": {
        "name": "神经网络",
        "sub": "搭网络 → 喂数据 → 看它自己变聪明 → 做预测",
        "icon": "NN",
        "color": "#4C6FE0",
        "blocks": NN_BLOCKS,
        "cats": ["data", "model", "train", "predict", "control", "output"],
        "desc": "真正会「学习」的那种 AI。你亲手搭出网络结构，"
                "看着它一次次调错、损失一路下降、分类边界逐渐弯折，"
                "最后用学好的模型去预测新数据。这就是大模型的原理骨架。",
    },
    "mm": {
        "name": "马尔可夫链",
        "sub": "数一数下一个字最可能是什么 → 照着往下编",
        "icon": "MK",
        "color": "#2E9E8F",
        "blocks": MM_BLOCKS,
        "cats": ["data", "model", "train", "predict", "control", "output"],
        "desc": "简单粗暴的「接话」模型：统计每个字后面最常跟哪个字，"
                "然后按概率一个字一个字往下接。它是 GPT 的极简前身 —— "
                "大语言模型做的也是「猜下一个词」，只是规模大得离谱。",
    },
    "qa": {
        "name": "预设回答",
        "sub": "人工写死规则 → 关键词匹配 → 查表回复",
        "icon": "QA",
        "color": "#E07B1A",
        "blocks": QA_BLOCKS,
        "cats": ["data", "predict", "control", "output"],
        "desc": "最「笨」的 AI：答案全靠人提前写好，程序只会拿你的话去比对关键词。"
                "它存在的意义是当反面教材 —— 让你一眼看出「查字典」和「真学习」"
                "差在哪。跟神经网络对比着玩，感受最明显。",
    },
    "classic": {
        "name": "经典算法",
        "sub": "KNN 投票 / K-Means 分堆 / 遗传算法进化",
        "icon": "CL",
        "color": "#8250E0",
        "blocks": CLASSIC_BLOCKS,
        "cats": ["data", "model", "train", "predict", "control", "output"],
        "desc": "三个好懂又好看的经典算法：KNN 看邻居投票决定类别、"
                "K-Means 让几个簇心慢慢挪到最合适的位置、"
                "遗传算法模拟自然选择，让一群候选解一代代进化出最优答案。",
    },
    "audio": {
        "name": "声音识别",
        "sub": "把声音变成数字 → 喂给神经网络 → 学会听声辨类",
        "icon": "AU",
        "color": "#D4568A",
        "blocks": AUDIO_BLOCKS,
        "cats": ["data", "model", "train", "predict", "control", "output"],
        "desc": "声音在电脑里就是一串数字。这里把一段 wav 压成 24 个特征数字"
                "（多大声、多刺耳、音调多高），再丢给神经网络去学。"
                "你只要按类别建好文件夹放样本，它就能学会区分不同的声音。",
    },
    "image": {
        "name": "图片识别",
        "sub": "把图片变成数字 → 喂给神经网络 → 学会认图",
        "icon": "IM",
        "color": "#3C9AE0",
        "blocks": IMAGE_BLOCKS,
        "cats": ["data", "model", "train", "predict", "control", "output"],
        "desc": "一张图在电脑里就是一堆像素数字。这里把它缩成 8×8 的灰度格子"
                "（64 个数字），再丢给神经网络去学。"
                "按类别建文件夹放图片，或者直接用画板手写，看它能不能认出来。",
    },
}

BOARD_ORDER = ["nn", "mm", "qa", "classic", "audio", "image"]

def merged_blocks(board_id):
    """某板块生效的完整积木表：通用积木 + 该板块专属积木。"""
    out = dict(COMMON_BLOCKS)
    out.update(BOARDS.get(board_id, {}).get("blocks", {}))
    return out

def board_text(board_id, field):
    """取板块的本地化文案（name / sub / desc）。"""
    import i18n
    return i18n.t(BOARDS.get(board_id, {}).get(field, ""))

BLOCKS = dict(COMMON_BLOCKS)
for _b in BOARDS.values():
    BLOCKS.update(_b["blocks"])

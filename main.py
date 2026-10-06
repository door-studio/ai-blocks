# -*- coding: utf-8 -*-
"""AI Blocks 入口。双击运行 / python main.py"""

import os
import sys
import tkinter as tk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import App
import i18n

def main():
    lang = i18n.load_pref()

    root = tk.Tk()
    try:
        root.tk.call("tk", "scaling", 1.2)
    except Exception:
        pass
    app = App(root)

    def on_close():
        try:
            job = getattr(app, "_tick_job", None)
            if job:
                root.after_cancel(job)
        except Exception:
            pass
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()

if __name__ == "__main__":
    main()

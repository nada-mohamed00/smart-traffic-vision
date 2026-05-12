import os
import tkinter as tk

import cv2
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from vision import canny, harris, objects, prewitt, road, sobel, template_match

SAMPLES = os.path.join(os.path.dirname(__file__), "samples")


def to_rgb(img):
    return img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def build_panels(path):
    bgr = cv2.imread(path)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    matched, tpl = template_match(bgr)
    return [
        ("Original", bgr),
        ("Sobel", sobel(gray)),
        ("Prewitt", prewitt(gray)),
        ("Canny", canny(gray)),
        ("Objects", objects(bgr)),
        ("Road", road(bgr)),
        ("Harris", harris(bgr)),
        ("Template", tpl),
        ("Match", matched),
    ]


def main():
    root = tk.Tk()
    root.title("Smart Traffic Vision System")
    root.geometry("1280x820")
    root.configure(bg="#1e1e2e")

    tk.Label(
        root, text="Smart Traffic Vision",
        font=("Helvetica", 18, "bold"),
        fg="#89b4fa", bg="#1e1e2e",
    ).pack(pady=(14, 4))

    bar = tk.Frame(root, bg="#1e1e2e")
    bar.pack(pady=(0, 10))

    fig = Figure(figsize=(13, 8), facecolor="#1e1e2e")
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(fill="both", expand=True, padx=14, pady=(0, 14))

    title = tk.StringVar(value="Choose a sample to analyse")
    tk.Label(
        root, textvariable=title,
        font=("Helvetica", 11, "bold"),
        fg="#a6adc8", bg="#1e1e2e",
    ).pack(before=bar, pady=(0, 6))

    def show(path):
        title.set(os.path.basename(path))
        fig.clear()
        for i, (name, img) in enumerate(build_panels(path)):
            ax = fig.add_subplot(3, 3, i + 1)
            ax.imshow(to_rgb(img), cmap="gray" if len(img.shape) == 2 else None)
            ax.set_title(name, color="#cdd6f4", fontsize=10)
            ax.axis("off")
            ax.set_facecolor("#1e1e2e")
        fig.tight_layout()
        canvas.draw()

    files = sorted(
        f for f in os.listdir(SAMPLES)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    )
    for name in files:
        path = os.path.join(SAMPLES, name)
        tk.Button(
            bar, text=name,
            font=("Helvetica", 11, "bold"),
            bg="#89b4fa", fg="#1e1e2e",
            activebackground="#74c7ec", relief="flat",
            padx=14, pady=6,
            command=lambda p=path: show(p),
        ).pack(side="left", padx=6)

    root.mainloop()


if __name__ == "__main__":
    main()

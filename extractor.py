from __future__ import annotations

"""extractor.py
Annotate LaTeX regions in lecture slides *and* run a simulated OCR worker.

CLI
---
python extractor.py slides.pdf -o latex_regions

Positional arguments
--------------------
pdf               Path to the PDF file to annotate (default: slides.pdf)

Optional arguments
------------------
-o, --out         Output directory where image crops and OCR .tex files are stored
                  (default: ./latex_regions)

Key bindings inside the **Slide Viewer** window
------------------------------------------------
click-drag : draw a box
u          : undo last box
q          : save boxes & next slide
b          : save boxes & back one slide
Esc        : quit program
"""

import argparse
import itertools
import time
from multiprocessing import Event, Process
from pathlib import Path

import cv2
import fitz  # PyMuPDF
import numpy as np


_LATEX_SNIPPETS: list[str] = [
    r"\hat{y}=\sigma(Wx+b)",
    r"L=\frac{1}{N}\sum_{i=1}^{N}(y_i-\hat{y}_i)^2",
    r"p(z\mid x)=\frac{p(x\mid z)p(z)}{p(x)}",
    r"\theta \leftarrow \theta-\eta\nabla_\theta L",
    r"q(z) \approx p(z \mid x)",
    r"\mathrm{ELBO}=\mathbb{E}_{q}[\log p(x,z)]-\mathbb{E}_{q}[\log q(z)]",
    r"K(x_i,x_j)=\exp\left(-\frac{\|x_i-x_j\|^2}{2\sigma^2}\right)",
    r"a^{(l)}=\mathrm{ReLU}(W^{(l)}a^{(l-1)}+b^{(l)})",
    r"\text{softmax}(z)_k = \frac{e^{z_k}}{\sum_j e^{z_j}}",
    r"f(x)=\mathrm{sign}(w^Tx+b)",
]
_LATEX_CYCLE = itertools.cycle(_LATEX_SNIPPETS)


def ocr_worker(folder: Path, stop_event: Event) -> None:
    """Simulate a heavy OCR service that watches *folder* for new PNG crops."""
    print("[OCR] Worker started")
    while True:
        if stop_event.is_set():
            break
        if not folder.exists():
            time.sleep(2)
            continue
        work_found = False
        for png_path in folder.glob("*.png"):
            if stop_event.is_set():
                break
            tex_path = png_path.with_suffix(".tex")
            if tex_path.exists():
                continue
            work_found = True
            latex_eq = next(_LATEX_CYCLE)
            print(f"[OCR] Processing {png_path.name} -> '{latex_eq}'")
            for _ in range(30):  # simulate inference
                if stop_event.is_set():
                    break
                time.sleep(0.1)
            if stop_event.is_set():
                break
            tex_path.write_text(latex_eq + "\n")
            print(f"[OCR]   -> wrote {tex_path.name}")
        if not work_found:
            time.sleep(1)
    print("[OCR] Worker shutting down")




class BoxDrawer:
    """Handles user interaction on a *single* slide image."""

    def __init__(self, image: np.ndarray, slide_num: int, total: int) -> None:
        self.window_name = "Slide Viewer"
        self.original = image
        self.boxes: list[tuple[tuple[int, int], tuple[int, int]]] = []
        self.slide_num = slide_num
        self.total_slides = total
        cv2.setMouseCallback(self.window_name, self._mouse_cb)

    def _mouse_cb(self, event, x, y, flags, _) -> None:  # noqa: D401
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            end = (x, y)
            self.boxes.append((self.start, end))

    def run(self) -> tuple[str, list[tuple]]:
        while True:
            frame = self.original.copy()
            for pt1, pt2 in self.boxes:
                cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 2)
            title = f"Slide Viewer - ({self.slide_num} / {self.total_slides})"
            cv2.setWindowTitle(self.window_name, title)
            cv2.imshow(self.window_name, frame)
            key = cv2.waitKey(1)
            if key == ord("w"):
                return "next", self.boxes
            if key == ord("q"):
                return "back", self.boxes
            if key == 27:
                return "quit", []
            if key == ord("u") and self.boxes:
                self.boxes.pop()




def annotate_pdf(pdf_path: Path, out_dir: Path) -> None:
    doc = fitz.open(pdf_path)
    n_slides = len(doc)
    cv2.namedWindow("Slide Viewer", cv2.WINDOW_NORMAL)
    cv2.moveWindow("Slide Viewer", 100, 100)
    slide_idx = 0

    while 0 <= slide_idx < n_slides:
        page = doc[slide_idx]
        pix = page.get_pixmap(dpi=200)
        img = cv2.imdecode(
            np.frombuffer(pix.tobytes("png"), np.uint8), cv2.IMREAD_COLOR
        )
        drawer = BoxDrawer(img, slide_idx + 1, n_slides)
        action, boxes = drawer.run()

        if action == "quit":
            break

        # Consolidated crop saving logic for "back" and "next"
        if boxes:
            for j, (pt1, pt2) in enumerate(boxes, start=1):
                x1, y1 = pt1
                x2, y2 = pt2
                h, w, _ = img.shape
                x1, x2 = sorted((max(0, x1), min(w, x2)))
                y1, y2 = sorted((max(0, y1), min(h, y2)))
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                crop_path = out_dir / f"slide_{slide_idx + 1:03}_crop_{j}.png"
                cv2.imwrite(str(crop_path), crop)
                print(f"[GUI] Saved {crop_path.name}")

        if action == "back" and slide_idx > 0:
            slide_idx -= 1
        elif action == "next":
            slide_idx += 1

    cv2.destroyAllWindows()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Annotate LaTeX regions in a PDF deck of slides."
    )
    parser.add_argument(
        "pdf",
        nargs="?",
        default="slides.pdf",
        help="Path to the PDF slide deck (default: slides.pdf)",
    )
    parser.add_argument(
        "-o",
        "--out",
        default="latex_regions",
        help="Directory to store image crops and OCR .tex files (default: ./latex_regions)",
    )
    args = parser.parse_args()

    PDF_FILE = Path(args.pdf).expanduser().resolve()
    OUT_DIR = Path(args.out).expanduser().resolve()

    if not PDF_FILE.is_file():
        raise FileNotFoundError(f"PDF file not found: {PDF_FILE}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    stop_evt = Event()
    worker = Process(target=ocr_worker, args=(OUT_DIR, stop_evt))
    worker.start()

    try:
        annotate_pdf(PDF_FILE, OUT_DIR)
    finally:
        stop_evt.set()
        worker.join()
        print("All done. Bye!")

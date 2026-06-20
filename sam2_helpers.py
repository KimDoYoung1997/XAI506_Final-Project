"""SAM2 inference + PyQt5 point picker (from scripts/01_hf_sam2.ipynb)."""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_fill_holes, label
from transformers import Sam2Model, Sam2Processor

MODEL_ID = "facebook/sam2.1-hiera-base-plus"


def pick_torch_device(verbose: bool = True) -> torch.device:
    forced = os.environ.get("SAM2_DEVICE", "").strip().lower()
    if forced in ("cuda", "mps", "cpu"):
        if forced == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("SAM2_DEVICE=cuda but CUDA is not available.")
        if forced == "mps":
            if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                raise RuntimeError("SAM2_DEVICE=mps but MPS is not available.")
        if verbose:
            print(f"[device] SAM2_DEVICE={forced}")
        return torch.device(forced)
    if torch.cuda.is_available():
        if verbose:
            print("[device] CUDA")
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        if verbose:
            print("[device] MPS")
        return torch.device("mps")
    if verbose:
        print("[device] CPU")
    return torch.device("cpu")


def to_int_xy(p):
    try:
        return [int(round(float(p[0]))), int(round(float(p[1])))]
    except Exception:
        return None


def norm_points_xy_list(xy_list):
    if xy_list is None:
        return []
    out = []
    for p in xy_list:
        q = to_int_xy(p)
        if q is not None:
            out.append(q)
    return out


def keep_largest_cc(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask).astype(bool)
    structure = np.ones((3, 3), dtype=np.int8)
    lab, n = label(mask, structure=structure)
    if int(n) == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    if sizes.size > 0:
        sizes[0] = 0
    k = int(np.argmax(sizes)) if sizes.size > 0 else 0
    return lab == k


@torch.inference_mode()
def sam2_predict_masks(
    model: Sam2Model,
    processor: Sam2Processor,
    device: torch.device,
    img_pil: Image.Image,
    pos,
    neg=None,
    fill_holes: bool = True,
    keep_largest: bool = True,
) -> np.ndarray:
    pos = norm_points_xy_list(pos)
    neg = norm_points_xy_list(neg or [])
    if len(pos) == 0 and len(neg) == 0:
        raise ValueError("Need at least one positive or negative point.")
    pts = pos + neg
    labels = [1] * len(pos) + [0] * len(neg)
    inputs = processor(
        images=img_pil,
        input_points=[[pts]],
        input_labels=[[labels]],
        return_tensors="pt",
    )
    for k in inputs:
        t = inputs[k]
        if hasattr(t, "to"):
            inputs[k] = t.to(device)
    outputs = model(**inputs)
    masks = processor.post_process_masks(
        outputs.pred_masks.cpu(), inputs["original_sizes"]
    )[0]
    cand_masks = masks[0].numpy().astype(bool)
    cand_i = int(np.argmax(cand_masks.reshape(3, -1).sum(axis=1)))
    mask = cand_masks[cand_i]
    if fill_holes:
        mask = binary_fill_holes(mask)
    if keep_largest:
        mask = keep_largest_cc(mask)
    return mask.astype(bool)


def load_sam2(device: torch.device | None = None):
    device = device or pick_torch_device()
    processor = Sam2Processor.from_pretrained(MODEL_ID)
    model = Sam2Model.from_pretrained(MODEL_ID).to(device)
    model.eval()
    print(f"loaded: {MODEL_ID} on {device}")
    return model, processor, device


def pick_pos_neg_points_qt(
    img_np,
    title="Point Picker",
    window_width=0.5,
    window_height=0.5,
    verbose=True,
):
    """Left=pos, Right=neg, Enter/Esc=done, Backspace=undo."""
    import matplotlib.pyplot as plt
    from PyQt5.QtCore import QPoint, Qt, QTimer
    from PyQt5.QtGui import QFont, QImage, QPainter, QPen, QPixmap
    from PyQt5.QtWidgets import QApplication, QLabel, QSizePolicy, QVBoxLayout, QWidget

    def _np_to_qimage_rgb(x):
        x = np.asarray(x)
        if x.dtype != np.uint8:
            x = np.clip(x, 0, 255).astype(np.uint8)
        if x.ndim == 2:
            x = np.stack([x, x, x], axis=-1)
        if x.ndim == 3 and x.shape[-1] == 4:
            x = x[..., :3]
        h, w = x.shape[0], x.shape[1]
        return QImage(x.data, w, h, 3 * w, QImage.Format_RGB888).copy()

    def _resolve_screen_value(v, full):
        v = float(v)
        if 0.0 < v <= 1.0:
            return int(round(v * float(full)))
        return int(round(v))

    def _get_monitor_size_fallback():
        app = QApplication.instance()
        if app is None:
            return (1920, 1080)
        scr = app.primaryScreen()
        if scr is None:
            return (1920, 1080)
        g = scr.availableGeometry()
        return int(g.width()), int(g.height())

    img_np = np.asarray(img_np)
    H0, W0 = img_np.shape[0], img_np.shape[1]

    class _Win(QWidget):
        def __init__(self):
            super().__init__()
            self.setWindowTitle(str(title))
            self.pos = []
            self.neg = []
            self._font = QFont("Menlo", 10)
            qimg0 = _np_to_qimage_rgb(img_np)
            self._pm0 = QPixmap.fromImage(qimg0)
            self._sx = 1.0
            self._sy = 1.0
            self._pm_disp = self._pm0
            self._label = QLabel(self)
            self._label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
            layout = QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(self._label)
            self.setLayout(layout)
            sw, sh = _get_monitor_size_fallback()
            self.setGeometry(
                _resolve_screen_value(0, sw),
                _resolve_screen_value(0, sh),
                _resolve_screen_value(window_width, sw),
                _resolve_screen_value(window_height, sh),
            )
            if verbose:
                print(f"[picker] {title}")
                print("[picker] Left=pos, Right=neg, Enter/Esc=done, Backspace=undo")
            QTimer.singleShot(0, self._update_pixmap)

        def _target_disp_size(self):
            r = self._label.contentsRect()
            w_av = max(1, int(r.width()))
            h_av = max(1, int(r.height()))
            s = min(float(w_av) / W0, float(h_av) / H0)
            return max(1, int(round(W0 * s))), max(1, int(round(H0 * s)))

        def _update_pixmap(self):
            w_disp, h_disp = self._target_disp_size()
            self._pm_disp = self._pm0.scaled(
                w_disp, h_disp, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self._label.setPixmap(self._pm_disp)
            self._sx = float(W0) / float(self._pm_disp.width())
            self._sy = float(H0) / float(self._pm_disp.height())
            self._redraw()

        def resizeEvent(self, event):
            super().resizeEvent(event)
            self._update_pixmap()

        def _redraw(self):
            pm = QPixmap(self._pm_disp)
            p = QPainter(pm)
            p.setFont(self._font)
            pen_pos = QPen(Qt.red, 2)
            pen_neg = QPen(Qt.blue, 2)
            for x0, y0 in self.pos:
                xd = int(round(x0 / self._sx))
                yd = int(round(y0 / self._sy))
                p.setPen(pen_pos)
                p.drawEllipse(QPoint(xd, yd), 6, 6)
            for x0, y0 in self.neg:
                xd = int(round(x0 / self._sx))
                yd = int(round(y0 / self._sy))
                p.setPen(pen_neg)
                p.drawLine(xd - 6, yd - 6, xd + 6, yd + 6)
                p.drawLine(xd - 6, yd + 6, xd + 6, yd - 6)
            p.end()
            self._label.setPixmap(pm)

        def _event_to_img_xy(self, event):
            gp = self._label.mapFrom(self, QPoint(int(event.x()), int(event.y())))
            xd, yd = int(gp.x()), int(gp.y())
            if xd < 0 or yd < 0 or xd >= self._pm_disp.width() or yd >= self._pm_disp.height():
                return None
            x0 = max(0, min(W0 - 1, int(round(xd * self._sx))))
            y0 = max(0, min(H0 - 1, int(round(yd * self._sy))))
            return x0, y0

        def mousePressEvent(self, event):
            xy = self._event_to_img_xy(event)
            if xy is None:
                return
            x0, y0 = xy
            if event.button() == Qt.LeftButton:
                self.pos.append([x0, y0])
                if verbose:
                    print(f"pos: ({x0},{y0})")
            elif event.button() == Qt.RightButton:
                self.neg.append([x0, y0])
                if verbose:
                    print(f"neg: ({x0},{y0})")
            self._redraw()

        def keyPressEvent(self, event):
            k = event.key()
            if k in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Escape):
                self.close()
                return
            if k == Qt.Key_Backspace:
                if self.neg:
                    self.neg.pop()
                elif self.pos:
                    self.pos.pop()
                self._redraw()

    app = QApplication.instance()
    created = False
    if app is None:
        app = QApplication(sys.argv)
        created = True
    win = _Win()
    win.show()
    app.exec_()
    pos, neg = list(win.pos), list(win.neg)
    if created:
        try:
            app.quit()
        except Exception:
            pass
    return pos, neg


def segment_player(
    model,
    processor,
    device,
    img_pil: Image.Image,
    img_np: np.ndarray,
    prompt: str,
) -> tuple[np.ndarray, list[int], list[int]]:
    print(f"\n=== {prompt} ===")
    print("Left-click on the player, then Enter")
    pos, neg = pick_pos_neg_points_qt(img_np, title=prompt)
    if not pos:
        raise RuntimeError(f"{prompt}: no click provided.")
    mask = sam2_predict_masks(model, processor, device, img_pil, pos, neg)
    return mask, pos, neg

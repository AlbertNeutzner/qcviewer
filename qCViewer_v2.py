"""
QC Viewer – Segmentation Quality Control Tool

Supports label TIFs in two formats:
  • RGB colour  (H×W×3 uint8)  – displayed as-is
  • Integer IDs (H×W uint16/32) – mapped to random colours

Layout:
  Left:  Raw (top) + Label (bottom), stacked, with synced crosshair
  Right: Zoom panels for Raw and Label

Usage:
    python qc_viewer.py

Dependencies:
    pip install PyQt5 tifffile numpy
"""

import sys
import os
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QFileDialog, QGroupBox,
    QSpinBox, QFormLayout, QStatusBar, QSizePolicy, QSplitter
)
from PyQt5.QtCore import Qt, pyqtSignal, QPoint
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QCursor

try:
    import tifffile
except ImportError:
    print("Please install tifffile:  pip install tifffile")
    sys.exit(1)


# ── helpers ───────────────────────────────────────────────────────────────────

def numpy_to_qimage(arr: np.ndarray) -> QImage:
    """Convert H×W (grayscale) or H×W×3 array to QImage RGB888."""
    if arr.ndim == 2:
        mn, mx = float(arr.min()), float(arr.max())
        norm = ((arr.astype(np.float32) - mn) / (mx - mn) * 255).astype(np.uint8) \
               if mx > mn else np.zeros(arr.shape, dtype=np.uint8)
        arr_rgb = np.stack([norm, norm, norm], axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 3:
        arr_rgb = arr.astype(np.uint8)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr_rgb = arr[:, :, :3].astype(np.uint8)   # drop alpha
    else:
        raise ValueError(f"Unsupported array shape: {arr.shape}")

    arr_rgb = np.ascontiguousarray(arr_rgb)
    h, w, _ = arr_rgb.shape
    return QImage(arr_rgb.data, w, h, w * 3, QImage.Format_RGB888).copy()


def label_to_color(label_arr: np.ndarray) -> np.ndarray:
    """
    Map label array to RGB.
    - H×W×3 uint8  → return as-is (already RGB)
    - H×W integer  → assign deterministic colours per ID
    """
    if label_arr.ndim == 3:
        return label_arr.astype(np.uint8)   # already RGB

    ids = label_arr.astype(np.int64)
    max_id = int(ids.max())
    if max_id < 1:
        return np.zeros((*ids.shape, 3), dtype=np.uint8)
    rng = np.random.default_rng(42)
    colours = rng.integers(80, 256, size=(max_id + 1, 3), dtype=np.uint8)
    colours[0] = [0, 0, 0]
    return colours[ids]


def load_tif(path: str) -> np.ndarray:
    arr = tifffile.imread(path)
    # Squeeze leading size-1 dimensions
    while arr.ndim > 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[0] < arr.shape[1]:
        arr = np.moveaxis(arr, 0, -1)  # CxHxW → HxWxC
    return arr


# ── image view with crosshair ─────────────────────────────────────────────────

class ImageView(QLabel):
    mouse_moved = pyqtSignal(float, float)   # fractional image coords 0..1

    def __init__(self):
        super().__init__()
        self.setMouseTracking(True)
        self.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.setStyleSheet("background: #1a1a1a;")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._qimage = None
        self._fx = -1.0
        self._fy = -1.0

    def set_image(self, qimage: QImage):
        self._qimage = qimage
        self._rescale()

    def set_cursor_frac(self, fx: float, fy: float):
        self._fx = fx
        self._fy = fy
        self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self):
        if self._qimage is None:
            return
        pix = QPixmap.fromImage(self._qimage)
        self.setPixmap(pix.scaled(self.width() or pix.width(),
                                   self.height() or pix.height(),
                                   Qt.KeepAspectRatio,
                                   Qt.SmoothTransformation))

    def _pix_size(self):
        pix = self.pixmap()
        return (pix.width(), pix.height()) if pix else (0, 0)

    def mouseMoveEvent(self, event):
        pw, ph = self._pix_size()
        if pw > 0 and ph > 0:
            fx = max(0.0, min(1.0, event.x() / pw))
            fy = max(0.0, min(1.0, event.y() / ph))
            self._fx, self._fy = fx, fy
            self.mouse_moved.emit(fx, fy)
        super().mouseMoveEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._fx < 0 or self.pixmap() is None:
            return
        pw, ph = self._pix_size()
        cx, cy = int(self._fx * pw), int(self._fy * ph)
        p = QPainter(self)
        # dark shadow for contrast
        p.setPen(QPen(QColor(0, 0, 0, 140), 3))
        p.drawLine(cx, 0, cx, ph)
        p.drawLine(0, cy, pw, cy)
        # bright green crosshair
        p.setPen(QPen(QColor(0, 255, 100), 1, Qt.DashLine))
        p.drawLine(cx, 0, cx, ph)
        p.drawLine(0, cy, pw, cy)
        # yellow dot
        p.setPen(QPen(QColor(255, 220, 0), 1))
        p.drawEllipse(QPoint(cx, cy), 6, 6)
        p.end()


# ── zoom panel ────────────────────────────────────────────────────────────────

class ZoomLabel(QLabel):
    """Resizable label that rescales its crop image and paints a centre crosshair."""

    def __init__(self, text=""):
        super().__init__(text)
        self._source = None   # stored QImage, rescaled on every resize

    def set_image(self, qi: QImage):
        self._source = qi
        self._rescale()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self):
        if self._source is None:
            return
        self.setPixmap(
            QPixmap.fromImage(self._source).scaled(
                self.width(), self.height(),
                Qt.KeepAspectRatio, Qt.FastTransformation))

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._source is None or self.pixmap() is None:
            return
        cx, cy = self.width() // 2, self.height() // 2
        p = QPainter(self)
        p.setPen(QPen(QColor(0, 0, 0, 140), 3))
        p.drawLine(cx, 0, cx, self.height())
        p.drawLine(0, cy, self.width(), cy)
        p.setPen(QPen(QColor(0, 255, 100), 1, Qt.DashLine))
        p.drawLine(cx, 0, cx, self.height())
        p.drawLine(0, cy, self.width(), cy)
        p.setPen(QPen(QColor(255, 220, 0), 1))
        p.drawEllipse(QPoint(cx, cy), 6, 6)
        p.end()


class ZoomPanel(QWidget):
    def __init__(self, title: str):
        super().__init__()
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        box = QGroupBox(title)
        box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        inner = QVBoxLayout(box)
        inner.setContentsMargins(4, 4, 4, 4)
        self.lbl = ZoomLabel("— hover over image —")
        self.lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.lbl.setAlignment(Qt.AlignCenter)
        self.lbl.setStyleSheet("background: #111; color: #555; font-size: 11px;")
        inner.addWidget(self.lbl)
        layout.addWidget(box)

    def update_crop(self, arr: np.ndarray, cx: int, cy: int,
                    half: int, is_label: bool):
        h, w = arr.shape[:2]
        crop = arr[max(0, cy-half):min(h, cy+half),
                   max(0, cx-half):min(w, cx+half)]
        if crop.size == 0:
            return
        crop_rgb = label_to_color(crop) if is_label else crop
        qi = numpy_to_qimage(crop_rgb)
        if qi.isNull():
            return
        self.lbl.set_image(qi)


# ── histogram widget ──────────────────────────────────────────────────────────

class HistogramWidget(QWidget):
    """Paints a bar histogram; grayscale = green, RGB = R/G/B overlaid."""

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(60)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background: #111;")
        self._channels = []   # list of (counts ndarray, QColor)

    def set_array(self, arr: np.ndarray):
        self._channels = []
        n_bins = 256
        if arr.ndim == 2:
            counts, _ = np.histogram(arr.ravel(), bins=n_bins)
            self._channels = [(counts, QColor(80, 220, 80, 200))]
        elif arr.ndim == 3:
            palette = [QColor(220, 80, 80, 160),
                       QColor(80, 220, 80, 160),
                       QColor(80, 140, 220, 160)]
            for c in range(min(arr.shape[2], 3)):
                counts, _ = np.histogram(arr[:, :, c].ravel(), bins=n_bins)
                self._channels.append((counts, palette[c]))
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(0x11, 0x11, 0x11))
        if not self._channels:
            p.end()
            return
        w, h = self.width(), self.height()
        max_count = max(int(c.max()) for c, _ in self._channels)
        if max_count == 0:
            p.end()
            return
        n = len(self._channels[0][0])
        bar_w = w / n
        p.setPen(Qt.NoPen)
        for counts, color in self._channels:
            p.setBrush(color)
            for i, cnt in enumerate(counts):
                bh = int(cnt / max_count * h)
                if bh == 0:
                    continue
                p.drawRect(int(i * bar_w), h - bh, max(1, int(bar_w) + 1), bh)
        p.end()


# ── stats panel ────────────────────────────────────────────────────────────────

class StatsPanel(QGroupBox):
    """Shows global statistics and a histogram for the original image."""

    def __init__(self):
        super().__init__("Stats – Original")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 8, 6, 6)
        layout.setSpacing(4)
        self._lbl = QLabel("— load an image —")
        self._lbl.setStyleSheet(
            "color: #aaa; font-size: 13px; font-family: monospace;")
        self._lbl.setWordWrap(True)
        self._hist = HistogramWidget()
        layout.addWidget(self._lbl)
        layout.addWidget(self._hist)

    def update_stats(self, arr: np.ndarray):
        lines = []
        if arr.ndim == 2:
            f = arr.astype(np.float64)
            lines.append(
                f"max={f.max():.0f}   mean={f.mean():.1f}   SD={f.std():.1f}")
        else:
            for c, ch in enumerate(["R", "G", "B"][:arr.shape[2]]):
                f = arr[:, :, c].astype(np.float64)
                lines.append(
                    f"{ch}: max={f.max():.0f}  mean={f.mean():.1f}  SD={f.std():.1f}")
        self._lbl.setText("\n".join(lines))
        self._hist.set_array(arr)


# ── main window ───────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QC Viewer – Segmentation Inspector")
        self.resize(1500, 900)
        self._raw = None
        self._label = None
        self._label_rgb = None
        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(0)
        main_splitter = QSplitter(Qt.Horizontal)

        # ── LEFT ─────────────────────────────────────────────────────────────
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setSpacing(4)

        tb = QHBoxLayout()
        btn_raw = QPushButton("📂  Open Raw TIF …");  btn_raw.clicked.connect(self._open_raw)
        btn_lbl = QPushButton("📂  Open Label TIF …"); btn_lbl.clicked.connect(self._open_label)
        self._path_raw = QLabel("—")
        self._path_lbl = QLabel("—")
        for l in (self._path_raw, self._path_lbl):
            l.setStyleSheet("color:#888; font-size:11px;")

        self._zoom_spin = QSpinBox()
        self._zoom_spin.setRange(10, 600)
        self._zoom_spin.setValue(60)
        self._zoom_spin.setSuffix(" px")
        form = QFormLayout()
        form.addRow("Zoom radius:", self._zoom_spin)

        tb.addWidget(btn_raw);  tb.addWidget(self._path_raw)
        tb.addSpacing(12)
        tb.addWidget(btn_lbl);  tb.addWidget(self._path_lbl)
        tb.addStretch(); tb.addLayout(form)
        lv.addLayout(tb)

        splitter = QSplitter(Qt.Vertical)

        raw_box = QGroupBox("Original (Raw)")
        QVBoxLayout(raw_box).addWidget(self._make_view(True))
        splitter.addWidget(raw_box)

        lbl_box = QGroupBox("Label / Segmentation")
        QVBoxLayout(lbl_box).addWidget(self._make_view(False))
        splitter.addWidget(lbl_box)

        splitter.setSizes([440, 440])
        lv.addWidget(splitter, stretch=1)

        # ── RIGHT ─────────────────────────────────────────────────────────────
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setSpacing(8)
        self._zpanel_raw = ZoomPanel("Zoom – Raw")
        self._zpanel_lbl = ZoomPanel("Zoom – Label")
        self._stats_panel = StatsPanel()
        rv.addWidget(self._zpanel_raw)
        rv.addWidget(self._zpanel_lbl)
        rv.addWidget(self._stats_panel)

        main_splitter.addWidget(left)
        main_splitter.addWidget(right)
        main_splitter.setSizes([1190, 310])
        main_splitter.setChildrenCollapsible(False)
        root.addWidget(main_splitter)

        self.setStatusBar(QStatusBar())

    def _make_view(self, is_raw: bool) -> ImageView:
        v = ImageView()
        v.setCursor(QCursor(Qt.CrossCursor))
        v.mouse_moved.connect(self._on_mouse_frac)
        if is_raw:
            self._view_raw = v
        else:
            self._view_lbl = v
        return v

    # ── loading ───────────────────────────────────────────────────────────────

    def _open_raw(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Raw TIF", "", "TIFF files (*.tif *.tiff);;All files (*)")
        if not path:
            return
        try:
            self._raw = load_tif(path)
            self._view_raw.set_image(numpy_to_qimage(self._raw))
            self._path_raw.setText(os.path.basename(path))
            self._stats_panel.update_stats(self._raw)
            self.statusBar().showMessage(
                f"Raw: {self._raw.shape}  dtype={self._raw.dtype}")
        except Exception as e:
            self.statusBar().showMessage(f"Error loading raw: {e}")

    def _open_label(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Label TIF", "", "TIFF files (*.tif *.tiff);;All files (*)")
        if not path:
            return
        try:
            self._label = load_tif(path)
            self._label_rgb = label_to_color(self._label)
            self._view_lbl.set_image(numpy_to_qimage(self._label_rgb))
            self._path_lbl.setText(os.path.basename(path))
            info = (f"Label: {self._label.shape}  dtype={self._label.dtype}")
            if self._label.ndim == 2:
                info += f"  objects={int(self._label.max())}"
            else:
                flat = self._label.reshape(-1, self._label.shape[-1])
                n_colors = len(np.unique(flat, axis=0))
                info += f"  unique colours={n_colors}"
            self.statusBar().showMessage(info)
        except Exception as e:
            self.statusBar().showMessage(f"Error loading label: {e}")

    # ── mouse ─────────────────────────────────────────────────────────────────

    def _on_mouse_frac(self, fx: float, fy: float):
        self._view_raw.set_cursor_frac(fx, fy)
        self._view_lbl.set_cursor_frac(fx, fy)

        half = self._zoom_spin.value()
        status = ""

        if self._raw is not None:
            h, w = self._raw.shape[:2]
            ix, iy = min(int(fx * w), w-1), min(int(fy * h), h-1)
            self._zpanel_raw.update_crop(self._raw, ix, iy, half, is_label=False)
            pv = self._raw[iy, ix]
            raw_str = str(int(pv)) if pv.ndim == 0 else str(tuple(int(v) for v in pv))
            status = f"x={ix}  y={iy}  original={raw_str}"

        if self._label is not None:
            lh, lw = self._label.shape[:2]
            lx, ly = min(int(fx * lw), lw-1), min(int(fy * lh), lh-1)
            self._zpanel_lbl.update_crop(self._label, lx, ly, half, is_label=True)
            if self._label.ndim == 2:
                status += f"  mask={int(self._label[ly, lx])}"
            else:
                status += f"  mask={tuple(int(v) for v in self._label[ly, lx])}"

        self.statusBar().showMessage(status)


# ── entry ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
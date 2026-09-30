"""Shared figure style for conference papers (ACM/IEEE two-column).

Single-column width 3.4 in, double-column 7.0 in; serif text at 8 pt; Okabe-Ito colour-blind-safe
palette with a fixed colour and marker per config so every figure reads the same way.
Each figure is saved as PNG (300 dpi) and PDF (vector, for LaTeX).
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SINGLE = 3.4
DOUBLE = 7.0

CONFIG_ORDER = ["wasm-1t", "wasm-4t", "wasm-8t", "webgl", "webgpu", "webgpu-native", "webgpu-gpuio", "webgpu-capture"]
CONFIG_STYLE = {
    "wasm-1t": dict(color="#BBBBBB", marker="v", label="WASM 1T"),
    "wasm-4t": dict(color="#777777", marker="^", label="WASM 4T"),
    "wasm-8t": dict(color="#000000", marker="s", label="WASM 8T"),
    "webgl": dict(color="#E69F00", marker="D", label="WebGL"),
    "webgpu": dict(color="#0072B2", marker="o", label="WebGPU (JSEP)"),
    "webgpu-native": dict(color="#CC79A7", marker="P", label="WebGPU (native EP)"),
    "webgpu-gpuio": dict(color="#56B4E9", marker="X", label="WebGPU GPU-I/O"),
    "webgpu-capture": dict(color="#009E73", marker="*", label="WebGPU + graph capture"),
}

MODEL_SHORT = {
    "mobilenetv2-7": "MobileNetV2",
    "mobilenetv3_large_100_Opset17": "MobileNetV3-L",
    "alexnet_Opset16": "AlexNet",
    "resnet50-v2": "ResNet-50",
    "adv_inception_v3_Opset17": "Inception-v3",
    "efficientnet_b5_Opset17": "EfficientNet-B5",
    "vit_large_patch16_224_in21k_Opset18": "ViT-L/16",
    "yolov2-coco-9": "YOLOv2",
    "deeplabv3p-resnet50-human": "DeepLabV3+",
    "mobilebert_Opset17": "MobileBERT",
    "bert_Opset17": "BERT-base",
    "gpt2lmhead_Opset18": "GPT-2",
}


def short(model):
    return MODEL_SHORT.get(model, model)


def label(config):
    return CONFIG_STYLE.get(config, {}).get("label", config)


def color(config):
    return CONFIG_STYLE.get(config, {}).get("color", "#999999")


def marker(config):
    return CONFIG_STYLE.get(config, {}).get("marker", "o")


def apply():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _plain(v, _pos=None):
    if v <= 0:
        return ""
    return f"{v:g}" if 1e-3 <= v < 1e5 else f"{v:.0e}"


def tidy_log_axes(fig):
    """Plain-number tick labels (0.5, 2, 10, 100) on every log axis instead of 2x10^0 style."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    for ax in fig.axes:
        for axis, scale in ((ax.xaxis, ax.get_xscale()), (ax.yaxis, ax.get_yscale())):
            if scale != "log":
                continue
            lo, hi = sorted(ax.get_ylim() if axis is ax.yaxis else ax.get_xlim())
            decades = (hi / lo) if lo > 0 else 1e9
            axis.set_major_locator(LogLocator(base=10, subs=(1.0,) if decades > 30 else (1.0, 2.0, 5.0)))
            axis.set_major_formatter(FuncFormatter(_plain))
            axis.set_minor_formatter(NullFormatter())


def save(fig, out_dir, name):
    tidy_log_axes(fig)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png", dpi=300)
    fig.savefig(out_dir / f"{name}.pdf")
    plt.close(fig)
    return [f"{name}.png", f"{name}.pdf"]

#!/usr/bin/env python3
"""Build the offline HTML documentation site under docs/html/.

The site is intentionally dependency-free: static HTML, one stylesheet, one
small script, and inline SVG diagrams.  Regenerate it with:

    python3 scripts/docs/build_site.py

All numbers come from docs/PERFORMANCE_VS_PYTORCH.md (appendix C) and the
2.0 GHz measurements recorded in docs/OPTIMIZATION_PLAN_2026-11-18.md.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "html"

# ---------------------------------------------------------------------------
# data (2.0 GHz, node0)
# ---------------------------------------------------------------------------
# shape: gflop, mlir ms, mlir GF, torch t1 GF, ratio
SINGLE = {
    "custom_0": (4.25, 33.3, 127.7, 87.7, 1.46),
    "custom_1": (67.95, 660.9, 102.8, 46.5, 2.21),
    "custom_2": (4.25, 37.5, 113.2, 75.9, 1.49),
    "custom_3": (16.99, 218.8, 77.6, 38.5, 2.02),
    "custom_4": (67.95, 653.0, 104.1, 73.3, 1.42),
    "custom_5": (4.25, 47.4, 89.5, 62.3, 1.44),
    "custom_6": (33.97, 412.2, 82.4, 58.0, 1.42),
    "custom_7": (1.06, 15.1, 70.5, 84.4, 0.83),
    "custom_8": (16.99, 168.4, 100.9, 65.9, 1.53),
    "custom_9": (16.99, 134.0, 126.8, 72.9, 1.74),
    "custom_10": (8.49, 76.0, 111.8, 50.0, 2.24),
}

# shape: serial, t1, t8, t16, t32, speedup, best GF, width
M3 = {
    "custom_0": (34.4, 34.6, 18.2, 16.2, 15.8, 2.18, 269, 10),
    "custom_1": (620.9, 665.3, 259.4, 200.8, 201.3, 3.09, 338, 10),
    "custom_2": (37.1, 41.0, 22.5, 19.2, 18.4, 2.02, 231, 10),
    "custom_3": (232.5, 226.3, 138.5, 139.5, 138.6, 1.68, 123, 5),
    "custom_4": (649.6, 676.4, 272.2, 210.7, 214.2, 3.08, 323, 10),
    "custom_5": (46.6, 46.7, 21.1, 18.6, 17.0, 2.74, 250, 20),
    "custom_6": (407.1, 422.8, 180.6, 148.3, 147.5, 2.76, 230, 10),
    "custom_7": (14.4, 15.9, 11.0, 10.8, 13.8, 1.33, 98, 5),
    "custom_8": (168.0, 172.9, 72.4, 57.6, 61.7, 2.92, 295, 10),
    "custom_9": (130.9, 141.1, 66.7, 56.4, 51.2, 2.56, 332, 10),
    "custom_10": (75.0, 80.3, 37.9, 31.3, 33.1, 2.40, 271, 10),
}

# shape: serial, t1, t8, t16, t32, speedup, best GF, nwin, width
M4 = {
    "custom_0": (31.5, 34.1, 16.7, 15.8, 14.4, 2.19, 295, 80, 20),
    "custom_1": (622.2, 668.1, 264.6, 201.4, 198.0, 3.14, 343, 160, 10),
    "custom_2": (34.9, 40.3, 22.0, 18.1, 18.4, 1.93, 231, 160, 10),
    "custom_3": (229.0, 249.7, 147.3, 131.5, 128.0, 1.79, 133, 40, 10),
    "custom_4": (618.3, 666.4, 242.5, 211.3, 175.9, 3.52, 386, 80, 20),
    "custom_5": (45.1, 48.5, 21.2, 18.8, 17.5, 2.58, 243, 160, 20),
    "custom_6": (426.5, 458.9, 170.4, 151.3, 131.6, 3.24, 258, 80, 20),
    "custom_7": (15.3, 17.5, 11.5, 10.4, 10.2, 1.50, 104, 40, 10),
    "custom_8": (169.2, 175.8, 65.8, 56.7, 48.9, 3.46, 347, 80, 20),
    "custom_9": (130.8, 142.6, 69.8, 51.8, 51.6, 2.53, 329, 160, 10),
    "custom_10": (74.0, 80.0, 34.0, 31.6, 26.9, 2.75, 316, 80, 20),
}

SHAPES = list(SINGLE)
BEST_M3 = {s: min(M3[s][1:5]) for s in SHAPES}
BEST_M4 = {s: min(M4[s][1:5]) for s in SHAPES}
GAIN = {s: (BEST_M3[s] - BEST_M4[s]) / BEST_M3[s] * 100 for s in SHAPES}

NAV = [
    ("index.html", "总览"),
    ("principles.html", "原理"),
    ("pipeline.html", "管线"),
    ("timeline.html", "演进"),
    ("multicore.html", "多核"),
    ("performance.html", "性能"),
    ("reproduce.html", "复现"),
]

C_TEAL = "#0f766e"
C_TEAL_B = "#14b8a6"
C_INDIGO = "#4338ca"
C_INDIGO_B = "#6366f1"
C_AMBER = "#b45309"
C_AMBER_B = "#f59e0b"
C_RED = "#b91c1c"
C_SLATE = "#475569"
C_LINE = "#cbd5e1"
C_INK = "#0b1220"


def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


def head(title, desc, active):
    nav_items = []
    for href, label in NAV:
        cls = ' class="active"' if href == active else ""
        nav_items.append(f'<a href="{href}"{cls}>{label}</a>')
    nav = "".join(nav_items)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)} · SConvTransform</title>
<meta name="description" content="{esc(desc)}">
<link rel="stylesheet" href="assets/site.css">
</head>
<body>
<header class="site-header">
  <div class="wrap">
    <a class="brand" href="index.html" aria-label="SConvTransform 文档首页">
      <svg viewBox="0 0 40 40" aria-hidden="true"><rect x="3" y="3" width="34" height="34" rx="9" fill="#0b1220"/><rect x="10" y="10" width="20" height="20" rx="5" fill="none" stroke="#14b8a6" stroke-width="2.4"/><rect x="16" y="16" width="8" height="8" rx="2" fill="#6366f1"/></svg>
      <span>SConvTransform<small>Kunpeng aarch64 · MLIR</small></span>
    </a>
    <nav class="nav" aria-label="主导航">{nav}</nav>
  </div>
</header>
"""


FOOTER = """<footer class="footer">
  <div class="wrap">
    <div><b>SConvTransform</b> · 离线 HTML 文档站 · 2.0 GHz / 单 NUMA 口径</div>
    <div>数据源：<code>docs/PERFORMANCE_VS_PYTORCH.md</code> 附录 C ·
      生成：<code>scripts/docs/build_site.py</code></div>
  </div>
</footer>
<script src="assets/site.js"></script>
</body>
</html>
"""


def page(title, desc, active, body, compact_hero=None):
    if compact_hero is None:
        return head(title, desc, active) + body + FOOTER
    h1, lead, eyebrow = compact_hero
    hero = f"""<section class="hero compact"><div class="wrap">
  <div>
    <span class="eyebrow">{eyebrow}</span>
    <h1>{h1}</h1>
    <p class="lead">{lead}</p>
  </div>
</div></section>"""
    return head(title, desc, active) + hero + body + FOOTER


# ---------------------------------------------------------------------------
# SVG helpers
# ---------------------------------------------------------------------------
def svg_open(w, h, label, dark=False):
    cls = ' class="dark"' if dark else ""
    return (f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{esc(label)}" '
            f'preserveAspectRatio="xMidYMid meet"{cls}>')


def arrow_defs(pid, color="#94a3b8"):
    return (f'<defs><marker id="{pid}" viewBox="0 0 10 10" refX="8" refY="5" '
            f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{color}"/></marker></defs>')


def box(x, y, w, h, title, sub="", fill="#ffffff", stroke=C_LINE,
        tcol=C_INK, scol="#64748b", rx=12, fs=14, sfs=11.5, lw=1.4):
    s = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
         f'fill="{fill}" stroke="{stroke}" stroke-width="{lw}"/>']
    if sub:
        s.append(f'<text x="{x + w/2}" y="{y + h/2 - 3}" text-anchor="middle" '
                 f'font-size="{fs}" font-weight="700" fill="{tcol}">{title}</text>')
        s.append(f'<text x="{x + w/2}" y="{y + h/2 + 15}" text-anchor="middle" '
                 f'font-size="{sfs}" fill="{scol}">{sub}</text>')
    else:
        s.append(f'<text x="{x + w/2}" y="{y + h/2 + 5}" text-anchor="middle" '
                 f'font-size="{fs}" font-weight="700" fill="{tcol}">{title}</text>')
    return "".join(s)


def txt(x, y, s, size=12, color=C_SLATE, weight=500, anchor="start", mono=False):
    fam = ' font-family="JetBrains Mono, Consolas, monospace"' if mono else ""
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" '
            f'font-weight="{weight}" text-anchor="{anchor}"{fam}>{s}</text>')


def path_arrow(x1, y1, x2, y2, pid, color="#94a3b8", width=1.8, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<path d="M {x1} {y1} L {x2} {y2}" fill="none" stroke="{color}" '
            f'stroke-width="{width}"{d} marker-end="url(#{pid})"/>')


def legend(x, y, items, size=11.5):
    out = []
    for i, (label, color) in enumerate(items):
        yy = y + i * 19
        out.append(f'<rect x="{x}" y="{yy - 8}" width="11" height="11" rx="2.5" fill="{color}"/>')
        out.append(txt(x + 17, yy + 1, label, size, "#475569", 600))
    return "".join(out)


# ---------------------------------------------------------------------------
# diagrams
# ---------------------------------------------------------------------------
def svg_hero_pipeline():
    w, h = 620, 330
    s = [svg_open(w, h, "SConv 三阶段流水线：输入打包、GEMM、还原"),
         arrow_defs("heroArr", "#5eead4")]
    s.append(f'<rect x="0" y="0" width="{w}" height="{h}" rx="16" fill="#0a1627"/>')
    s.append(txt(26, 34, "SConv pipeline", 13, "#5eead4", 800))
    s.append(txt(594, 34, "3 stages · 1 GEMM call per tile", 11, "#7c93b3", 500, "end"))
    # input
    s.append(box(24, 58, 92, 58, "Input", "NCHW", "#0f2038", "#1e3a5f", "#dbeafe", "#8fa6c4"))
    s.append(box(154, 58, 150, 58, "Pack (im2col)", "9× write amp", "#123b3a", "#1f6f68", "#ccfbf1", "#7dd3c0"))
    s.append(box(342, 58, 120, 58, "GEMM", "kblas", "#1d2450", "#4f46e5", "#e0e7ff", "#a5b4fc"))
    s.append(box(500, 58, 96, 58, "Unpack", "NCHW", "#2a2140", "#7c3aed", "#ede9fe", "#c4b5fd"))
    s.append(path_arrow(116, 87, 152, 87, "heroArr", "#5eead4"))
    s.append(path_arrow(304, 87, 340, 87, "heroArr", "#5eead4"))
    s.append(path_arrow(462, 87, 498, 87, "heroArr", "#5eead4"))
    # packed A / B
    s.append(box(154, 152, 150, 46, "packed A", "[Nc·Fh·Fw, Nwin]", "#0f2038", "#1e3a5f", "#dbeafe", "#8fa6c4", fs=12, sfs=10.5))
    s.append(box(342, 152, 120, 46, "packed B", "[K, Nf]", "#0f2038", "#1e3a5f", "#dbeafe", "#8fa6c4", fs=12, sfs=10.5))
    s.append(path_arrow(229, 116, 229, 150, "heroArr", "#5eead4", 1.4, "4 4"))
    s.append(path_arrow(402, 116, 402, 150, "heroArr", "#5eead4", 1.4, "4 4"))
    # output
    s.append(box(24, 246, 572, 54, "Output tile  =  A[Nwin, K] × B[K, Nf]", "Nwin × Nf × (Nc·Fh·Fw) per microkernel call", "#0a1b2e", "#1e3a5f", "#e2e8f0", "#8fa6c4", fs=13.5, sfs=11))
    # notes
    s.append(txt(24, 326, "pack 贴单核带宽 roofline；GEMM 贴微内核峰值；unpack 非瓶颈（5–13%）。", 11.5, "#93a9c6", 500))
    s.append("</svg>")
    return "".join(s)


def svg_conv_flow():
    w, h = 900, 250
    s = [svg_open(w, h, "卷积三阶段与 roofline"), arrow_defs("flowArr", "#94a3b8")]
    stages = [
        (30, "输入 tile", "Nc × Fh × Fw × Nwin", "#f0fdfa", C_TEAL_B, C_TEAL),
        (250, "Pack / im2col", "行分段 memcpy · 9× 写放大", "#eef2ff", C_INDIGO_B, C_INDIGO),
        (470, "GEMM", "m=Nwin · n=Nf · k=Nc·Fh·Fw", "#fff7ed", C_AMBER_B, C_AMBER),
        (690, "输出 tile", "Nwin × Nf → NCHW", "#f5f3ff", "#8b5cf6", "#6d28d9"),
    ]
    for x, t, sub, fill, stroke, col in stages:
        s.append(box(x, 44, 180, 76, t, sub, fill, stroke, col, "#475569", fs=14, sfs=10.5))
    for x in (210, 430, 650):
        s.append(path_arrow(x, 82, x + 38, 82, "flowArr", "#94a3b8"))
    # roofline strip
    s.append(f'<rect x="30" y="156" width="840" height="62" rx="12" fill="#f8fafc" stroke="{C_LINE}"/>')
    s.append(txt(46, 180, "各自的 roofline", 12, C_INK, 800))
    s.append(txt(46, 201, "pack：单核存储带宽 ~13 GB/s（三种写法同速）", 11.5, C_SLATE))
    s.append(txt(330, 201, "GEMM：微内核峰值 0.7–2.3 TFLOPS（t32）", 11.5, C_SLATE))
    s.append(txt(650, 201, "unpack：5–13% 总时间", 11.5, C_SLATE))
    s.append("</svg>")
    return "".join(s)


def svg_loopnest():
    w, h = 900, 430
    s = [svg_open(w, h, "SConv 循环嵌套与并行/归约维度")]
    s.append(f'<rect x="16" y="18" width="868" height="396" rx="14" fill="#f8fafc" stroke="{C_LINE}"/>')
    levels = [
        (34, 380, "batch  N", "串行（可入链）", "#eef2ff", C_INDIGO_B, C_INDIGO),
        (58, 332, "Nc  归约维", "不可并行 · 累加 C", "#fef2f2", "#ef4444", C_RED),
        (82, 284, "窗口组  Nwin × K3", "含共享 filter 打包", "#fffbeb", C_AMBER_B, C_AMBER),
        (106, 236, "filter tile  Nf × K2", "含 filter 打包", "#fffbeb", C_AMBER_B, C_AMBER),
        (130, 188, "窗口 tile  Nwin", "★ 并行维度", "#f0fdfa", C_TEAL_B, C_TEAL),
        (154, 140, "filter 子块  Nf", "微内核调用", "#f5f3ff", "#8b5cf6", "#6d28d9"),
        (178, 92, "microkernel", "sgemm", "#0b1526", "#334155", "#e2e8f0"),
    ]
    for x, ww, title, sub, fill, stroke, col in levels:
        tcol = "#e2e8f0" if fill == "#0b1526" else col
        scol = "#94a3b8" if fill == "#0b1526" else "#64748b"
        s.append(box(x, 34, ww, 46, title, sub, fill, stroke, tcol, scol, fs=13, sfs=10.5))
    # legend
    s.append(legend(560, 52, [("可并行：写不相交 C", C_TEAL_B),
                              ("归约：不能并行", "#ef4444"),
                              ("共享 scratch：不入链", C_AMBER_B)], 11))
    s.append(txt(34, 402, "M3/M4 只标记最内层窗口 tile 循环；M4-c 的 Nwin 选择会改变它的 trip count（并行宽度）。", 11.5, C_SLATE))
    s.append("</svg>")
    return "".join(s)


def svg_tile_params():
    w, h = 900, 250
    s = [svg_open(w, h, "tile 参数如何映射到 GEMM")]
    s.append(box(30, 44, 190, 100, "Nwin", "空间窗口 tile", "#f0fdfa", C_TEAL_B, C_TEAL, fs=20, sfs=11))
    s.append(box(30, 158, 190, 62, "Nf", "filter tile", "#eef2ff", C_INDIGO_B, C_INDIGO, fs=18, sfs=10.5))
    s.append(box(260, 44, 190, 100, "Nc", "输入通道 tile", "#fff7ed", C_AMBER_B, C_AMBER, fs=20, sfs=11))
    s.append(box(260, 158, 190, 62, "Fh × Fw = 9", "3×3 卷积", "#f8fafc", C_LINE, C_SLATE, fs=16, sfs=10.5))
    s.append(path_arrow(220, 94, 256, 94, "tpArr", C_TEAL_B, 2))
    s.append(path_arrow(220, 189, 256, 189, "tpArr", C_TEAL_B, 2))
    s.append(arrow_defs("tpArr", C_TEAL_B))
    s.append(box(520, 44, 330, 84, "GEMM  m = Nwin", "一次调用处理 Nwin 个输出位置", "#f8fafc", C_LINE, C_INK, C_SLATE, fs=15, sfs=10.5))
    s.append(box(520, 144, 330, 76, "GEMM  n = Nf · k = Nc·Fh·Fw", "k 决定每次调用的“有效工作量”", "#f8fafc", C_LINE, C_INK, C_SLATE, fs=14, sfs=10.5))
    s.append(path_arrow(450, 86, 516, 86, "tpArr", C_INDIGO_B, 2))
    s.append(path_arrow(450, 189, 516, 182, "tpArr", C_INDIGO_B, 2))
    s.append(txt(30, 240, "单线程要“大 tile 少调用”；多核要“小 tile 多任务”。M4-c 用调度保护在两者之间取安全点。", 11.5, C_SLATE))
    s.append("</svg>")
    return "".join(s)


def svg_transform_pipeline():
    w, h = 900, 320
    s = [svg_open(w, h, "MLIR 优化管线：transform 与 pass"), arrow_defs("tfArr", "#94a3b8")]
    row1 = [
        (20, "linalg.conv", "输入 IR"),
        (200, "transform.structured.sconv", "tiling / packing / marker"),
        (420, "transform.lower.to_pack", "pack_input_tile"),
        (640, "transform.lower.to_blas", "sgemm 微内核"),
    ]
    for x, t, sub in row1:
        s.append(box(x, 34, 220, 70, t, sub, "#ffffff", C_LINE, C_INK, "#64748b", fs=12.5, sfs=10.5))
    for x in (240, 460, 680):
        s.append(path_arrow(x, 69, x + 38, 69, "tfArr"))
    s.append(txt(20, 132, "sconv-opt passes", 12, C_TEAL, 800))
    row2 = [
        (20, "-sconv-parallelize", "窗口 tile → scf.parallel", "#f0fdfa", C_TEAL_B),
        (245, "-sconv-lower-parallel", "OpenMP + alloca_scope 清理", "#eef2ff", C_INDIGO_B),
        (470, "convert-* / LLVM", "可执行 LLVM IR", "#fff7ed", C_AMBER_B),
        (695, "mlir-cpu-runner", "OMP_NUM_THREADS", "#f5f3ff", "#8b5cf6"),
    ]
    for x, t, sub, fill, stroke in row2:
        s.append(box(x, 152, 190, 70, t, sub, fill, stroke, C_INK, "#475569", fs=11.5, sfs=10))
    for x in (210, 435, 660):
        s.append(path_arrow(x, 187, x + 33, 187, "tfArr"))
    s.append(f'<rect x="20" y="256" width="860" height="40" rx="10" fill="#f8fafc" stroke="{C_LINE}"/>')
    s.append(txt(36, 281, "一条命令：sconv-opt -transform=... -sconv-parallelize -sconv-lower-parallel conv.mlir -o conv.llvm.mlir", 11.5, C_SLATE, 600, mono=True))
    s.append("</svg>")
    return "".join(s)


def svg_parallel_chain():
    w, h = 900, 360
    s = [svg_open(w, h, "并行链折叠与 per-thread scratch"), arrow_defs("pcArr", C_TEAL_B)]
    # left: serial loops
    s.append(txt(20, 26, "标记前：窗口 tile 循环在串行链里", 12, C_INK, 800))
    ys = [46, 92, 138, 184]
    names = [("batch", "#eef2ff"), ("Nc（归约）", "#fef2f2"),
             ("窗口组 / filter", "#fffbeb"), ("窗口 tile ★", "#f0fdfa")]
    for y, (n, fill) in zip(ys, names):
        s.append(box(20, y, 250, 34, n, "", fill, C_LINE, C_INK, fs=12))
    s.append(box(20, 230, 250, 56, "body：pack → GEMM", "写 C 的同一列", "#f8fafc", C_LINE, C_SLATE, "#64748b", fs=12, sfs=10.5))
    # right: parallel
    s.append(txt(500, 26, "标记后：折叠成 scf.parallel", 12, C_INK, 800))
    s.append(box(500, 46, 360, 70, "scf.parallel  0 .. Ohw step Nwin", "每个迭代 = 一个完整工作项", "#f0fdfa", C_TEAL_B, C_TEAL, "#475569", fs=13, sfs=10.5))
    s.append(box(500, 140, 360, 60, "per-thread scratch", "[64][Nc·Fh·Fw·Nwin]", "#eef2ff", C_INDIGO_B, C_INDIGO, "#475569", fs=13, sfs=10.5))
    s.append(box(500, 224, 360, 62, "sgemm × (Oc/Nf)", "写不相交 C 列", "#fff7ed", C_AMBER_B, C_AMBER, "#475569", fs=13, sfs=10.5))
    s.append(path_arrow(380, 183, 496, 86, "pcArr", C_TEAL_B, 2.2))
    s.append(path_arrow(680, 116, 680, 138, "pcArr", C_INDIGO_B, 2))
    s.append(path_arrow(680, 200, 680, 222, "pcArr", C_AMBER_B, 2))
    s.append(txt(380, 300, "slot = omp_get_thread_num()", 11, C_SLATE, 600, "middle", mono=True))
    s.append(txt(380, 320, "槽必须 ≥ 实际线程数", 11, C_RED, 600, "middle"))
    s.append("</svg>")
    return "".join(s)


def svg_schedule_tree():
    w, h = 900, 370
    s = [svg_open(w, h, "M4-c 的多核 Nwin 决策与 IS/WS 保护"), arrow_defs("stArr", "#94a3b8")]
    s.append(box(330, 18, 240, 50, "自动选择 Nwin", "", "#0b1526", "#334155", "#e2e8f0", fs=14))
    s.append(box(330, 96, 240, 54, "IC ≤ 64 ?", "薄通道 / 打包受限", "#fff7ed", C_AMBER_B, C_AMBER, "#475569", fs=13, sfs=10.5))
    s.append(path_arrow(450, 68, 450, 94, "stArr"))
    s.append(box(600, 96, 270, 54, "保持串行 tile（c5：160）", "", "#f0fdfa", C_TEAL_B, C_TEAL, fs=12))
    s.append(path_arrow(570, 123, 598, 123, "stArr", C_AMBER_B, 1.8))
    s.append(txt(578, 112, "是", 11, C_AMBER, 700))
    s.append(box(330, 180, 240, 54, "candidate = min(base,80)", "Ohw/cand < 8 → 40", "#eef2ff", C_INDIGO_B, C_INDIGO, "#475569", fs=12))
    s.append(path_arrow(450, 150, 450, 178, "stArr"))
    s.append(txt(458, 170, "否", 11, C_SLATE, 700))
    s.append(box(330, 264, 240, 54, "CSA(base).schd == CSA(cand).schd ?", "", "#f5f3ff", "#8b5cf6", "#5b21b6", fs=11.5))
    s.append(path_arrow(450, 234, 450, 262, "stArr"))
    s.append(box(600, 256, 270, 70, "采用 candidate", "c0/c3/c4/c6/c7/c8/c10", "#f0fdfa", C_TEAL_B, C_TEAL, "#475569", fs=13, sfs=10.5))
    s.append(box(20, 256, 270, 70, "保持 base（160）", "c1/c2/c9：IS → WS 翻转", "#fef2f2", "#ef4444", C_RED, "#475569", fs=13, sfs=10.5))
    s.append(path_arrow(570, 291, 598, 291, "stArr", C_TEAL_B, 1.8))
    s.append(txt(578, 280, "是", 11, C_TEAL, 700))
    s.append(path_arrow(330, 291, 292, 291, "stArr", "#ef4444", 1.8))
    s.append(txt(310, 280, "否", 11, C_RED, 700))
    s.append(txt(450, 352, "Nwin 会改变 k3 / tile_c，进而改变 IS/WS —— 只看任务数会得到“更多但更慢”。", 11.5, C_SLATE, 600, "middle"))
    s.append("</svg>")
    return "".join(s)


# ---------------------------------------------------------------------------
# charts
# ---------------------------------------------------------------------------
def svg_hbars(items, vmax, width=860, row=30, left=112, right=78,
               title="", ref=None, value_fmt="{:.2f}", unit=""):
    """items: [(label, value, kind)] with kind in teal/indigo/amber/red."""
    n = len(items)
    top = 34 if title else 14
    h = top + n * row + 24
    plot_w = width - left - right
    s = [svg_open(width, h, title or "bar chart")]
    if title:
        s.append(txt(left, 20, title, 12, C_INK, 800))
    # grid
    for g in range(5):
        x = left + plot_w * g / 4
        s.append(f'<line x1="{x}" y1="{top - 6}" x2="{x}" y2="{top + n*row}" stroke="#eef2f7" stroke-width="1"/>')
    for i, (label, value, kind) in enumerate(items):
        y = top + i * row
        bw = max(2, plot_w * min(max(value, 0.0), vmax) / vmax)
        color = {"teal": C_TEAL_B, "indigo": C_INDIGO_B,
                 "amber": C_AMBER_B, "red": "#ef4444"}.get(kind, C_INDIGO_B)
        s.append(txt(left - 10, y + 16, label, 11.5, C_SLATE, 600, "end"))
        s.append(f'<rect x="{left}" y="{y + 5}" width="{plot_w}" height="18" rx="5" fill="#f1f5f9"/>')
        s.append(f'<rect x="{left}" y="{y + 5}" width="{bw:.1f}" height="18" rx="5" fill="{color}"/>')
        s.append(txt(left + bw + 8, y + 18, value_fmt.format(value) + unit, 11.5, C_INK, 700))
    if ref is not None:
        x = left + plot_w * ref / vmax
        s.append(f'<line x1="{x}" y1="{top - 6}" x2="{x}" y2="{top + n*row}" stroke="{C_RED}" stroke-width="1.6" stroke-dasharray="5 4"/>')
        s.append(txt(x + 5, top - 9, f"{ref:g}×", 10.5, C_RED, 700))
    s.append("</svg>")
    return "".join(s)


def svg_grouped(items, vals_a, vals_b, vmax, width=880, row=40, left=112,
                right=92, title="", fmt="{:.1f}"):
    n = len(items)
    top = 44 if title else 20
    h = top + n * row + 26
    plot_w = width - left - right
    s = [svg_open(width, h, title or "grouped bar chart")]
    if title:
        s.append(txt(left, 20, title, 12, C_INK, 800))
        s.append(legend(width - 190, 16, [("M3", "#94a3b8"), ("M4-c", C_TEAL_B)], 11))
    for g in range(5):
        x = left + plot_w * g / 4
        s.append(f'<line x1="{x}" y1="{top - 6}" x2="{x}" y2="{top + n*row}" stroke="#eef2f7" stroke-width="1"/>')
    for i, label in enumerate(items):
        y = top + i * row
        b1 = max(2, plot_w * min(vals_a[i], vmax) / vmax)
        b2 = max(2, plot_w * min(vals_b[i], vmax) / vmax)
        s.append(txt(left - 10, y + 21, label, 11.5, C_SLATE, 600, "end"))
        s.append(f'<rect x="{left}" y="{y + 5}" width="{plot_w}" height="11" rx="4" fill="#f1f5f9"/>')
        s.append(f'<rect x="{left}" y="{y + 5}" width="{b1:.1f}" height="11" rx="4" fill="#94a3b8"/>')
        s.append(f'<rect x="{left}" y="{y + 20}" width="{plot_w}" height="11" rx="4" fill="#f1f5f9"/>')
        s.append(f'<rect x="{left}" y="{y + 20}" width="{b2:.1f}" height="11" rx="4" fill="{C_TEAL_B}"/>')
        s.append(txt(left + max(b1, b2) + 8, y + 23, fmt.format(vals_b[i]) + " ms", 10.5, C_INK, 700))
    s.append("</svg>")
    return "".join(s)


def svg_multiline(panels, width=880, height=240):
    """panels: [(title, m3 values, m4 values)]; xs fixed t1,t8,t16,t32."""
    xs = [1, 8, 16, 32]
    n = len(panels)
    pw = (width - 40) / n
    s = [svg_open(width, height, "线程扩展曲线")]
    y0, y1 = 34, height - 34
    for k, (title, a, b) in enumerate(panels):
        x0 = 20 + k * pw
        # panel background
        s.append(f'<rect x="{x0}" y="10" width="{pw - 14}" height="{height - 20}" rx="10" fill="#f8fafc" stroke="{C_LINE}"/>')
        s.append(txt(x0 + 12, 29, title, 12, C_INK, 800))
        vmax = max(a + b) * 1.08
        pw_in = pw - 54
        def px(i):
            return x0 + 30 + pw_in * i / 3
        def py(v):
            return y1 - (y1 - y0) * (v / vmax)
        for g in range(4):
            yy = y0 + (y1 - y0) * g / 3
            s.append(f'<line x1="{x0+26}" y1="{yy}" x2="{x0+pw-22}" y2="{yy}" stroke="#e2e8f0" stroke-width="1"/>')
        for i, xv in enumerate(xs):
            s.append(txt(px(i), height - 17, f"t{xv}", 9.5, "#64748b", 600, "middle"))
        for vals, col in ((a, "#94a3b8"), (b, C_TEAL_B)):
            pts = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(vals))
            s.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.4" stroke-linejoin="round" stroke-linecap="round"/>')
            for i, v in enumerate(vals):
                s.append(f'<circle cx="{px(i):.1f}" cy="{py(v):.1f}" r="3.2" fill="{col}"/>')
    s.append("</svg>")
    return "".join(s)


def svg_gate_flow():
    w, h = 900, 300
    s = [svg_open(w, h, "三层门禁：IR、正确性、性能"), arrow_defs("gfArr", C_TEAL_B)]
    stages = [
        (20, "构建", "cmake --build build", "#f8fafc", C_LINE, C_INK),
        (185, "ir", "transform 产物结构", "#eef2ff", C_INDIGO_B, C_INDIGO),
        (350, "auto", "11 形状 Nwin 表", "#f0fdfa", C_TEAL_B, C_TEAL),
        (515, "correct", "逐元素一致", "#fff7ed", C_AMBER_B, C_AMBER),
        (680, "perf", "阈值 + 交错重复", "#f5f3ff", "#8b5cf6", "#5b21b6"),
    ]
    for x, t, sub, fill, stroke, col in stages:
        s.append(box(x, 46, 150, 78, t, sub, fill, stroke, col, "#475569", fs=14, sfs=10))
    for x in (170, 335, 500, 665):
        s.append(path_arrow(x, 85, x + 13, 85, "gfArr", C_TEAL_B, 2))
    # failure loop
    s.append(f'<path d="M 755 128 L 755 168 L 95 168 L 95 128" fill="none" stroke="{C_RED}" stroke-width="1.8" stroke-dasharray="5 4" marker-end="url(#gfArr)"/>')
    s.append(txt(430, 162, "任一层失败 → 回退改动，先补失败测试", 11.5, C_RED, 700, "middle"))
    s.append(f'<rect x="20" y="196" width="860" height="82" rx="12" fill="#f8fafc" stroke="{C_LINE}"/>')
    s.append(txt(38, 222, "门禁对应脚本", 12, C_INK, 800))
    s.append(txt(38, 245, "test_fastpack.py --stage all", 11.5, C_TEAL, 700, mono=True))
    s.append(txt(330, 245, "test_mk_auto.py", 11.5, C_INDIGO, 700, mono=True))
    s.append(txt(520, 245, "test_multicore.py --stage all", 11.5, C_AMBER, 700, mono=True))
    s.append(txt(38, 266, "IR 结构 / 数值一致 / 性能阈值三层都必须在同一次提交里通过。", 11.5, C_SLATE))
    s.append("</svg>")
    return "".join(s)


def svg_measure_protocol():
    w, h = 900, 236
    s = [svg_open(w, h, "测量协议：同二进制 A/B、交错重复、取中位数"), arrow_defs("mpArr", C_INDIGO_B)]
    s.append(txt(20, 26, "为什么不能用单次运行比较两个版本", 12, C_INK, 800))
    rows = [
        ("版本 A", "#94a3b8", 40),
        ("版本 B", C_TEAL_B, 78),
        ("版本 A", "#94a3b8", 116),
        ("版本 B", C_TEAL_B, 154),
    ]
    for label, col, y in rows:
        s.append(txt(24, y + 14, label, 11.5, C_SLATE, 700))
        for i in range(3):
            x = 90 + i * 150
            s.append(f'<rect x="{x}" y="{y}" width="132" height="20" rx="5" fill="{col}" opacity="{0.85 - i*0.18}"/>')
            s.append(txt(x + 66, y + 14, f"run {i+1}", 10.5, "#0b1220", 700, "middle"))
    s.append(f'<rect x="560" y="34" width="310" height="140" rx="12" fill="#f8fafc" stroke="{C_LINE}"/>')
    s.append(txt(578, 58, "判定规则", 12, C_INK, 800))
    s.append(txt(578, 82, "· 交错 A/B/A/B，不跨进程", 11.5, C_SLATE))
    s.append(txt(578, 104, "· 每档 3–6 轮，取中位数", 11.5, C_SLATE))
    s.append(txt(578, 126, "· 未改动用例 = 噪声底线", 11.5, C_SLATE))
    s.append(txt(578, 148, "· 收益 < 噪声 ⇒ 视为无收益", 11.5, C_RED, 700))
    s.append(txt(20, 208, "本机共享节点噪声约 ±4–7%；自动调优本身也会因计时噪声选到不同配置。", 11.5, C_SLATE))
    s.append("</svg>")
    return "".join(s)


def svg_timeline_strip():
    w, h = 900, 150
    s = [svg_open(w, h, "优化里程碑时间线")]
    s.append(f'<line x1="40" y1="66" x2="860" y2="66" stroke="{C_LINE}" stroke-width="3"/>')
    marks = [
        ("M1", "pack helper", 70, C_TEAL_B),
        ("M2", "tile 放大", 230, C_INDIGO_B),
        ("A+C+G", "33.3 ms", 390, C_AMBER_B),
        ("M3", "2.36×", 560, "#8b5cf6"),
        ("M4-c", "2.51×", 750, C_TEAL_B),
    ]
    for label, sub, x, col in marks:
        s.append(f'<circle cx="{x}" cy="66" r="9" fill="#fff" stroke="{col}" stroke-width="4"/>')
        s.append(txt(x, 44, label, 13, C_INK, 800, "middle"))
        s.append(txt(x, 96, sub, 11, C_SLATE, 600, "middle"))
    s.append(txt(40, 130, "2026-11：单线程从 156.6 ms 追到 33.3 ms；随后多核 2.36× → 2.51×。", 11.5, C_SLATE))
    s.append("</svg>")
    return "".join(s)


# ---------------------------------------------------------------------------
# shared content blocks
# ---------------------------------------------------------------------------
def stat_cards():
    return f"""<div class="metric-strip">
  <div class="metric teal"><div class="k">单线程 vs torch t1</div><div class="v">1.57<small>×</small></div><div class="s">11 形状 geomean · 10/11 ≥ 1.0×</div></div>
  <div class="metric indigo"><div class="k">多核 geomean</div><div class="v">2.51<small>×</small></div><div class="s">M4-c · 单 NUMA · t1–t32</div></div>
  <div class="metric amber"><div class="k">最高吞吐</div><div class="v">386<small> GF</small></div><div class="s">custom_4 @ t32</div></div>
  <div class="metric"><div class="k">正确性</div><div class="v">11/11</div><div class="s">t8 与 t32 各一遍</div></div>
</div>"""


def table_single():
    rows = []
    for s in SHAPES:
        g, ms, gf, tg, ratio = SINGLE[s]
        cls = "up" if ratio >= 1 else "down"
        rows.append(f"<tr><td>{s}</td><td>{g:.2f}</td><td>{ms:.1f}</td>"
                    f"<td>{gf:.1f}</td><td>{tg:.1f}</td>"
                    f'<td class="{cls}">{ratio:.2f}×</td></tr>')
    return f"""<div class="table-wrap"><table class="table">
<thead><tr><th>形状</th><th>GFLOP</th><th>MLIR ms</th><th>MLIR GF</th><th>torch t1 GF</th><th>比值</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>"""


def table_m4():
    rows = []
    for s in SHAPES:
        ser, t1, t8, t16, t32, su, gf, nw, wid = M4[s]
        best = min(t1, t8, t16, t32)
        cells = []
        for v in (ser, t1, t8, t16, t32):
            cls = ' class="best"' if v == best else ""
            cells.append(f"<td{cls}>{v:.1f}</td>")
        rows.append(f"<tr><td>{s}</td>{''.join(cells)}<td>{su:.2f}×</td>"
                    f"<td>{gf}</td><td>{nw}</td><td>{wid}</td></tr>")
    return f"""<div class="table-wrap"><table class="table">
<thead><tr><th>形状</th><th>串行</th><th>t1</th><th>t8</th><th>t16</th><th>t32</th><th>加速比</th><th>GF</th><th>Nwin</th><th>宽度</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>"""


# ---------------------------------------------------------------------------
# pages
# ---------------------------------------------------------------------------
def build_index():
    body = f"""<section class="hero"><div class="wrap">
  <div>
    <span class="eyebrow">MLIR · ConvBench · Kunpeng aarch64</span>
    <h1>从 <em>0.66×</em> 到 <em>2.51×</em><br>一条编译器管线的优化轨迹</h1>
    <p class="lead">把卷积拆成 pack → GEMM → unpack 三段，各自贴近自己的 roofline；
      单线程追上 PyTorch，多核在单 NUMA 节点内做到 2.51×。所有改动都在 IR / transform / pass 层。</p>
    <div class="chips">
      <span class="chip"><b>2.0 GHz</b> 固定频率</span>
      <span class="chip"><b>单 NUMA</b> node0</span>
      <span class="chip"><b>≤32</b> 线程报数</span>
      <span class="chip"><b>11/11</b> 正确</span>
    </div>
  </div>
  <div class="hero-panel">
    <div class="panel-title"><span>SConv pipeline</span><span>live diagram</span></div>
    {svg_hero_pipeline()}
  </div>
</div></section>

<section class="stats-band"><div class="wrap">{stat_cards()}</div></section>

<section class="section"><div class="wrap">
  <div class="section-head">
    <span class="eyebrow">两条路径</span>
    <h2>编译器交付物，与性能上限探针</h2>
    <p>同一套算法有两种承载方式：一条必须留在 IR 里，一条可以激进地做运行时自适应。</p>
  </div>
  <div class="grid-2">
    <div class="card accent-teal"><span class="pill teal">交付路径</span>
      <h3>MLIR / ConvBench</h3>
      <p>交付物是 <code>sconv-opt</code> 一条命令。优化发生在 <code>transform.structured.sconv</code>、
        packing lowering、<code>-sconv-parallelize</code> 与 OpenMP lowering 中；没有第二套运行时线程池。</p>
      <p class="small mt-10">成绩：单线程 33.3 ms / 127.7 GF（custom_0），多核 geomean 2.51×。</p>
    </div>
    <div class="card accent-indigo"><span class="pill indigo">参考实现</span>
      <h3>native_impl</h3>
      <p>不受编译器抽象约束，用来回答“这个算法在 Kunpeng 上到底能跑多快”。它的数字是上限参考，
        不直接算作交付路径的成绩。</p>
      <p class="small mt-10">成绩：18 用例 t1–t32 对 PyTorch geomean 1.51×–1.62×。</p>
    </div>
  </div>
</div></section>

<section class="section"><div class="wrap">
  <div class="section-head">
    <span class="eyebrow">优化阶梯</span>
    <h2>每一步都先回答“钱花在哪”</h2>
    <p>从 ABI 时间、打包带宽、微内核粒度，到多核宽度与调度策略——顺序不是随意的。</p>
  </div>
  <div class="diagram mb-16">{svg_timeline_strip()}<div class="caption">里程碑与关键数字（2.0 GHz）</div></div>
  <div class="grid-3">
    <div class="card accent-teal"><div class="num">M1</div><h3>快速输入打包</h3>
      <p>行分段 <code>memcpy</code> + 打包 buffer 上提复用。<b>156.6 → 115.7 ms（−26%）</b>，
        其中约 28 ms 来自“不再每次落到冷页”。</p></div>
    <div class="card accent-indigo"><div class="num">M2</div><h3>微内核 tile 放大</h3>
      <p>支持 multipacking，把 <code>Nf</code> 放大到 32。<b>≈84–87 ms</b>，未达 75 ms；
        原因是打包后的 A/C 放不进 L2，端到端受 DRAM 限制。</p></div>
    <div class="card accent-amber"><div class="num">A+C+G</div><h3>边界拷贝与 tile 自适应</h3>
      <p>快 <code>memrefCopy</code>、宽 tile 下的 <code>Nc</code> 自适应、<code>Nwin=160</code>。
        <b>33.3 ms / 127.7 GF</b>；单线程 geomean 1.57×。</p></div>
    <div class="card accent-indigo"><div class="num">M3</div><h3>单 NUMA 多核</h3>
      <p>窗口 tile 循环 → <code>scf.parallel</code> → OpenMP + per-thread scratch。
        <b>geomean 2.36×</b>，最高 338 GF；t16≈t32，宽度只有 5–20。</p></div>
    <div class="card accent-teal"><div class="num">M4-c</div><h3>多核专用 Nwin</h3>
      <p>多数形状降到 80/40，并用 CSA 探测阻止 IS→WS 翻转。
        <b>2.51× / 386 GF</b>；宽度 40 不再更快，于是停手。</p></div>
    <div class="card"><div class="num">下一步</div><h3>串行 / 带宽项</h3>
      <p>E1 filter 打包去 3/4 冗余 → E2 输入打包分段 → B 边界拷贝。
        继续加宽（M4-a/b）按实测判据暂缓。</p></div>
  </div>
</div></section>

<section class="section"><div class="wrap">
  <div class="section-head"><span class="eyebrow">关键判断</span><h2>三条最贵的经验</h2></div>
  <div class="grid-3">
    <div class="card"><span class="pill red">坑 1</span><h3>归约维不能并行</h3>
      <p>把 <code>Nc</code> 标成并行，8/11 形状错、max_diff 1e6–1e8。它不是调度问题，是语义错误。</p></div>
    <div class="card"><span class="pill amber">坑 2</span><h3>“更多任务”不一定更快</h3>
      <p>Nwin 80 会让 c1/c2/9 翻到 WS、窗口组缩小；任务更多，时间反而 2 倍。tile 选择必须调度感知。</p></div>
    <div class="card"><span class="pill teal">坑 3</span><h3>微基准快 ≠ 端到端快</h3>
      <p>warm-cache 里 <code>m16/n32/k108</code> 只要 21 ms，in-situ 仍是 50–60 ms：A/C 放不进 L2。</p></div>
  </div>
</div></section>

<section class="section"><div class="wrap">
  <div class="section-head"><span class="eyebrow">文档地图</span><h2>从哪里继续读</h2>
    <p>HTML 负责讲清楚“为什么”和“怎么走”；Markdown 负责完整数据与可复现命令。</p></div>
  <div class="grid-3">
    <a class="card accent-teal" href="principles.html"><h3>原理</h3><p>三阶段流水线、循环嵌套、tile 参数与 roofline。</p><p class="small mt-10">principles.html →</p></a>
    <a class="card accent-indigo" href="pipeline.html"><h3>管线</h3><p>transform / pass 的顺序，M1–M4 的 IR 变化与代码地图。</p><p class="small mt-10">pipeline.html →</p></a>
    <a class="card accent-amber" href="timeline.html"><h3>演进实录</h3><p>每个假设、翻车、根因与决策；包括被证伪的方向。</p><p class="small mt-10">timeline.html →</p></a>
    <a class="card" href="multicore.html"><h3>多核</h3><p>标记、并行链、per-thread scratch、宽度与调度翻转。</p><p class="small mt-10">multicore.html →</p></a>
    <a class="card" href="performance.html"><h3>性能</h3><p>11 形状单线程/多核数据、线程曲线与测量口径。</p><p class="small mt-10">performance.html →</p></a>
    <a class="card" href="reproduce.html"><h3>复现</h3><p>环境、构建、运行命令、门禁与预期结果。</p><p class="small mt-10">reproduce.html →</p></a>
  </div>
  <div class="callout note mt-24"><strong>Markdown 权威数据：</strong>
    <code>docs/PERFORMANCE_VS_PYTORCH.md</code>（附录 C）、
    <code>docs/OPTIMIZATION_PLAN_2026-11-18.md</code>（§8–§9）、
    <code>docs/OPTIMIZATION_DESIGN_AND_RESULTS.md</code>（§3.6–§3.12）、
    <code>docs/OPTIMIZATION_JOURNEY_2026-11-19.md</code>（演进实录）。</div>
</div></section>"""
    return page("总览", "SConvTransform 优化总览：从 0.66× 到 2.51× 的编译器优化轨迹。", "index.html", body)


def build_principles():
    body = f"""<section class="section"><div class="wrap">
  <div class="section-head"><span class="eyebrow">原理 01</span><h2>卷积被拆成三个物理阶段</h2>
    <p>把“搬运”和“计算”显式分开，每段都有自己的 roofline，才能判断一个优化到底值不值得做。</p></div>
  <div class="diagram mb-16">{svg_conv_flow()}<div class="caption">pack → GEMM → unpack 与各自的瓶颈</div></div>
  <div class="grid-3">
    <div class="card accent-teal"><h3>pack</h3><p>im2col 物化，9× 写放大。三种写法都停在 <b>13 GB/s/core</b> ⇒ 已贴带宽顶。</p></div>
    <div class="card accent-amber"><h3>GEMM</h3><p>kblas 微内核，<b>0.7–2.3 TFLOPS</b>（t32）。效率由 <code>m/n/k</code> 与 tile 是否留在 L2 决定。</p></div>
    <div class="card accent-indigo"><h3>unpack</h3><p>还原 NCHW，占 t1 总时间 <b>5–13%</b>，不是瓶颈。</p></div>
  </div>
</div></section>

<section class="section"><div class="wrap">
  <div class="section-head"><span class="eyebrow">原理 02</span><h2>并行维度是 schedule 的属性</h2>
    <p>同一个循环嵌套里，只有“写不相交 C 列、body 是完整工作项”的窗口 tile 循环可以并行。</p></div>
  <div class="diagram mb-16">{svg_loopnest()}<div class="caption">循环嵌套与并行/归约/共享 scratch 的边界</div></div>
  <div class="callout bad"><strong>为什么 Nc 不能并行：</strong>它是 GEMM 的归约维。同一个输出 tile 被 6 个
    <code>Nc</code> 迭代累加，并行后多个线程同时写同一块 C——结果直接错，不是“慢一点”。</div>
</div></section>

<section class="section"><div class="wrap">
  <div class="section-head"><span class="eyebrow">原理 03</span><h2>tile 参数决定一次 GEMM 干多少活</h2>
    <p><code>m = Nwin</code>，<code>n = Nf</code>，<code>k = Nc·Fh·Fw</code>。单线程与多核对这三个参数的偏好相反。</p></div>
  <div class="diagram mb-16">{svg_tile_params()}<div class="caption">Nwin / Nf / Nc 如何映射到 GEMM 的 m / n / k</div></div>
  <div class="grid-2">
    <div class="card accent-teal"><h3>单线程：大 tile 少调用</h3>
      <p>Nwin=160、Nf=32、Nc=16–32，把每次 GEMM 的固定开销摊薄。custom_0 的 720 次调用降到更少、更宽。</p></div>
    <div class="card accent-amber"><h3>多核：小 tile 多任务</h3>
      <p>并行宽度 = 窗口 tile 循环 trip count。但 Nwin 会改变 k3 / tile_c，甚至翻转 IS/WS —— 不能盲目减半。</p></div>
  </div>
</div></section>

<section class="section"><div class="wrap">
  <div class="section-head"><span class="eyebrow">原理 04</span><h2>六条设计原则</h2></div>
  <div class="grid-3">
    <div class="card"><div class="num">1</div><h3>搬运与计算分开</h3><p>各自贴各自的 roofline；ABI 时间不能算进算法。</p></div>
    <div class="card"><div class="num">2</div><h3>一次性打包、多轮消费</h3><p>输入 tile 提到 <code>oc</code> 循环外；filter 打包只在 Nf 变化时重建。</p></div>
    <div class="card"><div class="num">3</div><h3>粒度跟着形状走</h3><p>batch=1、Ohw=49 时大 tile 只有 2 个任务；参数必须能自适应。</p></div>
    <div class="card"><div class="num">4</div><h3>先看端到端口径</h3><p>kernel-only 会漏掉 pad、alloc、线程创建等固定开销。</p></div>
    <div class="card"><div class="num">5</div><h3>同二进制 A/B</h3><p>交错重复、取中位数、用未改动用例给噪声底线。</p></div>
    <div class="card"><div class="num">6</div><h3>特殊硬件只喂擅长形状</h3><p>SME 只保留给能铺满 16×16 tile 的 c9/c10，扩面实测全面更慢。</p></div>
  </div>
</div></section>"""
    return page("原理", "SConv 的三阶段流水线、循环嵌套、tile 参数与 roofline。", "principles.html", body,
                ("原理：为什么这样拆", "从 im2col、GEMM 到并行维度，先把“钱花在哪”讲清楚。", "Principles"))


def build_pipeline():
    body = f"""<section class="section"><div class="wrap">
  <div class="section-head"><span class="eyebrow">管线 01</span><h2>一条命令，从 linalg 到 OpenMP</h2>
    <p>所有优化都在 transform / pass 层完成；并行不是运行时偷偷开线程，而是 IR 里显式的 <code>scf.parallel</code>。</p></div>
  <div class="diagram mb-16">{svg_transform_pipeline()}<div class="caption">transform 与 pass 的顺序（M3/M4 共用同一条 lowering）</div></div>
  <div class="code-card"><div class="code-head"><span>端到端命令</span><span>bash</span></div>
<pre><code>sconv-opt -transform=test/lowering/sconv-blas-fastp-mc.mlir \\
  -sconv-parallelize -sconv-lower-parallel conv.mlir -o conv.mc.llvm.mlir
OMP_NUM_THREADS=16 numactl --cpunodebind=0 --membind=0 \\
  mlir-cpu-runner conv.mc.llvm.mlir -e main -entry-point-result=void \\
  -shared-libs=...,libomp.so</code></pre></div>
</div></section>

<section class="section"><div class="wrap">
  <div class="section-head"><span class="eyebrow">管线 02</span><h2>M1–M4 各自改了什么</h2></div>
  <div class="table-wrap"><table class="table">
    <thead><tr><th>阶段</th><th>代码位置</th><th>关键改动</th><th>结果</th></tr></thead>
    <tbody>
      <tr><td>M1 快速打包</td><td><code>lib/LowerToPack.cpp</code> + <code>runtime/pack_input_tile.c</code></td><td>标量 gather → 行分段 memcpy；buffer 上提复用</td><td>156.6 → 115.7 ms</td></tr>
      <tr><td>M2 tile 放大</td><td><code>inputMultipackingOpt</code> + <code>lower.to_pack</code></td><td>支持 multipacking；Nf 8→32</td><td>≈84–87 ms</td></tr>
      <tr><td>A 自适应 mK</td><td><code>lib/SConv.cpp</code></td><td><code>[0,0]</code> 哨兵 + 最大整除因子</td><td>11/11 无回退</td></tr>
      <tr><td>A+C+G</td><td><code>fast_memref_copy.c</code>、<code>computeTileC</code>、<code>selectAutoNwin</code></td><td>快边界拷贝、Nc 自适应、Nwin 160</td><td>33.3 ms / 127.7 GF</td></tr>
      <tr><td>M3 多核</td><td><code>marker</code>、<code>lib/Parallelize.cpp</code></td><td>窗口 tile → scf.parallel；per-thread scratch</td><td>geomean 2.36×</td></tr>
      <tr><td>M4-c</td><td><code>selectAutoNwinMulticore</code>、marker/scratch 修复</td><td>多核专用 Nwin + CSA 调度保护</td><td>geomean 2.51×</td></tr>
    </tbody></table></div>
</div></section>

<section class="section"><div class="wrap">
  <div class="section-head"><span class="eyebrow">管线 03</span><h2>IR 上看得见的变化</h2>
    <p>以输入打包为例：从逐元素 <code>linalg.generic</code> 变成一次运行时调用。</p></div>
  <div class="grid-2">
    <div class="code-card"><div class="code-head"><span>M1 之前：标量 gather</span><span>MLIR</span></div>
<pre><code>scf.for %ic ...
  scf.for %fh ...
    scf.for %fw ...
      %v = tensor.extract %in[%n, %ic, %h, %w]
      tensor.insert %v into %pack[...]</code></pre></div>
    <div class="code-card"><div class="code-head"><span>M1 之后：行分段 helper</span><span>MLIR</span></div>
<pre><code>llvm.call @pack_input_tile(
  %in, %dst, in_off, ow, iw, stride_h, stride_w,
  dil_h, dil_w, nc, fh, fw, nwin, ...) : ...</code></pre></div>
  </div>
  <div class="callout note mt-16"><strong>几何信息显式携带：</strong><code>sconv.pack_geom</code> 记录
    <code>[Ow, Iw, strideH, strideW, dilH, dilW]</code>，避免从 affine map 反推。</div>
</div></section>

<section class="section"><div class="wrap">
  <div class="section-head"><span class="eyebrow">管线 04</span><h2>代码地图</h2></div>
  <div class="table-wrap"><table class="table">
    <thead><tr><th>想改什么</th><th>文件</th><th>关键位置 / 开关</th></tr></thead>
    <tbody>
      <tr><td>tiling / packing / 微内核</td><td><code>lib/SConv.cpp</code>、<code>lib/LowerToBLAS.cpp</code></td><td><code>applyTileTo</code>、<code>tileUsingSCF</code></td></tr>
      <tr><td>输入打包</td><td><code>lib/LowerToPack.cpp</code>、<code>runtime/pack_input_tile.c</code></td><td><code>applyInputPacking</code>、<code>sconv.parallel_scratch</code></td></tr>
      <tr><td>多核</td><td><code>lib/SConv.cpp</code>、<code>lib/Parallelize.cpp</code>、<code>lib/LowerToPack.cpp</code></td><td>marker、<code>sconv-parallelize</code>、<code>-sconv-lower-parallel</code></td></tr>
      <tr><td>多核 tile 选择</td><td><code>lib/SConv.cpp</code></td><td><code>selectAutoNwinMulticore</code>、<code>multicore</code> attr</td></tr>
      <tr><td>门禁</td><td><code>scripts/convbench/</code></td><td><code>test_fastpack.py</code>、<code>test_mk_auto.py</code>、<code>test_multicore.py</code></td></tr>
    </tbody></table></div>
</div></section>"""
    return page("管线", "MLIR transform/pass 管线、M1–M4 的改动与代码地图。", "pipeline.html", body,
                ("管线：从 linalg 到 OpenMP", "一条命令产出可执行的 LLVM IR；每一步都能在 IR 里看到。", "Pipeline"))


def build_timeline():
    tl = [
        ("2026-09", "起点：两条路径", "native_impl 作为性能探针，MLIR 作为交付物。",
         ["18 用例 t1–t32 geomean 1.51×–1.62×", "明确“交付 / 上限”分工"], ""),
        ("2026-11", "M1：快速输入打包", "标量 gather 换成行分段 memcpy，并把打包 buffer 提到循环外。",
         ["156.6 → 115.7 ms", "−26%；约 28 ms 来自 buffer 复用"], "teal"),
        ("2026-11", "M2：微内核 tile 放大", "支持 multipacking，Nf 8→32。微基准很快，但 in-situ 受 L2/DRAM 限制。",
         ["≈84–87 ms", "原目标 75 ms 未达"], "indigo"),
        ("2026-11", "A：形状自适应 mK", "固定 [64,32] 在不整除形状上落入 edge 路径，改用 [0,0] + 最大整除因子。",
         ["11/11 优于旧 [16,8]", "无回退"], "amber"),
        ("2026-11-18", "A+C+G：33.3 ms", "快边界拷贝、宽 tile Nc 自适应、Nwin=160。单线程追到 1.57×。",
         ["33.3 ms / 127.7 GF", "10/11 ≥ 1.0×；c7 0.83×"], "teal"),
        ("2026-11-18", "M3：单 NUMA 多核", "窗口 tile → scf.parallel → OpenMP；per-thread scratch。",
         ["geomean 2.36×", "最高 338 GF；t16≈t32"], "indigo"),
        ("2026-11-19", "M4-c：多核专用 Nwin", "第一版盲目减半翻车；加入 marker/scratch 修复与 CSA 调度保护。",
         ["geomean 2.51× / 386 GF", "宽度 40 不再更快 → 停手"], "teal"),
    ]
    items = []
    for when, title, desc, pills, kind in tl:
        pill_html = "".join(f'<span class="pill {"teal" if i == 0 else "gray"}">{p}</span>' for i, p in enumerate(pills))
        items.append(f"""<div class="tl-item {kind}"><div class="when">{when}</div>
          <h3>{title}</h3><p>{desc}</p><div class="meta">{pill_html}</div></div>""")
    body = f"""<section class="section"><div class="wrap">
  <div class="section-head"><span class="eyebrow">演进 01</span><h2>时间线：每一次“以为会快”</h2>
    <p>完整版见 <code>docs/OPTIMIZATION_JOURNEY_2026-11-19.md</code>。这里保留决策主线与关键数字。</p></div>
  <div class="timeline">{''.join(items)}</div>
  <div class="diagram mt-24">{svg_timeline_strip()}<div class="caption">里程碑与关键数字（2.0 GHz）</div></div>
</div></section>

<section class="section"><div class="wrap">
  <div class="section-head"><span class="eyebrow">演进 02</span><h2>怎么判定“真的变快了”</h2>
    <p>共享节点噪声约 ±4–7%，所以所有结论都建立在同二进制交错 A/B 上，而不是单次运行。</p></div>
  <div class="diagram mb-16">{svg_measure_protocol()}<div class="caption">测量协议：交错重复、取中位数、用未改动用例给噪声底线</div></div>
</div></section>

<section class="section"><div class="wrap">
  <div class="section-head"><span class="eyebrow">演进 03</span><h2>被证伪的方向，同样是结论</h2>
    <p>记录“为什么不继续”，和记录“为什么继续”一样重要。</p></div>
  <div class="table-wrap"><table class="table">
    <thead><tr><th>方向</th><th>做法</th><th>结果</th><th>判定依据</th></tr></thead>
    <tbody>
      <tr><td>打包向量化</td><td>内层换 std::copy 特化</td><td class="flat">0.91–1.06×</td><td>噪声内；clang 已向量化且带宽受限</td></tr>
      <tr><td>打包三变体</td><td>标量 / memcpy / 合并写</td><td class="flat">13.17 / 13.33 / 12.39 GB/s</td><td>同速 ⇒ 带宽 roofline</td></tr>
      <tr><td>扩大 SME 覆盖</td><td>c0/c2/c5/c7 路由 SME</td><td class="down">全面更慢</td><td>kernel-only 均低于 GEMM</td></tr>
      <tr><td>放大 Nf</td><td><code>Nf ∈ {{16,32,64,128,256}}</code></td><td class="flat"><code>{{16,32}}</code> 最优</td><td>无单调收益</td></tr>
      <tr><td>无条件搜 Nc</td><td>所有形状都搜</td><td class="down">thr_n8 −43%</td><td>必须加“饥饿形状 + 3% 门槛”</td></tr>
      <tr><td>固定 [64,32]</td><td>全局写死</td><td class="down">c5 229→6227 ms</td><td>不整除就落 edge 路径</td></tr>
      <tr><td>M4 第一版盲目减半</td><td>所有 IC&gt;64 都降到 80</td><td class="down">c1/c2/9 慢 2×</td><td>IS→WS 翻转 + k3 缩小</td></tr>
      <tr><td>每迭代 memref.alloc</td><td>每次新分配消除共享</td><td class="down">t1 499→820 ms</td><td>malloc/缺页成本</td></tr>
    </tbody></table></div>
</div></section>

<section class="section"><div class="wrap">
  <div class="section-head"><span class="eyebrow">演进 04</span><h2>M4-c 的三次“原来如此”</h2></div>
  <div class="grid-3">
    <div class="card accent-amber"><span class="pill amber">根因 1</span><h3>marker 标错循环</h3>
      <p><code>loopOps[size-2]</code> 在 tiling 顺序翻转时是 filter-subblock。改成找最近的
        <code>step == nwindows</code> 祖先。</p></div>
    <div class="card accent-indigo"><span class="pill indigo">根因 2</span><h3>打包被 hoist</h3>
      <p>被 hoist 的打包只读共享 buffer，不能私有化。只有位于 <code>sconv.parallel</code> 内的打包才打 scratch 标记。</p></div>
    <div class="card accent-teal"><span class="pill red">根因 3</span><h3>IS → WS 翻转</h3>
      <p>Nwin=80 让 c1/c2/9 翻到 WS，窗口组从 1600 缩到 400。用 CSA 探测 base/candidate，不同就保留 base。</p></div>
  </div>
  <div class="callout good mt-16"><strong>停手判据：</strong>计划 §9.6 预先写明“宽度 40 仍 &lt;600 GF
    就先做串行项”。M4-c 实测宽度 40 不比 20 快、t16→t32 ≤17%，因此 M4-a/b 暂缓，
    下一步 E1/E2/B。</div>
</div></section>"""
    return page("演进", "优化演进实录：假设、翻车、根因与决策。", "timeline.html", body,
                ("演进实录：尝试、翻车与抉择", "从 0.66× 到 2.51×，记录每一步为什么这样做、为什么停。", "Journey"))


def build_multicore():
    body = f"""<section class="section"><div class="wrap">
  <div class="section-head"><span class="eyebrow">多核 01</span><h2>并行链：只标记一个循环</h2>
    <p><code>sconv-parallelize</code> 把“直接嵌套的已标记循环链”折成 <code>scf.parallel</code>；
      共享 scratch 的循环不入链。</p></div>
  <div class="diagram mb-16">{svg_parallel_chain()}<div class="caption">窗口 tile 循环 → scf.parallel；scratch 按线程取槽</div></div>
  <div class="grid-3">
    <div class="card accent-teal"><h3>标记</h3><p>从微内核向上找最近的 <code>step == nwindows</code> 循环；不再用位置索引。</p></div>
    <div class="card accent-indigo"><h3>私化</h3><p>打包在并行区内 → scratch 扩成 <code>[64][...]</code>，按 <code>omp_get_thread_num()</code> 取槽。</p></div>
    <div class="card accent-amber"><h3>共享</h3><p>打包被 hoist 到并行区外 → buffer 只读，保持共享，不打 <code>parallel_scratch</code>。</p></div>
  </div>
  <div class="callout bad mt-16"><strong>正确性红线：</strong>Nc 是 GEMM 的归约维。原型并行 Nc 后
    8/11 形状错、max_diff 1e6–1e8。现在只并行写不相交 C 列的窗口 tile。</div>
</div></section>

<section class="section"><div class="wrap">
  <div class="section-head"><span class="eyebrow">多核 02</span><h2>M3：宽度只有 5–20</h2>
    <p>线程数超过任务数后只是空转；t16 与 t32 基本持平。</p></div>
  <div class="grid-2">
    <div class="diagram">{svg_multiline([
        ("custom_0", [M3["custom_0"][1], M3["custom_0"][2], M3["custom_0"][3], M3["custom_0"][4]],
         [M4["custom_0"][1], M4["custom_0"][2], M4["custom_0"][3], M4["custom_0"][4]]),
        ("custom_4", [M3["custom_4"][1], M3["custom_4"][2], M3["custom_4"][3], M3["custom_4"][4]],
         [M4["custom_4"][1], M4["custom_4"][2], M4["custom_4"][3], M4["custom_4"][4]]),
        ("custom_8", [M3["custom_8"][1], M3["custom_8"][2], M3["custom_8"][3], M3["custom_8"][4]],
         [M4["custom_8"][1], M4["custom_8"][2], M4["custom_8"][3], M4["custom_8"][4]])],
        width=880, height=250)}
      <div class="caption">灰色 = M3，青色 = M4-c；纵轴 ms/iter，横轴线程数</div></div>
    <div class="card"><h3>宽度从哪来</h3>
      <p>并行宽度 = 每个窗口组里的窗口 tile 数，由 <code>Ohw</code>、<code>Nwin</code> 和分组方式共同决定。</p>
      <div class="table-wrap mt-16"><table class="table" style="min-width:0">
        <thead><tr><th>形状</th><th>M3 宽度</th><th>M4-c 宽度</th></tr></thead>
        <tbody>
          <tr><td>c0/c4/c6/c8/c10</td><td>10</td><td class="best">20</td></tr>
          <tr><td>c3/c7</td><td>5</td><td class="best">10</td></tr>
          <tr><td>c5</td><td>20</td><td>20</td></tr>
          <tr><td>c1/c2/c9</td><td>10</td><td>10（调度保护）</td></tr>
        </tbody></table></div>
      <p class="small mt-10">宽度 40 的 mc40 实测不再更快，因此停在 20–40。</p>
    </div>
  </div>
</div></section>

<section class="section"><div class="wrap">
  <div class="section-head"><span class="eyebrow">多核 03</span><h2>M4-c：多核专用 Nwin 与调度保护</h2>
    <p>“任务更多”不一定更快。Nwin 会改变 k3 / tile_c，甚至翻转 IS/WS。</p></div>
  <div class="diagram mb-16">{svg_schedule_tree()}<div class="caption">M4-c 的 Nwin 决策树；c1/c2/9 因 IS→WS 翻转保留 160</div></div>
  <div class="grid-2">
    <div class="card accent-teal"><h3>采用 candidate</h3>
      <p>c0/c4/c6/c8/c10：80；c3/c7：40。宽度翻倍，提升 12%–20%（c3/c7 约 6%–8%）。</p></div>
    <div class="card accent-amber"><h3>保留 base</h3>
      <p>c5（IC=48，打包受限）；c1/c2/9（Nwin=80 翻到 WS，且 c2/9 的窗口组缩到 400，宽度反而 10→5）。</p></div>
  </div>
</div></section>

<section class="section"><div class="wrap">
  <div class="section-head"><span class="eyebrow">多核 04</span><h2>M4-c 结果：2.51× / 386 GF</h2></div>
  {table_m4()}
  <div class="grid-3 mt-16">
    <div class="card accent-teal"><h3>geomean</h3><p><b>2.51×</b>（M3 2.36×）；相对 M3 best-time geomean 1.08×。</p></div>
    <div class="card accent-indigo"><h3>最高吞吐</h3><p><b>386 GF</b>（custom_4 @t32）；9/11 形状提升。</p></div>
    <div class="card accent-amber"><h3>正确性</h3><p><b>11/11</b>（t8、t32）；门禁 <code>test_multicore.py --stage all</code> 全绿。</p></div>
  </div>
</div></section>

<section class="section"><div class="wrap">
  <div class="section-head"><span class="eyebrow">多核 05</span><h2>五个实现坑</h2></div>
  <div class="grid-2">
    <div class="card"><h3>1. alloca_scope 锁死 lowering</h3><p><code>convert-scf-to-openmp</code> 无条件包 scope，随后 verifier 报错。新增 <code>sconv-erase-alloca-scope</code>。</p></div>
    <div class="card"><h3>2. reinterpret_cast 的 offset</h3><p>私化后 offset 要以 base 为基准加 <code>tid * slotElements</code>，结果类型改成动态 offset。</p></div>
    <div class="card"><h3>3. 不能每迭代 alloc</h3><p>每次新分配让 t1 从 499 → 820 ms；改成 <code>[64][...]</code> 槽整轮复用。</p></div>
    <div class="card"><h3>4. 窗口循环不能按位置选</h3><p>tiling 顺序会翻转；按 <code>step == nwindows</code> 找。</p></div>
    <div class="card"><h3>5. 不是所有打包都在并行区</h3><p>hoist 的打包只读共享 buffer；无条件私化会让 pass 拒绝构建。</p></div>
    <div class="card accent-teal"><h3>结论</h3><p>能解释“为什么不继续”与“为什么继续”同样重要：M4-a/b 暂缓，先做 E1/E2/B。</p></div>
  </div>
</div></section>"""
    return page("多核", "M3/M4 多核：并行链、per-thread scratch、宽度与调度保护。", "multicore.html", body,
                ("多核：把并行留在 IR 里", "窗口 tile → scf.parallel → OpenMP；宽度与调度共同决定收益。", "Multicore"))


def build_performance():
    single_items = [(s, SINGLE[s][4], "teal" if SINGLE[s][4] >= 1 else "red") for s in SHAPES]
    gain_items = [(s, GAIN[s], "red" if GAIN[s] < 0 else ("teal" if GAIN[s] > 0.5 else "amber")) for s in SHAPES]
    gf_items = [(s, M4[s][6], "indigo") for s in SHAPES]
    body = f"""<section class="section"><div class="wrap">
  <div class="section-head"><span class="eyebrow">性能 01</span><h2>单线程：geomean 1.57× vs torch t1</h2>
    <p>11 个 custom 形状，10/11 ≥ 1.0×；唯一落后的 c7 是 1.06 GFLOP 的极小形状。</p></div>
  <div class="grid-2">
    <div class="diagram">{svg_hbars(single_items, 2.4, width=860, title="MLIR / torch t1（比值，越高越好）", ref=1.0, value_fmt="{:.2f}", unit="×")}
      <div class="caption">红色虚线 = 1.0×；c7 0.83× 是结构性落后</div></div>
    <div class="card"><h3>怎么读</h3>
      <p>custom_0 33.3 ms / 127.7 GF，已接近单核 SVE 峰值（≈128 GF）。超过 SVE 口径的部分只能来自
        KBLAS 内部对矩阵引擎的使用。</p>
      <p class="small mt-10">c7：Ohw=400 只能取 Nwin=80，720 次 m80/n32/k288 小 GEMM。</p></div>
  </div>
  {table_single()}
</div></section>

<section class="section"><div class="wrap">
  <div class="section-head"><span class="eyebrow">性能 02</span><h2>M3 → M4-c：多核再涨 8%</h2>
    <p>两次全量扫描逐格取最小；t1/t8/t16/t32 取最佳档。</p></div>
  <div class="grid-2">
    <div class="diagram">{svg_grouped(SHAPES, [BEST_M3[s] for s in SHAPES], [BEST_M4[s] for s in SHAPES], 680, title="最佳 ms/iter（越短越好）")}
      <div class="caption">灰色 = M3；青色 = M4-c</div></div>
    <div class="diagram">{svg_hbars(gain_items, 22, width=860, title="M4-c 相对 M3 的最佳时间提升（%）", value_fmt="{:.1f}", unit="%")}
      <div class="caption">9 个形状提升；c5/c9 同配置持平（噪声内）</div></div>
  </div>
</div></section>

<section class="section"><div class="wrap">
  <div class="section-head"><span class="eyebrow">性能 03</span><h2>线程扩展：t16 之后为什么不动</h2>
    <p>宽度 20–40 时，t32 相对 t16 只涨 ≤17%；宽度 40 的 mc40 也不比 20 快。</p></div>
  <div class="diagram">{svg_multiline([
      ("custom_0", [M3["custom_0"][1], M3["custom_0"][2], M3["custom_0"][3], M3["custom_0"][4]],
       [M4["custom_0"][1], M4["custom_0"][2], M4["custom_0"][3], M4["custom_0"][4]]),
      ("custom_4", [M3["custom_4"][1], M3["custom_4"][2], M3["custom_4"][3], M3["custom_4"][4]],
       [M4["custom_4"][1], M4["custom_4"][2], M4["custom_4"][3], M4["custom_4"][4]]),
      ("custom_8", [M3["custom_8"][1], M3["custom_8"][2], M3["custom_8"][3], M3["custom_8"][4]],
       [M4["custom_8"][1], M4["custom_8"][2], M4["custom_8"][3], M4["custom_8"][4]])],
      width=880, height=250)}
    <div class="caption">灰色 = M3，青色 = M4-c；纵轴 ms/iter</div></div>
  <div class="callout warn mt-16"><strong>测量口径：</strong>t≥35 是本机慢模式，不作结论；
    2.0 GHz 固定频率；单 NUMA 节点；线程数 ≤32。</div>
</div></section>

<section class="section"><div class="wrap">
  <div class="section-head"><span class="eyebrow">性能 04</span><h2>11 形状：M4-c 完整数据</h2></div>
  {table_m4()}
  <div class="diagram mt-16">{svg_hbars(gf_items, 420, width=860, title="M4-c 最佳吞吐（GFLOPS）", value_fmt="{:.0f}", unit=" GF")}
    <div class="caption">最高 386 GF（custom_4）；原验收线 600 GF 仍未达</div></div>
</div></section>

<section class="section"><div class="wrap">
  <div class="section-head"><span class="eyebrow">性能 05</span><h2>观测到的三个口径问题</h2></div>
  <div class="grid-3">
    <div class="card accent-amber"><h3>t≥35 是慢模式</h3><p>node0 有 38 核，但 t≥35 稳定进入不可复现慢模式；线程扫描截到 t32。</p></div>
    <div class="card accent-indigo"><h3>GFLOPS 曾漏乘 batch</h3><p>旧公式漏了输出 batch，batch=4 时 GF 被低估 4×；ms 与加速比不受影响，脚本已修复。</p></div>
    <div class="card accent-teal"><h3>调优是噪声决策器</h3><p>同一形状跨进程可能选到不同参数；不要用单次运行比较两个版本。</p></div>
  </div>
</div></section>"""
    return page("性能", "SConvTransform 单线程与多核性能数据、图表与测量口径。", "performance.html", body,
                ("性能：数字与口径", "单线程 1.57×、多核 2.51×；每个数字都标注测量条件。", "Performance"))


def build_reproduce():
    body = f"""<section class="section"><div class="wrap">
  <div class="section-head"><span class="eyebrow">复现 01</span><h2>环境与构建</h2>
    <p>依赖 LLVM 19.1.7、kblas、libomp 与 ConvBench payload。</p></div>
  <div class="code-card"><div class="code-head"><span>构建</span><span>bash</span></div>
<pre><code>export LLVM=/home/z00889957/000App/clang+llvm-19.1.7-aarch64-linux-gnu
export PATH=$LLVM/bin:$PATH
export LD_LIBRARY_PATH=/home/z00889957/miniconda3/lib:$LLVM/lib:$LD_LIBRARY_PATH
cmake --build build -j 24</code></pre></div>
  <div class="callout note mt-16"><strong>运行期还需要：</strong>kblas 的 multi 目录、
    <code>$LLVM/lib/aarch64-unknown-linux-gnu</code>（libomp）、miniconda3/lib、<code>$LLVM/lib</code>。</div>
</div></section>

<section class="section"><div class="wrap">
  <div class="section-head"><span class="eyebrow">复现 02</span><h2>单线程与多核</h2></div>
  <div class="grid-2">
    <div class="code-card"><div class="code-head"><span>单线程</span><span>bash</span></div>
<pre><code>python3 scripts/convbench/run_convbench.py \\
  --csv scripts/convbench/samples/custom_conv.csv \\
  --transform test/lowering/sconv-blas-fastp.mlir \\
  --num 11 --runs 5 --warmup 3</code></pre></div>
    <div class="code-card"><div class="code-head"><span>多核（M4-c）</span><span>bash</span></div>
<pre><code>build/bin/sconv-opt \\
  -transform=test/lowering/sconv-blas-fastp-mc.mlir \\
  -sconv-parallelize -sconv-lower-parallel \\
  conv.mlir -o conv.mc.llvm.mlir

OMP_NUM_THREADS=16 numactl --cpunodebind=0 --membind=0 \\
  $LLVM/bin/mlir-cpu-runner conv.mc.llvm.mlir \\
  -e main -entry-point-result=void \\
  -shared-libs=...,libomp.so</code></pre></div>
  </div>
</div></section>

<section class="section"><div class="wrap">
  <div class="section-head"><span class="eyebrow">复现 03</span><h2>门禁：每一层都有失败测试先行</h2></div>
  <div class="diagram mb-16">{svg_gate_flow()}<div class="caption">构建 → ir → auto → correct → perf；任一层失败就回退改动</div></div>
  <div class="table-wrap"><table class="table">
    <thead><tr><th>门禁</th><th>阶段</th><th>检查什么</th><th>当前结果</th></tr></thead>
    <tbody>
      <tr><td><code>test_fastpack.py</code></td><td>ir / correct / perf</td><td>输入打包替换为 <code>pack_input_tile</code>；逐元素一致；custom_0 ≤55 ms</td><td class="up">PASS（40.3 ms）</td></tr>
      <tr><td><code>test_mk_auto.py</code></td><td>ir / perf</td><td>auto IR == 显式 IR；11 形状性能</td><td class="up">PASS</td></tr>
      <tr><td><code>test_multicore.py</code></td><td>ir / auto / correct / perf</td><td>marker 标对窗口循环；Nwin 表；调度不翻转；5 形状正确；custom_0 t16 ≤24 ms</td><td class="up">PASS（19.5 ms）</td></tr>
    </tbody></table></div>
  <div class="grid-3 mt-16">
    <div class="card accent-teal"><h3>ir</h3><p>显式 pin Nwin，检查并行循环的 step 等于 Nwin；覆盖两种 tiling 顺序。</p></div>
    <div class="card accent-indigo"><h3>auto</h3><p>11 形状的 Nwin 表 + CSA 调度一致性；c1/c2/9 必须保持 160。</p></div>
    <div class="card accent-amber"><h3>correct</h3><p>5 形状覆盖私化与共享两条 scratch 路径；另有 11/11（t8/t32）全量复核。</p></div>
  </div>
</div></section>

<section class="section"><div class="wrap">
  <div class="section-head"><span class="eyebrow">复现 04</span><h2>预期结果（2.0 GHz）</h2></div>
  <div class="grid-2">
    <div class="card"><h3>单线程</h3><p>custom_0 <b>33.3 ms / 127.7 GF</b>；11 形状 geomean <b>1.57×</b> vs torch t1。</p></div>
    <div class="card"><h3>多核（M4-c）</h3><p>custom_0 t16 <b>15.8 ms</b>；geomean <b>2.51×</b>；最高 <b>386 GF</b>（custom_4 @t32）。</p></div>
  </div>
  <div class="callout warn mt-16"><strong>注意：</strong>共享节点噪声约 ±4–7%；不要用单次运行比较两个版本。
    线程数 ≤32；t≥35 的慢模式不作为结论。</div>
</div></section>

<section class="section"><div class="wrap">
  <div class="section-head"><span class="eyebrow">复现 05</span><h2>文件地图</h2></div>
  <div class="table-wrap"><table class="table">
    <thead><tr><th>路径</th><th>作用</th></tr></thead>
    <tbody>
      <tr><td><code>include/SConv.td</code></td><td><code>sconv</code> transform op，含 <code>multicore</code> attr</td></tr>
      <tr><td><code>lib/SConv.cpp</code></td><td>tiling、marker、<code>selectAutoNwin*</code></td></tr>
      <tr><td><code>lib/LowerToPack.cpp</code></td><td>输入打包 lowering、<code>parallel_scratch</code> 标记</td></tr>
      <tr><td><code>lib/Parallelize.cpp</code></td><td><code>scf.parallel</code> 折叠、per-thread scratch、alloca_scope 清理</td></tr>
      <tr><td><code>test/lowering/sconv-blas-fastp*.mlir</code></td><td>串行 / 多核 / pinned tile 的 transform 序列</td></tr>
      <tr><td><code>scripts/convbench/test_*.py</code></td><td>IR / 正确性 / 性能门禁</td></tr>
    </tbody></table></div>
</div></section>"""
    return page("复现", "构建、运行、门禁与预期结果。", "reproduce.html", body,
                ("复现：命令与门禁", "从构建到单线程/多核运行，再到三层门禁。", "Reproduce"))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pages = {
        "index.html": build_index(),
        "principles.html": build_principles(),
        "pipeline.html": build_pipeline(),
        "timeline.html": build_timeline(),
        "multicore.html": build_multicore(),
        "performance.html": build_performance(),
        "reproduce.html": build_reproduce(),
    }
    for name, content in pages.items():
        (OUT / name).write_text(content, encoding="utf-8")
    print(f"wrote {len(pages)} pages to {OUT}")


if __name__ == "__main__":
    main()

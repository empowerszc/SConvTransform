# bench_vs_torch

Runs the project's convolution against **PyTorch `conv2d`** and emits a machine-readable
`out/data.json` used by `docs/PERFORMANCE_VS_PYTORCH.md`.

| 文件 | 作用 |
|------|------|
| `torch_baseline.py` | PyTorch conv2d 基线（18 个配置），打印 `name,flops,ms,gflops` |
| `compare_vs_pytorch.py` | 采集 `torch` / `after`(/`before`) 三方 GFLOPS，写 `out/data.json` |

## 用法

```bash
source env.sh                                  # clang + kblas
export SCONV_ROOT=$PWD
export TORCH_PYTHON=/home/z00889957/miniconda3/envs/torch210_env/bin/python

# after = native_impl (并行 Bc + 自适应 Nwin/Nf)；custom_9/10 用 SME 探针(kernel-only)
python3 scripts/bench_vs_torch/compare_vs_pytorch.py --threads 1,2,4,8,16,32,36

# 可选：与优化前的构建对比（另建一份未优化的 native_impl）
python3 scripts/bench_vs_torch/compare_vs_pytorch.py \
  --before-binary /tmp/native_common/build/sconv_bench \
  --threads 1,2,4,8,16,32,36
```

参数：`--runs/--warmup/--repeats`（默认 8/3/3，取 best-of-repeats）、`--numa 0`（设为 `''` 关闭绑定）、
`--out`（默认本目录 `out/`）。

## 口径说明

- `after`/`before` 为端到端（含 im2col 打包与解包）。`custom_9/custom_10` 优先用 **SME 端到端**
  （`sme/e2e.out`，含 shim 权重打包/pad/unpad）；若不存在则退回 kernel-only 探针 `sme/direct.out`。
  构建端到端基准：`cd sme && make e2e.out`。
- `torch` 为端到端 `torch.nn.functional.conv2d`（`torch.set_num_threads(t)`）。
- 运行需 `numactl`（可用 `--numa ''` 关闭）与可用的 torch 环境（`TORCH_PYTHON`）。

`out/data.json` 结构：`{ "torch": {t:{name:gflops}}, "after": {...}, "before": {...} }`。
`docs/PERFORMANCE_VS_PYTORCH.md` 的矩阵即由此文件汇总生成。

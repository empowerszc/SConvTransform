# 可视化文档站（离线）

直接用浏览器打开 `index.html` 即可，无需服务器、无外部 CDN 依赖。

- 由 `scripts/docs/build_site.py` 生成，**不要手改**这里的 HTML/CSS/JS；
  改内容或样式后重新运行：

  ```bash
  python3 scripts/docs/build_site.py
  ```

- 数据来源（2.0 GHz，单 NUMA node0）：
  - `docs/PERFORMANCE_VS_PYTORCH.md` 附录 C.1 / C.2
  - `docs/OPTIMIZATION_PLAN_2026-11-18.md` §8–§9
  - `docs/OPTIMIZATION_JOURNEY_2026-11-19.md`
  - `docs/OPTIMIZATION_DESIGN_AND_RESULTS.md`

页面（8 页，21 张内联 SVG 示意图）：

| 页面 | 内容 |
|---|---|
| `index` | 总览：指标、两条路径、优化阶梯、三条最贵经验 |
| `principles` | 原理：三阶段 roofline、im2col 图解、循环嵌套、tile→GEMM 映射 |
| `pipeline` | 管线：transform/pass 顺序、M1 与并行 lowering 的 IR 前后对照 |
| `timeline` | 演进：时间线、custom_0 毫秒阶梯（156.6→14.4）、测量协议、被证伪方向 |
| `multicore` | 多核：并行链、宽度 vs 加速比散点、M4-c 决策树、完整数据 |
| `performance` | 性能：单线程/多核图表、6 形状线程曲线、测量口径 |
| `reproduce` | 复现：环境、命令、门禁流程图、预期结果 |
| `glossary` | 术语：Nwin/Nf/Nc、IS/WS、CSA、roofline 等全部缩写 |

功能：响应式布局、深色模式（跟随系统偏好，右上角可切换并记忆）、
阅读进度条、回顶按钮、打印样式。

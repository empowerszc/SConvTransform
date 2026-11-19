# 可视化文档站（离线）

直接用浏览器打开 `index.html` 即可，无需服务器、无外部 CDN 依赖。

- 由 `scripts/docs/build_site.py` 生成，**不要手改**这里的 HTML/CSS/JS；
  改内容或样式后重新运行：

  ```bash
  python3 scripts/docs/build_site.py
  ```

- 数据来源（2.0 GHz，单 NUMA node0）：
  - `docs/PERFORMANCE_VS_PYTORCH.md` 附录 C.1 / C.2
  - `docs/OPTIMIZATION_PLAN_2026-11-18.md` §9
  - `docs/OPTIMIZATION_JOURNEY_2026-11-19.md`
  - `docs/OPTIMIZATION_DESIGN_AND_RESULTS.md`

页面：`index` / `principles` / `pipeline` / `timeline` / `multicore` /
`performance` / `reproduce`。每页含内联 SVG 示意图，支持响应式与打印。

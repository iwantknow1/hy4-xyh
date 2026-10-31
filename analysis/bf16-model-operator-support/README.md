# HY4 BF16 权重模型算子功能支持分析

本报告对象是 **BF16 权重模型所使用的算子**，不是仅支持 BF16 I/O dtype 的算子目录。FP32、整数、BOOL、AI_CPU 与 dtype 未报告的路径均纳入模型功能执行证据。引擎、dtype、功能状态分别列示。

- [Excel：支持口径、模块、逐 kernel、dtype/shape、全部原始证据](bf16-operator-coverage.xlsx)
- [阅读报告 PDF](report.pdf) / [HTML（下载后打开）](report.html)
- [逐 kernel 汇总 CSV](operator-summary.csv) / [4760 条执行证据 CSV](kernel-evidence.csv)
- [dtype/shape 矩阵 CSV](dtype-shape-matrix.csv) / [主要模块 CSV](module-matrix.csv)
- [完整 JSON，含来源 SHA256、CSV 路径与行号](report.json)

## 结果与范围

成功减层 2L/TP2，物理卡 2、3；请求输入 32768、输出 4096 token。每 rank 在延迟 2048 个 worker 步后采集中段 decode 10 步。两卡共 4760 条任务记录、89 个 kernel 名称、56 个 CANN/自定义 Type；98 个 kernel/引擎/dtype 分类组合、272 个 rank/dtype/shape 组合。任务/kernel 数不等于前端算子数量。

4760 条均是成功轮次中的已观测执行记录，不以 BF16 I/O 筛选功能支持。1900 条明确记录 BF16 I/O，2360 条记录其他 dtype，500 条未报告 dtype；这些是元数据分组，不是支持/不支持分组。

引擎调用数：AI_VECTOR_CORE 3840、AI_CPU 220、COMMUNICATION 40、MIX_AIC 240、AI_CORE 360、MIX_AIV 60。AI_CPU 包含 80 次明确 BF16 I/O 的 Index 和 140 次 HCCL 启动任务；AI_CPU 是设备侧 CPU，不能因未在 AI Core 执行就认定不支持。通信任务单列，不当作算术算子统计。

已有 GEMM/Grouped GEMM、RMSNorm、稀疏 Attention、Indexer、RoPE/KV cache、SwiGLU、路由与通信等执行证据。iHC/Triton RoPE dtype 未报告，但已有执行记录，功能证据有效。

**覆盖限制：** 本采集不包含完整视觉/prefill 窗口、MTP、全量共享 Indexer 和四机专家通信。全量精度已另行完成，但没有全量 kernel Profiling。本报告不计算“全模型 100% 支持率”，也不做逐 kernel 高精度参考误差认证；未采到不等于不支持。

跨设备 kernel duration 相加不是墙钟时间、batch 延时或端到端吞吐。weighted_attainment 是独立经验效率指标，不是算子支持率。

## 可复核生成

原始数据位于 [reduced/profiles](../../reduced/profiles)，来源文件哈希保存在 JSON 与 Excel Sources 页。Python 生成器使用标准库；Excel 导出依赖 npm `xlsx`（本交付使用 0.18.5）。

```sh
# 在 uv 管理的 Python 环境中运行，默认 archive root 为当前仓库
.venv/bin/python analysis/bf16-model-operator-support/analyze.py
# 使用已安装的 xlsx；也可通过 HY4_XLSX_MODULE 指定模块绝对路径
node analysis/bf16-model-operator-support/export.cjs
```

本次仅离线分析既有成功数据，没有启动模型、NPU 测试或裁判请求。CSV 带 UTF-8 BOM，便于 Windows Excel 打开。

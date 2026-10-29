# 减层 Profiling 配置和脚本

## 文件入口

- [实际启动入口](../reduced/scripts/run_profile.py)：当时传给采集器的完整 argv 和额外环境变量。
- [实际参数快照](../reduced/logs/resume-exclusive/profile-parameter-snapshot.json)：包含实际 PATH、LD_LIBRARY_PATH、容量预检和命令。
- [完整采集 manifest](../reduced/logs/resume-exclusive/profile-final-manifest.json)：服务配置、采集窗口、bench 结果、设备/rank 与分析状态。
- [验收报告](../reduced/logs/resume-exclusive/profile-final.json)：实际 token 数和十步证据。
- [核心采集器](../reduced/tooling/skills/inference-model-skills/performance-optimization/ascend-profiling-collection/scripts/collect_torch_profile_case.py)：负责拉起、控制采集、执行 bench、停止服务和解析。
- [服务工具](../reduced/tooling/skills/inference-model-skills/deployment/vllm-ascend-serving/scripts/)：采集器依赖的 serve_start/stop、SSH 和公共工具。
- [原始 Profiling](../reduced/profiles/)：两个 rank 的 trace、kernel_details.csv 等。
- [bench stdout](../reduced/logs/resume-exclusive/profile-final-benchmark.stdout.log)、[bench stderr](../reduced/logs/resume-exclusive/profile-final-benchmark.stderr.log)。

核心采集器与服务工具最初未随入口一起归档，本次从四机复用镜像的 `/codex/tooling/skills/` 补齐；该镜像继承了减层阶段实际适配后的工具。保留原目录层级和源码，不用原仓库其他版本替代。文件校验值见 [profiling-tooling.json](../delivery/profiling-tooling.json)。

## 当时使用的配置

| 项目 | 实际值 |
| --- | --- |
| 模型 | HY4 2L 减层权重，`/models/Hy4-2L` |
| 并行和设备 | TP2、DP1、EP 开启；物理卡 2、3 |
| 模态 | 一张 256×256 图片＋文本 |
| 总输入 | 32768 token；文本部分 32673，图片/模板校准开销 95 |
| 输出 | 4096 token，ignore EOS |
| 并发 / 请求数 | 1 / 1 |
| 上下文长度 | 36864 |
| KV | 固定 8 GiB/卡 |
| 图模式 | FULL_DECODE_ONLY；capture size 1 |
| 采集窗口 | 延迟 2048 个 worker 步，采集中段 10 步，ignore frontend |
| runner | `VLLM_USE_V2_MODEL_RUNNER=0` |
| bench | `vllm bench serve`，通过 uv 环境的 Python 模块入口执行 |

没有另外传 sampling 参数，沿用当时 benchmark/runtime 默认值，不能写成显式 temperature=0。记录的十步依据是各 rank 的 MODEL_EXECUTE 标记，不把 step_trace_time.csv 聚合行数当成推理步数。

## 复用前准备

这是实测命令和工具归档，不是已经在新机器验证的一键脚本。

1. 使用减层验证镜像及相同模型结构，准备实际减层权重和 tokenizer。
2. 修改入口中的主机、容器名、设备、CANN、模型路径和自己的 SSH key 路径，避免使用别人正在运行的容器。
3. 保持工具目录结构：把本仓库 `reduced/tooling/` 的内容放到独立运行目录，例如镜像中的 `/codex/tooling/`。采集器通过相对层级查找 serving 工具，并在 tooling 下创建 `.workspace-local`；该运行副本须可写。
4. 用 uv 管理的环境及 `.venv/bin/python` 运行 `run_profile.py`，并准备 `/codex/logs/resume-exclusive/`。入口依赖已加载的 CANN PATH 和 LD_LIBRARY_PATH。
5. SSH key 自行挂载并限制权限。本仓库不提供私钥、服务器密码或 API key。

脚本已包含当时的 Python 模块入口和 CANN PATH 适配，不需要再次执行历史 `fix_*` 修改脚本。减层采集结果只证明功能与采集流程，不能外推为全量模型性能。本次补充归档没有启动服务、bench 或新的 Profiling。

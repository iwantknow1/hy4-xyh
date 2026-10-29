# 开箱经过与关键问题

## 1. 隔离开发、减层功能验证

使用独立目录、源码副本和容器，与 DeepSeek 的工作分开。先审计模型结构与 checkpoint，再验证两个 HTTP 接口、sleep/wake 和权重切换。减层随机输出只用于功能、数值有效性和一致性，不用于精度。

reload 不只需要把数值写进去，还要走包装过的 weight_loader，保留加载计数、运行时参数存储和图引用。sink 原分支 TP 切片后直接 copy_ 绕过 loader；修改为对原参数调用 loader。meta 专家分支原先提前返回，导致加载记账缺失，恢复合法 loader 调用并保留严格完整性检查。

HCCL/IPC 按 chunk 分别 load_weights 会把部分权重误当完整 checkpoint。HCCL 改为完整流一次加载，明确默认 stream 和缓冲生命周期；IPC 在 finish 时提交完整事务，每个 chunk 重新导出引用计数。可选 CPU staging 是真实传输后的暂存方式，不是绕过 HCCL/IPC；需要按全量主机内存重新评估。

6L TP2 finalize 时分配失败：在 finalize 前清理空闲 allocator cache，为延迟 materialization 腾空间。实测释放约 19.76 GiB reserved，活跃参数不变，相同 TP2/8GiB KV 配置 eager 和图模式复验成功。仓库保留成功的 98 项单元回归、ruff、A-B-A、重复重载及两接口逐题输出；失败轮次的数据和日志不上传。

详见 [修复说明](../fixes/CODE-CHANGES.zh-CN.txt)、[减层完整报告](../reduced/model_openbox/hy4/VERIFICATION-REPORT.zh-CN.md)。

## 2. 并行、采集与镜像

分别补测 SP、单机 PP、DP 并发、HTTP/进程内 R3、MTP 和 pause/resume。PP 对照必须固定 TP，首轮混入 TP 差异导致对照失败，调整为相同 TP 后通过；失败数据不上传。R3 进程内入口使用 spawn；不能在初始化 NPU 后 fork。减层 MTP 只有功能通过，接受数为 0，不宣称加速。

DCP/PCP 分别核实：HY4 sink 实现明确要求二者都为 1；没有删除断言或将通用 DSA 的 CP 能力当成 HY4 已支持。

多模态 32768 输入/4096 输出、中段十步采集成功；算子分类和单卡 Excel 独立重算通过。测算仅适用于减层采集。原 Excel 工具因缺依赖失败，后通过公开库备用导出器完成；只上传成功导出物及其校验。最终减层镜像另有不挂源码的 HCCL 实机复验。

## 3. 四机全量阶段

按用户要求跳过双机，四台 A5、共 32 卡，使用官方全量 BF16 权重：78 层、246 分片、2655 张量。权重索引、头部、容量及四节点源码哈希先审计。

早期 TP8/PP4 与 DP4 尝试遇到 Gloo malformed-header。管理网为 1Gb，32 进程压力测试出现载荷校验错误；200Gb 路径压力测试通过。切换到 `enp39s0f0` 和 `197.13.49.1–197.13.52.1`。证据支持该配置下恢复稳定，但没有证明故障的唯一底层根因，也没有换网卡后重测旧 PP。

DP4 后续初始化超过默认 600 秒：采用一个 API server，engine ready timeout 3600 秒；不把清理阶段的 Gloo 断连写成最初原因。

图片请求出现 slot-mapping TypeError：我们自己的 PATH 覆盖了 CANN 编译器路径，Triton 选择了占位普通函数。恢复 `/usr/local/Ascend/cann-9.1.0/bin` 后四节点实际 kernel 验证通过，run07 文本/图片及完整数据生成成功。该问题通过启动环境修正，没有增加模型源码补丁。

正式配置为 TP8、DP4、PP1、EP32、MTP3、max_num_seqs=64、max_model_len=147456、max_num_batched_tokens=8192、固定 KV 14GiB/卡、FULL_DECODE_ONLY。不能将 DP4 错认为每个 DP 都复制全部专家；专家并行覆盖 32 卡。

## 4. 评测与交付

GPQA 198 题按参考原始顺序/选项/gold 运行，输出长度 102400，最终 185/198；旧部分请求失败的分数无效，其数据和日志不上传。

Hallusion 原串行生成脚本的 1200 秒 HTTP 超时不足以容纳 102400 token。只增加 transport timeout 参数并设置 7200，默认仍为 1200；未改变采样、请求顺序或长度。正式 1129 题从头完整生成，不混用两轮输出；最初未完成轮次不上传。

模型回答全部完成后停止我方四个推理容器、释放 NPU；裁判在 25 的 CPU 容器通过 DeepSeek API 运行。原评分脚本哈希、提示词和统计逻辑未改，2258 次串行判断完成。逐题缓存落盘共享存储；重启前另外做过本地快照。

全量评分于 2026-10-29 12:33 CST 完成，生成 skill README、逐项 JSON、最终报告和镜像元数据。下一轮用户要求使用有限并发：这是后续计划，本轮没有并发评分；应保持阶段依赖、结果顺序和缓存写入安全，并对限流/失败做完整重试。

镜像由我方推理容器保存，CPU 裁判容器没有打包；权重和驱动继续外部挂载。镜像中的便携启动 helper 做了语法检查，本轮实机验证使用等效的原 skill launcher；没有将 helper 的再次部署写成已执行。

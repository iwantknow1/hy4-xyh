# 四机全量审计入口

本文件为交付时补充的证据索引；原始审计实际在启动前执行，未重新运行模型。

- 模型：官方全量 BF16，78 层、246 safetensors 分片、2655 张量、649 vision 张量。
- 权重 payload：1562453830516 字节；文件合计 1562454136204 字节。
- [原始权重审计](../../logs/full-model-audit.json)、[容量预算](../../logs/tp8-dp4-mtp3-64-capacity.json)。
- 权重索引 SHA256：`605941bd325eec75ffe918932217e526b687f96f78cfe78eb2b5ae188779f863`。
- 实际模型/代码/配置、前置修复继承及版本见 [manifest](manifest.json) 和 [README](README.md)。
- 实际每 rank 报告权重内存 55.1109 GB，KV capacity 155520 tokens；四机正式评测无 OOM。
- 四节点源码核对见 `full/logs/node*/source-hash-audit.json`（按仓库根目录定位）。
- 未在此阶段外推全量其他功能、PP 或 CP 通过。

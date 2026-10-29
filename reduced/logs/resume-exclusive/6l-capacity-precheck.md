# 6-layer HCCL capacity precheck
Allowed devices: 2-7 only; 0/1 belong to colleagues.
Actual checkpoint scan: 810 tensors, 12 shards, 106403732980 tensor bytes (99.1 GiB), largest tensor 12 GiB. Host RAM available approximately 2.2 TiB.
HCCL layout: TP2 inference on physical 6/7, separate producer on physical 5. Fixed KV cache 8 GiB/card. CPU staging explicitly enabled; full received checkpoint is not retained on NPU.
Per-worker weight baseline approximately 49.55 GiB plus replicated small tensors. Allow approximately 51 GiB persistent weights, 15 GiB for local expert materialization/transposes, 5 GiB runtime, graph allowance <1 GiB, and 8 GiB KV in generation phase. Conservative combined estimate approximately 80 GiB, below ~91.75 GiB usable. Packed receiver batches are copied to CPU and released before deferred processing; allocator scratch cache is emptied before KV wake.
Producer packed batch can exceed the 12 GiB threshold by the next tensor; bound <24 GiB. Source plus packed allocations <48 GiB, separately on device 5. Inference receives no producer-resident model.
Previous old-host 6L TP2 direct checkpoint reload passed; the new CPU staging removes raw full-checkpoint NPU retention that caused the prior HCCL OOM. This is a capacity estimate, not acceptance. Record any actual capacity failure and stop instead of reducing settings blindly.

# IPC TP2 capacity precheck (functional 2-layer model)
Devices 6 and 7, each about 91.75 GiB usable. Producers colocated with inference workers.
Observed TP2 model loading 14.19 GB per card; sleep frees 36.95 GiB including fixed 8 GiB KV cache. Weight wake therefore restores approximately 29 GiB, plus runtime approximately 3 GiB.
Checkpoint largest tensor 12 GiB. Producer source plus packed storage approximately 24 GiB; receiver clone up to 12 GiB (CPU copy releases it per chunk). During finalization allow 18 GiB for layer materialization and conversion; KV remains asleep until producers release buffers.
Conservative overlapping budget: 29+3+24+12+18 = 86 GiB per card, below 91.75 GiB. Actual phase peaks must be checked, not treated as proven by this estimate.
Host RAM above 2 TiB; deferred checkpoint CPU staging approximately 24.8 GiB per worker.
Configuration inherits TP2 eager baseline, EP, RL mode, fixed KV 8 GiB. Only npu_ipc backend and trusted serialization flag added.

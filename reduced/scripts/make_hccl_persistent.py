from pathlib import Path
p=Path('/codex/scripts/skill-scripts/weight-update/test_hccl_packed_cpu_stage.py')
s=p.read_text(); start=s.index('    transfer_group = None\n',s.index('def run_validation')); end=s.index('\n\n\ndef main',start)
s=s[:start]+'''    transfer_group = None
    report["rounds"] = []
    try:
        for round_index in range(3):
            validation.sleep_service()
            if transfer_group is None:
                transfer_group = initialize_hccl_transfer(
                    arguments, client, HCCLWeightTransferEngine, inference_worker_count,
                )
            validation.wake_weight_buffers()
            client.start_weight_update()
            report["transfer"] = transfer_checkpoint_weights(arguments, weight_source, client, transfer_group)
            client.finish_weight_update()
            report["checks"]["checkpoint_stream_transfer"] = True
            report["checks"]["init_start_update_finish"] = True
            gc.collect()
            torch.npu.synchronize()
            torch.npu.empty_cache()
            validation.wake_kv_cache()
            validation.verify_updated_outputs()
            report["rounds"].append({"round": round_index + 1, "outputs_identical": report["checks"]["outputs_identical"]})
            print(f"[round {round_index + 1}] passed", flush=True)
    finally:
        validation.record_incomplete_update()
''' + s[end:]
out=p.with_name('test_hccl_packed_persistent.py');out.write_text(s)

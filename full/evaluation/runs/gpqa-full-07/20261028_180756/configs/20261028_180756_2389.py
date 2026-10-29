cli_args=dict(
    {'agent': None,
    'agent_api_key': None,
    'agent_dataset_path': None,
    'agent_deps': None,
    'agent_env': None,
    'agent_import_path': None,
    'agent_kwarg': None,
    'api_base': None,
    'api_key': None,
    'batch_size': None,
    'cfg_time_str': '20261028_180756',
    'config': '/task/evaluation/gpqa-full-config-07.py',
    'config_dir': 'configs',
    'custom_dataset_data_type': None,
    'custom_dataset_infer_method': None,
    'custom_dataset_meta_path': None,
    'custom_dataset_path': None,
    'dataset': None,
    'datasets': None,
    'debug': True,
    'delete/__no_delete': None,
    'dir_time_str': '20261028_180756',
    'disable_verification': None,
    'dry_run': False,
    'dump_eval_details': True,
    'dump_extract_rate': True,
    'env_file': None,
    'environment': None,
    'exclude_task_name': None,
    'extra_docker_compose': None,
    'force_build/__no_force_build': None,
    'generation_kwargs': None,
    'host_ip': None,
    'host_network': None,
    'host_port': None,
    'include_task_name': None,
    'max_num_workers': 1,
    'max_out_len': None,
    'max_retries': None,
    'max_workers_per_gpu': 1,
    'merge_ds': False,
    'mode': 'all',
    'model': None,
    'model_name': None,
    'models': None,
    'monitor_port': 0,
    'n_attempts': None,
    'n_concurrent': None,
    'n_tasks': None,
    'num_prompts': 198,
    'num_warmups': 0,
    'path': None,
    'pressure': False,
    'pressure_time': 15,
    'purge_exception_cases': False,
    'quiet': None,
    'request_rate': None,
    'response_anomaly': False,
    'response_anomaly_payload_retention': None,
    'retry': None,
    'reuse': None,
    'search': False,
    'spec_decode': False,
    'summarizer': None,
    'timeout_multiplier': None,
    'trust_remote_code': None,
    'url': None,
    'work_dir': None,
    'yes': None})
datasets=[
    dict(abbr='GPQA_diamond',
        eval_cfg=dict(
            evaluator=dict(
                type='ais_bench.benchmark.datasets.gpqa.GPQAEvaluator'),
            pred_postprocessor=dict(
                options='ABCD',
                type='ais_bench.benchmark.utils.postprocess.text_postprocessors.first_option_postprocess')),
        infer_cfg=dict(
            inferencer=dict(
                type='ais_bench.benchmark.openicl.icl_inferencer.icl_gen_inferencer.GenInferencer'),
            prompt_template=dict(
                template=dict(
                    round=[
                        dict(prompt='{prompt}',
                            role='HUMAN'),
                        ]),
                type='ais_bench.benchmark.openicl.icl_prompt_template.icl_prompt_template.PromptTemplate'),
            retriever=dict(
                prompt_template=dict(
                    template=dict(
                        round=[
                            dict(prompt='{prompt}',
                                role='HUMAN'),
                            ]),
                    type='ais_bench.benchmark.openicl.icl_prompt_template.icl_prompt_template.PromptTemplate'),
                type='ais_bench.benchmark.openicl.icl_retriever.icl_zero_retriever.ZeroRetriever')),
        path='/task/evaluation/gpqa-exact-prompts.jsonl',
        reader_cfg=dict(
            input_columns=[
                'prompt',
                ],
            output_column='gold',
            test_range='[:198]'),
        type='ais_bench.benchmark.datasets.custom.CustomDataset'),
    ]
eval=dict(
    partitioner=dict(
        out_dir='/task/evaluation/runs/gpqa-full-07/20261028_180756/results/',
        type='ais_bench.benchmark.partitioners.naive.NaivePartitioner'),
    runner=dict(
        debug=True,
        max_num_workers=1,
        max_workers_per_gpu=1,
        task=dict(
            cal_extract_rate=True,
            dump_details=True,
            type='ais_bench.benchmark.tasks.openicl_eval.OpenICLEvalTask'),
        type='ais_bench.benchmark.runners.local.LocalRunner'))
infer=dict(
    partitioner=dict(
        out_dir='/task/evaluation/runs/gpqa-full-07/20261028_180756/predictions/',
        type='ais_bench.benchmark.partitioners.naive.NaivePartitioner'),
    runner=dict(
        debug=True,
        max_num_workers=1,
        max_workers_per_gpu=1,
        task=dict(
            type='ais_bench.benchmark.tasks.openicl_api_infer.OpenICLApiInferTask'),
        type='ais_bench.benchmark.runners.local.LocalRunner'))
models=[
    dict(abbr='hy4-gpu-aligned',
        attr='service',
        batch_size=256,
        generation_kwargs=dict(
            chat_template_kwargs=dict(
                reasoning_effort='high'),
            temperature=0.9,
            top_k=-1,
            top_p=1.0),
        host_ip='177.11.25.1',
        host_port=18080,
        max_out_len=102400,
        model='hy4',
        path='/mnt/share/z30059446/hy4_safetensors',
        pred_postprocessor=dict(
            think_end_token='</think:opensource>',
            think_start_token='<think:opensource>',
            type='ais_bench.benchmark.utils.postprocess.model_postprocessors.extract_non_reasoning_content'),
        retry=2,
        stream=False,
        trust_remote_code=True,
        type='ais_bench.benchmark.models.api_models.vllm_custom_api_chat.VLLMCustomAPIChat'),
    ]
response_anomaly=dict(
    enabled=False,
    msprobe_config_path=None,
    payload_retention='anomalies',
    payload_storage=dict(
        compression='zstd',
        compression_level=3,
        format='jsonl',
        rows_per_shard=2000))
summarizer=dict(
    dataset_abbrs=[
        'GPQA_diamond',
        ],
    summary_groups=[
        ],
    type='ais_bench.benchmark.summarizers.default.DefaultSummarizer')
work_dir='/task/evaluation/runs/gpqa-full-07/20261028_180756'
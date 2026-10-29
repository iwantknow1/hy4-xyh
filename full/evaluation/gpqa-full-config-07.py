
from ais_bench.benchmark.datasets.custom import CustomDataset
from ais_bench.benchmark.datasets.gpqa import GPQAEvaluator
from ais_bench.benchmark.models import VLLMCustomAPIChat
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.utils.postprocess.model_postprocessors import extract_non_reasoning_content
from ais_bench.benchmark.utils.postprocess.text_postprocessors import first_option_postprocess

summarizer = dict(attr="accuracy", dataset_abbrs=["GPQA_diamond"], summary_groups=[])

datasets = [dict(
    abbr="GPQA_diamond",
    type=CustomDataset,
    path="/task/evaluation/gpqa-exact-prompts.jsonl",
    reader_cfg=dict(input_columns=["prompt"], output_column="gold"),
    infer_cfg=dict(
        prompt_template=dict(type=PromptTemplate, template=dict(round=[dict(role="HUMAN", prompt="{prompt}")])),
        retriever=dict(type=ZeroRetriever), inferencer=dict(type=GenInferencer),
    ),
    eval_cfg=dict(
        evaluator=dict(type=GPQAEvaluator),
        pred_postprocessor=dict(type=first_option_postprocess, options="ABCD"),
    ),
)]
models = [dict(
    attr="service", type=VLLMCustomAPIChat, abbr="hy4-gpu-aligned",
    path="/mnt/share/z30059446/hy4_safetensors", model="hy4", stream=False,
    host_ip="177.11.25.1", host_port=18080, max_out_len=102400, batch_size=256,
    retry=2, trust_remote_code=True,
    generation_kwargs=dict(
        temperature=0.9, top_p=1.0, top_k=-1,
        chat_template_kwargs=dict(reasoning_effort="high"),
    ),
    pred_postprocessor=dict(
        type=extract_non_reasoning_content,
        think_start_token="<think:opensource>", think_end_token="</think:opensource>",
    ),
)]
work_dir = "/task/evaluation/runs/gpqa-full-07"

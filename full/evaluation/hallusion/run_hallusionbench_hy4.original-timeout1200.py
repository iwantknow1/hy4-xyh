"""Sequential OpenAI-compatible HallusionBench inference; standard library only."""
import argparse
import base64
import json
import mimetypes
import os
from pathlib import Path
import urllib.request


def image_path(root, row):
    p = Path(root) / row["filename"]
    if not p.is_file():
        p = Path(root) / "hallusion_bench" / row["filename"]
    if not p.is_file():
        raise FileNotFoundError(p)
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="Official HallusionBench.json")
    ap.add_argument("--root", required=True, help="Directory containing hallusion_bench/")
    ap.add_argument("--api", default="http://177.11.21.1:18001/v1")
    ap.add_argument("--model", default="hy4")
    ap.add_argument("--output", required=True, help="New experiment directory")
    ap.add_argument("--limit", type=int, help="Run the first N samples; omit or use 0 for the full dataset")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.6)
    args = ap.parse_args()
    rows = json.loads(Path(args.data).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("Expected official dataset JSON array")
    if args.limit is not None and args.limit < 0:
        ap.error("--limit must be 0 (full dataset) or a positive integer")
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        raise ValueError("Empty dataset")
    # Validate every required image before sending any requests.
    for row in rows:
        if int(row["visual_input"]) != 0:
            image_path(args.root, row)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    results, tokens, reasons = [], [], []
    with (out / "responses.jsonl").open("w", encoding="utf-8") as log:
        for i, row in enumerate(rows):
            content = []
            if int(row["visual_input"]) != 0:
                p = image_path(args.root, row)
                mime = mimetypes.guess_type(str(p))[0] or "image/png"
                uri = "data:" + mime + ";base64," + base64.b64encode(p.read_bytes()).decode()
                content.append({"type": "image_url", "image_url": {"url": uri}})
            content.append({"type": "text", "text": row["question"]})
            body = dict(model=args.model, messages=[dict(role="user", content=content)],
                        max_tokens=args.max_tokens, temperature=args.temperature, stream=False)
            headers = {"Content-Type": "application/json"}
            if os.environ.get("HY4_API_KEY"):
                headers["Authorization"] = "Bearer " + os.environ["HY4_API_KEY"]
            request = urllib.request.Request(args.api.rstrip("/") + "/chat/completions",
                                             data=json.dumps(body).encode(), headers=headers)
            # Direct access to the local cluster API; do not use public HTTP proxies.
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            try:
                with opener.open(request, timeout=1200) as response:
                    raw = json.load(response)
                log.write(json.dumps(dict(index=i, sample=row, response=raw), ensure_ascii=False) + "\n")
                log.flush()
                choice = raw["choices"][0]
                text = choice["message"].get("content")
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(f"Empty final answer at sample {i}")
                # Preserve raw output above; score only the final answer after HY4 reasoning.
                for marker in ("</think:opensource>", "</think>"):
                    if marker in text:
                        text = text.rsplit(marker, 1)[1].strip()
                if not text or "<think" in text:
                    raise ValueError(f"Incomplete/unrecognized reasoning output at sample {i}")
                item = dict(row, model_prediction=text)
                results.append(item)
                n = raw.get("usage", {}).get("completion_tokens")
                tokens.append(n if isinstance(n, int) else None)
                reasons.append(choice.get("finish_reason"))
                (out / "partial_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"{i+1}/{len(rows)} completion_tokens={n} finish_reason={reasons[-1]}", flush=True)
            except Exception as exc:
                (out / "failure.json").write_text(json.dumps(dict(index=i, error=str(exc)), ensure_ascii=False), encoding="utf-8")
                raise
    known = [n for n in tokens if n is not None]
    summary = dict(samples=len(rows), token_usage_coverage=len(known),
                   avg_completion_token=sum(known)/len(known) if len(known)==len(rows) else None,
                   length_truncated=sum(r=="length" for r in reasons),
                   note="Inference only; accuracy requires the matching evaluator")
    (out / "HallusionBench_result.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

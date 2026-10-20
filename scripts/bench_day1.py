"""第1回 宿題（発展）: 応答時間とトークン数を複数モデル・質問で比較する。

使い方:
    uv run python scripts/bench_day1.py

モデル × 質問 × 繰り返し回数 の組み合わせで generate_content を呼び、
応答時間（秒）と usage_metadata のトークン数を記録して表にして出す。
無料枠のレート制限（429）に当たった場合も記録し、待機して継続する。
"""

import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from google import genai
from google.genai import errors

from app.config import get_api_key

MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-2.5-flash-lite",
]

QUESTIONS = [
    "LoRAとは何ですか？一言で答えてください。",
    "AIE2026の講座の宿題にある「3レベル」とは何ですか？",
    "日本の都道府県を5つ挙げ、それぞれの県庁所在地を答えてください。",
]

REPEAT = 3
SLEEP_SEC = 3  # 無料枠のレート制限を避ける待機
RETRY_WAIT_SEC = 60


@dataclass
class Result:
    model: str
    question: str
    run: int
    latency_sec: float | None
    prompt_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    error: str | None


def call_once(client, model: str, question: str) -> Result:
    start = time.perf_counter()
    try:
        resp = client.models.generate_content(model=model, contents=question)
        latency = time.perf_counter() - start
        u = resp.usage_metadata
        return Result(
            model=model,
            question=question,
            run=0,
            latency_sec=round(latency, 2),
            prompt_tokens=getattr(u, "prompt_token_count", None),
            output_tokens=getattr(u, "candidates_token_count", None),
            total_tokens=getattr(u, "total_token_count", None),
            error=None,
        )
    except errors.APIError as e:
        latency = time.perf_counter() - start
        return Result(
            model=model,
            question=question,
            run=0,
            latency_sec=round(latency, 2),
            prompt_tokens=None,
            output_tokens=None,
            total_tokens=None,
            error=f"{e.code}: {e.message[:80]}",
        )


def main() -> None:
    client = genai.Client(api_key=get_api_key())
    results: list[Result] = []

    for model in MODELS:
        for question in QUESTIONS:
            for i in range(REPEAT):
                r = call_once(client, model, question)
                if r.error and r.error.startswith("429"):
                    print(f"  [429] {model}: {RETRY_WAIT_SEC}秒待って再試行", flush=True)
                    time.sleep(RETRY_WAIT_SEC)
                    r = call_once(client, model, question)
                r.run = i + 1
                results.append(r)
                status = r.error or f"{r.latency_sec}s in={r.prompt_tokens} out={r.output_tokens}"
                print(f"{model} | Q{QUESTIONS.index(question)+1} | run{i+1}: {status}", flush=True)
                time.sleep(SLEEP_SEC)

    # 集計（エラーを除く）
    print("\n=== 集計（成功分のみ、平均値） ===")
    print(f"{'model':<28} {'Q':<3} {'runs':>4} {'avg_s':>7} {'avg_in':>7} {'avg_out':>8}")
    for model in MODELS:
        for qi, question in enumerate(QUESTIONS):
            ok = [r for r in results if r.model == model and r.question == question and r.error is None]
            if not ok:
                print(f"{model:<28} Q{qi+1:<2} {'0':>4}  (全滅)")
                continue
            avg_s = sum(r.latency_sec for r in ok) / len(ok)
            avg_in = sum(r.prompt_tokens for r in ok) / len(ok)
            avg_out = sum(r.output_tokens for r in ok) / len(ok)
            print(f"{model:<28} Q{qi+1:<2} {len(ok):>4} {avg_s:>7.2f} {avg_in:>7.1f} {avg_out:>8.1f}")

    out = "bench_day1_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, ensure_ascii=False, indent=2)
    print(f"\n生データ: {out}")


if __name__ == "__main__":
    main()

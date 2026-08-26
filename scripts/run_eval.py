import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.eval import K, load_eval_set, run_eval

RESULTS_PATH = Path(__file__).resolve().parent.parent / "data" / "eval_results.json"

n = len(load_eval_set())
results = json.loads(RESULTS_PATH.read_text()) if RESULTS_PATH.exists() else {}

for strategy in ["fixed", "structure"]:
    if strategy in results:
        print(f"skipping strategy={strategy} -- already in {RESULTS_PATH.name} from a previous run")
        continue
    print(f"evaluating strategy={strategy} ({n} questions)...")
    results[strategy] = run_eval(strategy)
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"  saved to {RESULTS_PATH.name}")

methods = ["bm25", "dense", "hybrid", "hybrid_weighted", "rerank"]
print(f"\n=== recall@{K} matrix (n={n} questions) ===")
header = f"{'strategy':<12}" + "".join(f"{m:>17}" for m in methods)
print(header)
for strategy, scores in results.items():
    row = f"{strategy:<12}" + "".join(f"{scores[m]:>16.2%} " for m in methods)
    print(row)

"""Evaluation-only entry point; the original AIME entry is untouched."""
from .cli import benchmark, build_parser
from .learning import load_fixed_jsonl
from .config import canonical_dataset_name
parser = build_parser()
args = parser.parse_args()
if args.command != "benchmark":
    parser.error("aime_qwen_eval supports benchmark only")
if args.verifier not in {"auto", "numeric"}:
    parser.error("AIME uses the original numeric verifier")
for path in (args.dataset, *args.additional_dataset):
    if any(canonical_dataset_name((example.metadata or {}).get("dataset")) != "aime"
           or (example.metadata or {}).get("split") != "test" for example in load_fixed_jsonl(path)):
        parser.error("aime_qwen_eval requires AIME test data")
raise SystemExit(benchmark(args))

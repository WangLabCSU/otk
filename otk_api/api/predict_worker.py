"""Standalone prediction worker.

Runs a single prediction job in its own Python process and writes the result as
JSON. The API server is long-lived, and on macOS torch and xgboost each bundle
their own OpenMP (libomp) runtime: loading both into one process and running
heavy compute in each hangs/segfaults. Isolating every job in a subprocess means
each job only ever loads the single backend its model needs.

Usage:
    python -m otk_api.api.predict_worker \
        --job-id <id> --input <csv> --output-dir <dir> \
        --model <model.pkl> --device-type <cpu|cuda> [--gpu-id <n>] \
        --result-file <result.json>
"""

import argparse
import json
import sys
import traceback
from pathlib import Path

# Ensure ``otk`` (src layout) and the local ``otk_api`` package are importable
# when this runs as ``python -m otk_api.api.predict_worker`` from the repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = REPO_ROOT / "src"
for candidate in (str(REPO_ROOT), str(SRC_DIR)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from otk_api.api.predictor_wrapper import PredictorWrapper, fill_missing_columns

def main() -> int:
    parser = argparse.ArgumentParser(description="Run a single OTK prediction job")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--input", required=True, dest="input_path")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device-type", default="cpu")
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--result-file", required=True)
    args = parser.parse_args()

    result = {"success": False, "job_id": args.job_id}

    try:
        from otk_api.api.config import UPLOAD_DIR

        processed_path = Path(UPLOAD_DIR) / f"{args.job_id}_processed.csv"
        if not fill_missing_columns(args.input_path, str(processed_path)):
            processed_path = Path(args.input_path)

        wrapper = PredictorWrapper(args.model, args.device_type, args.gpu_id)
        result = wrapper.predict(str(processed_path), args.output_dir)
        wrapper.cleanup()

        result["job_id"] = args.job_id
        result["model_used"] = Path(args.model).parent.name
        result["device_used"] = (
            f"{args.device_type}:{args.gpu_id}" if args.gpu_id is not None else args.device_type
        )
        if "results_path" in result:
            result["results_path"] = str(result["results_path"])
    except Exception as exc:  # pragma: no cover - defensive
        result = {
            "success": False,
            "job_id": args.job_id,
            "error": str(exc),
            "error_trace": traceback.format_exc(),
        }

    Path(args.result_file).write_text(json.dumps(result, default=str))
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    sys.exit(main())

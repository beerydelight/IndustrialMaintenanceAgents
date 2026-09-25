#!/usr/bin/env python3
"""Human-facing entry point for the LangGraph maintenance supervisor."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
import os
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# 2. Aggressively silence the specific noisy loggers
# (Must be done before any HF/sentence-transformers modules are imported)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

from agents.supervisor_agent import run_pipeline


def configure_logging() -> None:
    """Send operational state logs to stderr, keeping reports clean on stdout."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
        force=True,
    )


def main() -> int:
    configure_logging()
    logger = logging.getLogger(__name__)
    try:
        human_input = (
            " ".join(sys.argv[1:]).strip()
            if len(sys.argv) > 1
            else (print("Describe the equipment issue: ", end="", file=sys.stderr) or input().strip())
        )
    except EOFError:
        logger.error("error=missing_input message=no input received")
        print("Error: no equipment description was provided.", file=sys.stderr)
        return 2
    if not human_input:
        logger.error("error=empty_input message=equipment description is required")
        print("Error: equipment description is required.", file=sys.stderr)
        return 2

    started = time.perf_counter()
    try:
        report = run_pipeline(human_input)
    except (ConnectionError, TimeoutError, OSError, ValueError, RuntimeError) as exc:
        elapsed = time.perf_counter() - started
        logger.error(
            "pipeline status=error seconds_to_think=%.3f error=%s",
            elapsed,
            exc,
        )
        print(f"Error: maintenance workflow failed: {exc}", file=sys.stderr)
        return 1

    logger.info(
        "pipeline status=complete seconds_to_think=%.3f",
        time.perf_counter() - started,
    )
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

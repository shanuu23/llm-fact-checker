"""Backward-compatible entry point: python -m llm_fact_checker.baseline <command>.

The command-line interface now lives in llm_fact_checker.cli; this module keeps the
original documented command working.
"""

from llm_fact_checker.cli import main

if __name__ == "__main__":
    main()

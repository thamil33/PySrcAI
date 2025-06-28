"""Entry point for running the basic debate scenario."""

from pathlib import Path
import logging
import sys

# Ensure project root is on the import path when executed directly
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pysrcai.pysrcai.debate.engine import generic_debate
from pysrcai.pysrcai.debate.scenarios import two_debate


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    log = generic_debate.run_simulation(two_debate)
    output = Path("debate_results.html")
    output.write_text(log, encoding="utf-8")
    print(f"Results written to {output}")


if __name__ == "__main__":
    main()

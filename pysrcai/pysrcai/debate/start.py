"""CLI entry point for running a simple debate."""

from importlib import import_module
import argparse

from pysrcai.pysrcai.debate.engine.generic_debate import DebateEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a debate simulation.")
    parser.add_argument(
        "--scenario",
        default="pysrcai.pysrcai.debate.scenarios.two_debate",
        help="Python module containing SCENES and INSTANCES.",
    )
    args = parser.parse_args()

    scenario = import_module(args.scenario)
    engine = DebateEngine(scenario)
    engine.run()


if __name__ == "__main__":
    main()

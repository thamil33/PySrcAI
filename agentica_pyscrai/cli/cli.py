from __future__ import annotations

import argparse
from pathlib import Path

from agentica_pyscrai.config.config import load_config, load_template, list_templates


def main() -> None:
    parser = argparse.ArgumentParser(description="Agentica PyScRAI CLI")
    config_group = parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument("--config", type=str, help="Custom config YAML file")
    config_group.add_argument("--template", type=str, choices=list_templates(), help="Use a template configuration")
    args = parser.parse_args()

    if args.config:
        cfg = load_config(args.config)
    else:
        cfg = load_template(args.template)

    print("Loaded config for collection", cfg.vectordb.collection_name)


if __name__ == "__main__":
    main()

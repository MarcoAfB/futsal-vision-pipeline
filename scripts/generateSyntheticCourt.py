import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from futsal.datasets.syntheticCourt import (
    defaultConfig,
    generateSyntheticCourt,
    verifySyntheticCourt,
)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate and save training-only synthetic court data."
    )
    parser.add_argument("--config", type=Path, default=defaultConfig)
    parser.add_argument("--outputDir", type=Path, default=Path("data/generated/syntheticCourtV1"))
    parser.add_argument("--verify", action="store_true")
    for name in ("count", "seed", "width", "height", "textureScale", "debugCount"):
        parser.add_argument(f"--{name}", type=int)
    args = parser.parse_args(argv)
    try:
        if args.verify:
            contract = verifySyntheticCourt(args.outputDir)
        else:
            contract = generateSyntheticCourt(
                args.outputDir,
                args.config,
                **{
                    name: getattr(args, name)
                    for name in ("count", "seed", "width", "height", "textureScale", "debugCount")
                },
            )
        print(f"Verified synthetic court contract: {contract['contractId']}")
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        print(f"Synthetic court operation failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

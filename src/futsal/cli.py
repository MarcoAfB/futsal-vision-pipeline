"""Command-line entry point."""

import argparse

from futsal import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Futsal dataset research tools.")
    parser.add_argument("--version", action="version", version=f"futsal {__version__}")
    parser.parse_args(argv)
    parser.print_help()
    return 0

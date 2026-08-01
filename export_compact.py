#!/usr/bin/env python3
import argparse
from pathlib import Path

from pcft.io.state import export_compact_adapter, verify_compact_adapter


def main():
    parser = argparse.ArgumentParser(description="Export and verify an ID-DR compact adapter")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    source = Path(args.source).expanduser().resolve()
    output = Path(args.output).expanduser().resolve() if args.output else source / "compact"
    export_compact_adapter(str(source), str(output))
    error = verify_compact_adapter(str(source), str(output))
    print(f"Compact adapter: {output}")
    print(f"Maximum branch output error: {error:.3e}")


if __name__ == "__main__":
    main()

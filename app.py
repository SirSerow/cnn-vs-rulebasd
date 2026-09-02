#!/usr/bin/env python3
"""Small entry point that also works without installing a console script."""

from conveyor_counter.cli import main

if __name__ == "__main__":
    raise SystemExit(main())

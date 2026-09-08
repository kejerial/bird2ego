#!/usr/bin/env python
"""CLI script for processing videos with the vision pipeline.

Thin wrapper kept for the documented scripts/ path. The implementation lives
in bird2ego.cli, which the `bird2ego` console command also uses.
"""

from __future__ import annotations

import sys

from bird2ego.cli import main

if __name__ == "__main__":
    sys.exit(main())

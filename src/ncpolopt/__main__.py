"""Command-line entry point for ncpolopt.

Prints the package version, the platform it runs on, and the SDP solvers
that are currently usable. The actual problem building happens through the
Python API; the CLI exists so ``pip install ncpolopt`` provides a useful
``ncpolopt`` command out of the box, and ``python -m ncpolopt`` works too.
"""

from __future__ import annotations

import argparse
import platform
import sys

from . import __version__
from ._logging import setup_logging
from .solvers.registry import available_solvers


def main() -> None:
    """Run the CLI: report version, platform and available solvers."""
    parser = argparse.ArgumentParser(prog="ncpolopt", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    args = parser.parse_args()  # noqa: F841 - kept for future subcommands
    setup_logging()
    print(f"ncpolopt {__version__}")
    print(f"Python {platform.python_version()} on {platform.platform()}")
    solvers = available_solvers()
    if solvers:
        print("Available solvers: " + ", ".join(s.name for s in solvers))
    else:
        print("No SDP solver found. Install one, e.g. `pip install ncpolopt[cvxpy]`.")
    sys.exit(0)


if __name__ == "__main__":
    main()

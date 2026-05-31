"""Command-line entry point: ``python -m fastmatch``.

Parses CLI arguments, configures the ``QApplication`` (OpenGL desktop attribute
must be set *before* the ``QApplication`` is created, per ``DESIGN.md`` §E.8),
loads or synthesizes an image, builds the main window, and runs the event loop.

Two non-GUI fast paths exist:

* ``--generate-sample PATH`` writes a labelled synthetic image and exits without
  ever creating a ``QApplication`` (useful in CI / headless setups).

If no positional image is given and we are not generating, we auto-synthesize a
small demo sample so the app always has something to show.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for the CLI."""
    p = argparse.ArgumentParser(
        prog="fastmatch",
        description="GPU-accelerated visual pattern matcher.",
    )
    p.add_argument(
        "image",
        nargs="?",
        default=None,
        help="Image file to open. If omitted, a small demo sample is generated.",
    )
    p.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Engine device preference (default: auto; canary-gated, CPU fallback).",
    )
    p.add_argument(
        "--generate-sample",
        metavar="PATH",
        default=None,
        help="Write a labelled synthetic sample to PATH and exit (no GUI).",
    )
    p.add_argument(
        "--w",
        type=int,
        default=12000,
        help="Sample width in px (default: 12000).",
    )
    p.add_argument(
        "--h",
        type=int,
        default=12000,
        help="Sample height in px (default: 12000).",
    )
    p.add_argument(
        "--n-targets",
        type=int,
        default=40,
        help="Number of motif stamps in a generated sample (default: 40).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for sample generation (default: 0).",
    )
    p.add_argument(
        "--max-ram-bytes",
        type=int,
        default=6_000_000_000,
        help="In-RAM RGB buffer cap before load falls back to a memmap "
        "(default: 6e9).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv[1:]``).
    """
    args = _build_parser().parse_args(argv)

    # Import the loader lazily so --help / arg errors don't pay import costs.
    from .loader import generate_sample, load_image

    # --- Headless sample-generation fast path (no GUI) ---------------------
    if args.generate_sample is not None:
        truth = generate_sample(
            args.generate_sample,
            w=args.w,
            h=args.h,
            n_targets=args.n_targets,
            seed=args.seed,
        )
        print(
            f"Wrote sample {args.generate_sample} "
            f"({args.w}x{args.h}, {len(truth)} motif stamps, seed={args.seed})."
        )
        return 0

    # --- GUI path ----------------------------------------------------------
    # Qt attributes that affect QApplication construction MUST be set before the
    # QApplication is instantiated (DESIGN.md §E.8).
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseDesktopOpenGL, True)
    app = QApplication(sys.argv if argv is None else [sys.argv[0], *argv])

    # Resolve the image: either the user's file, or an auto-generated demo.
    if args.image is not None:
        doc = load_image(args.image, max_ram_bytes=args.max_ram_bytes)
    else:
        # Auto-generate a modest demo sample (full 12000^2 would be slow to make
        # and to search on CPU, so the no-arg demo is deliberately small).
        tmp_dir = Path(tempfile.mkdtemp(prefix="fastmatch_demo_"))
        demo_path = tmp_dir / "demo_sample.png"
        generate_sample(demo_path, w=2400, h=1800, tile=160, n_targets=12, seed=args.seed)
        doc = load_image(demo_path, max_ram_bytes=args.max_ram_bytes)

    # Build + show the window. Imported here so the headless path above never
    # imports the GUI stack.
    from .app import build_main_window

    win = build_main_window(doc, device=args.device)
    win.resize(1280, 860)
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

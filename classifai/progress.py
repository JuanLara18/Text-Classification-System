"""
Rich progress bars and status display for classifai.

All progress output goes through this module so it can be
easily suppressed (e.g. in tests or CI) by passing quiet=True.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterable, Iterator, Optional

__all__ = ["PipelineProgress"]


def _rich_available() -> bool:
    try:
        import rich  # noqa: F401
        return True
    except ImportError:
        return False


class PipelineProgress:
    """
    Thin wrapper around rich.progress for classifai pipeline stages.

    Usage
    -----
        prog = PipelineProgress()

        with prog.task("Loading data", total=None) as advance:
            df = load(...)
            advance()          # mark done

        with prog.task("Embedding", total=n_texts) as advance:
            for chunk in chunks:
                embed(chunk)
                advance(len(chunk))

        # Cost tracker (LLM only)
        prog.update_cost(0.0012)   # called after each API call
        prog.print_cost_summary()
    """

    def __init__(self, quiet: bool = False) -> None:
        self.quiet = quiet
        self._total_cost: float = 0.0
        self._api_calls: int = 0
        self._progress = None   # rich Progress instance, created lazily

    # ── internal ──────────────────────────────────────────────────────────────

    def _get_progress(self):
        if self._progress is not None:
            return self._progress
        if not _rich_available():
            return None
        from rich.progress import (
            Progress,
            SpinnerColumn,
            BarColumn,
            TextColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
            MofNCompleteColumn,
        )
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            refresh_per_second=4,
        )
        return self._progress

    # ── public API ────────────────────────────────────────────────────────────

    @contextmanager
    def task(
        self,
        description: str,
        total: Optional[int] = None,
    ) -> Iterator:
        """
        Context manager for a single pipeline task.

        Yields an ``advance(n=1)`` callable. Call it once per processed
        unit (row, batch, …). When total=None the bar is indeterminate.
        """
        if self.quiet or not _rich_available():
            # Fallback: plain print, no-op advance
            print(f"  {description}...")
            t0 = time.time()
            yield lambda n=1: None
            print(f"  {description} done ({time.time() - t0:.1f}s)")
            return

        prog = self._get_progress()
        with prog:
            task_id = prog.add_task(description, total=total)

            def advance(n: int = 1) -> None:
                prog.advance(task_id, n)

            yield advance
            prog.update(task_id, completed=total or 1)

    @contextmanager
    def multi(self) -> Iterator["_MultiTaskContext"]:
        """
        Context manager for multiple concurrent progress bars
        (embedding + clustering running in stages).

        Yields a _MultiTaskContext on which you call .add_task().
        """
        if self.quiet or not _rich_available():
            yield _NoopMultiContext()
            return

        prog = self._get_progress()
        with prog:
            yield _MultiTaskContext(prog)

    def track(
        self,
        iterable: Iterable,
        description: str,
        total: Optional[int] = None,
    ):
        """
        Wrap an iterable with a progress bar (like tqdm or rich.track).

        Example::

            for batch in prog.track(batches, "Classifying", total=n_batches):
                classify(batch)
        """
        if self.quiet or not _rich_available():
            for item in iterable:
                yield item
            return

        from rich.progress import track as rich_track
        yield from rich_track(iterable, description=description, total=total)

    # ── cost tracking ─────────────────────────────────────────────────────────

    def update_cost(self, cost_usd: float) -> None:
        """Accumulate cost from a single LLM API call."""
        self._total_cost += cost_usd
        self._api_calls += 1

    def print_cost_summary(self) -> None:
        """Print a brief cost summary after the LLM classification step."""
        if self.quiet:
            return
        if _rich_available():
            from rich.console import Console
            from rich import print as rprint
            console = Console()
            console.print(
                f"  [dim]LLM calls:[/dim] {self._api_calls:,}  "
                f"[dim]estimated cost:[/dim] [green]${self._total_cost:.4f}[/green]"
            )
        else:
            print(f"  LLM calls: {self._api_calls:,}  cost: ${self._total_cost:.4f}")

    # ── simple status helpers ─────────────────────────────────────────────────

    def status(self, message: str) -> None:
        """Print a one-line status message."""
        if self.quiet:
            return
        if _rich_available():
            from rich.console import Console
            Console().print(f"  [cyan]{message}[/cyan]")
        else:
            print(f"  {message}")

    def success(self, message: str) -> None:
        if self.quiet:
            return
        if _rich_available():
            from rich.console import Console
            Console().print(f"  [green]✓[/green] {message}")
        else:
            print(f"  ✓ {message}")

    def warning(self, message: str) -> None:
        if self.quiet:
            return
        if _rich_available():
            from rich.console import Console
            Console().print(f"  [yellow]⚠[/yellow] {message}")
        else:
            print(f"  ⚠ {message}")


# ── helpers ───────────────────────────────────────────────────────────────────

class _MultiTaskContext:
    """Used inside PipelineProgress.multi() when rich is available."""

    def __init__(self, progress) -> None:
        self._progress = progress

    def add_task(self, description: str, total: Optional[int] = None):
        task_id = self._progress.add_task(description, total=total)

        def advance(n: int = 1) -> None:
            self._progress.advance(task_id, n)

        return advance


class _NoopMultiContext:
    """Fallback when rich is not available or quiet=True."""

    def add_task(self, description: str, total=None):
        return lambda n=1: None

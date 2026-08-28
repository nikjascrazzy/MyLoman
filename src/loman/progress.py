"""Console and logging progress reporters for long-running computations.

Both reporters attach as computation subscribers and render the same
whole-computation tally the widget shows --- how many nodes are computed out of
the total, and how many failed --- so a headless or terminal run reports the
progress a notebook would show visually.

Pair either reporter with :attr:`~loman.Computation.event_flush_interval` to have
the line advance *during* a single long operation such as
:meth:`~loman.Computation.compute_all`. Without an interval a subscriber sees one
event when the operation completes, which is the right cadence for a final
summary but does not tick up as work lands.
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

from tqdm import tqdm

from loman.consts import States

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from types import TracebackType
    from typing import TextIO

    from loman.computeengine import Computation, ComputationEvent

#: Fallback logger used when :func:`log_progress` is called without one.
LOG = logging.getLogger("loman.progress")


def _tally(counts: Mapping[str, int]) -> tuple[int, int, int]:
    """Reduce a state-count snapshot to ``(done, total, failed)``.

    ``done`` is the total minus the nodes still to compute --- those in STALE and
    COMPUTABLE --- which matches the widget's readout so console, log and widget
    all agree on the same numbers.
    """
    total = sum(counts.values())
    pending = counts[States.STALE.name] + counts[States.COMPUTABLE.name]
    failed = counts[States.ERROR.name]
    return total - pending, total, failed


def format_progress(event: ComputationEvent) -> str:
    """Render one progress line from an event's whole-computation tally.

    :param event: The event whose :attr:`~loman.ComputationEvent.state_counts`
        snapshot is summarised.
    :return: A line such as ``"7/10 computed"`` or ``"7/10 computed, 2 failed"``.
    """
    done, total, failed = _tally(event.state_counts)
    line = f"{done}/{total} computed"
    if failed:
        line += f", {failed} failed"
    return line


def log_progress(
    computation: Computation,
    *,
    logger: logging.Logger | None = None,
    level: int = logging.INFO,
    prefix: str = "",
) -> Callable[[], None]:
    """Log a progress line after each batched computation event.

    Suited to non-interactive runs, where a rewritten console line would be lost.
    Emits one record per event, so the cadence follows the computation's batching:
    set :attr:`~loman.Computation.event_flush_interval` for periodic ticks during a
    long operation, or leave it unset for a single summary line at the end.

    :param computation: The computation to report on.
    :param logger: Logger to emit through; defaults to ``loman.progress``.
    :param level: Logging level for each line.
    :param prefix: String prepended to every line, to tell concurrent
        computations apart in a shared log.
    :return: The unsubscribe callable from
        :meth:`~loman.Computation.subscribe`; call it to stop reporting.
    """
    sink = LOG if logger is None else logger

    def _report(event: ComputationEvent) -> None:
        """Emit one progress line for a computation event."""
        sink.log(level, "%s%s", prefix, format_progress(event))

    return computation.subscribe(_report)


class ConsoleProgress:
    """A single-line console progress bar fed by computation events.

    Use it as a context manager around the work so the line is closed with a
    newline on exit::

        with ConsoleProgress(comp):
            comp.compute_all()

    It rewrites one line in place with a carriage return, which suits an
    interactive terminal; pass ``stream`` to send it elsewhere. Pair it with
    :attr:`~loman.Computation.event_flush_interval` to make the line advance
    during a single long operation rather than only settling at the end.
    """

    def __init__(self, computation: Computation, *, stream: TextIO | None = None) -> None:
        """Subscribe to ``computation`` and render to ``stream`` (default stderr)."""
        self._stream: TextIO = sys.stderr if stream is None else stream
        self._width = 0
        self._active = False
        df = computation.to_df()
        self._last_work_left = len(df.loc[df["state"] != States.UPTODATE])
        self._bar = tqdm(file=self._stream, unit="node", total=self._last_work_left)

        self._unsubscribe: Callable[[], None] | None = computation.subscribe(self._report)

    def _report(self, event: ComputationEvent) -> None:
        """Rewrite the progress line for a computation event."""
        # done, total, failed = _tally(event.state_counts)

        # self._bar.total= total
        # self._bar.n = done

        # if failed:
        #     self._bar.set_postfix_str(f"{failed} failed")
        # else:
        #     self._bar.set_postfix_str(f"{failed} failed")

        # self._bar.refresh()
        sc_dict = {**event.state_counts}
        sc_dict.pop("UPTODATE")
        sc_dict.pop("ERROR")
        sc_dict.pop("PINNED")
        n_work_left = sum(sc_dict.values())

        self._bar.update(self._last_work_left - n_work_left)

        # line = format_progress(event)

        # # Pad to the widest line seen so a shorter update does not leave stale
        # # characters from the previous, longer one behind.
        # pad = max(self._width - len(line), 0)
        # self._stream.write("\r" + line + " " * pad)
        # self._stream.flush()
        # self._width = len(line)
        self._active = True

    def close(self) -> None:
        """Stop reporting and finish the line with a newline if one was drawn."""
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None
        if self._active:
            self._stream.write("\n")
            self._stream.flush()
            self._active = False

    def __enter__(self) -> ConsoleProgress:
        """Enter the context, returning self."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close the bar on leaving the context, even if the body raised."""
        self.close()

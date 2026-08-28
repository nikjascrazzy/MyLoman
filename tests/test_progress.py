"""Tests for the console and logging progress reporters."""

import io
import logging
import time

from loman import Computation, ConsoleProgress, States, format_progress, log_progress


def _make_computation():
    """A three-node computation: one input and two calc nodes, one of which fails."""
    comp = Computation(event_flush_interval=0.1)
    comp.add_node("x", lambda: (time.sleep(2), 1)[1])
    comp.add_node("ok", lambda x: (time.sleep(2), x + 1)[1], kwds={"x": "x"})
    comp.add_node("boom", lambda x: (time.sleep(2), 1 / 0)[1], kwds={"x": "x"})
    return comp


def test_visual_console_progress():
    """The console reporter updates while a failing computation runs."""
    comp = _make_computation()
    #    stream = io.StringIO()
    with ConsoleProgress(comp):
        comp.compute_all()


def test_format_progress_counts_done_out_of_total():
    """The line reads back the done/total tally, with no failure clause when clean."""
    comp = Computation()
    comp.add_node("x", value=1)
    comp.add_node("y", lambda x: x + 1, kwds={"x": "x"})
    events = []
    comp.subscribe(events.append)
    comp.compute_all()

    # x is up to date and y just computed: both done, none pending, none failed.
    assert format_progress(events[-1]) == "2/2 computed"


def test_format_progress_reports_failures():
    """A failed node is called out after the count."""
    comp = _make_computation()
    events = []
    comp.subscribe(events.append)
    comp.compute_all()

    assert format_progress(events[-1]) == "3/3 computed, 1 failed"


def test_log_progress_emits_a_record_per_event(caplog):
    """Each batched event logs one progress line, and unsubscribe stops it."""
    comp = _make_computation()
    unsubscribe = log_progress(comp)
    with caplog.at_level(logging.INFO, logger="loman.progress"):
        comp.compute_all()
    assert any("computed" in message for message in caplog.messages)
    assert "3/3 computed, 1 failed" in caplog.messages[-1]

    caplog.clear()
    unsubscribe()
    comp.add_node("z", value=5)
    comp.compute_all()
    assert [record for record in caplog.records if record.name == "loman.progress"] == []


def test_log_progress_honours_logger_level_and_prefix(caplog):
    """A caller's logger, level and prefix all flow through to the record."""
    comp = _make_computation()
    logger = logging.getLogger("test.progress.custom")
    log_progress(comp, logger=logger, level=logging.WARNING, prefix="[run-7] ")
    with caplog.at_level(logging.WARNING, logger="test.progress.custom"):
        comp.compute_all()
    assert caplog.records[-1].levelno == logging.WARNING
    assert caplog.messages[-1].startswith("[run-7] ")


def test_console_progress_rewrites_a_line_and_closes_with_newline():
    """The bar rewrites in place with a carriage return, ending in one newline."""
    comp = _make_computation()
    stream = io.StringIO()
    with ConsoleProgress(comp, stream=stream):
        comp.compute_all()
    output = stream.getvalue()

    assert output.startswith("\r")
    assert output.endswith("\n")
    assert "3/3 computed, 1 failed" in output


def test_console_progress_pads_over_a_shorter_update():
    """A shorter line is padded so leftovers from a longer one are cleared."""
    comp = Computation()
    comp.add_node("x", value=1)
    stream = io.StringIO()
    bar = ConsoleProgress(comp, stream=stream)
    # First a long line, then force a shorter one and check it is padded out.
    bar._report(_FakeEvent({States.STALE.name: 0, States.COMPUTABLE.name: 0, States.ERROR.name: 12, "UPTODATE": 100}))
    long_len = len(stream.getvalue())
    bar._report(_FakeEvent({States.STALE.name: 0, States.COMPUTABLE.name: 0, States.ERROR.name: 0, "UPTODATE": 1}))
    bar.close()
    second_write = stream.getvalue()[long_len:]
    # The short line carries trailing spaces to overwrite the wider one.
    assert "  " in second_write


def test_console_progress_close_is_idempotent_and_quiet_when_unused():
    """Closing twice is safe, and a bar that never drew prints no newline."""
    comp = Computation()
    comp.add_node("x", value=1)
    stream = io.StringIO()
    bar = ConsoleProgress(comp, stream=stream)
    bar.close()
    bar.close()
    # No event ever fired, so nothing --- not even a newline --- was written.
    assert stream.getvalue() == ""


def test_flush_interval_ticks_the_bar_up_during_a_long_compute():
    """With a flush interval set, the bar advances mid-compute, not just at the end."""
    comp = Computation(event_flush_interval=1e-9)
    comp.add_node("n0", value=0)
    for i in range(1, 5):
        comp.add_node(f"n{i}", lambda prev: prev + 1, kwds={"prev": f"n{i - 1}"})
    stream = io.StringIO()
    with ConsoleProgress(comp, stream=stream):
        comp.compute_all()

    # A near-zero interval flushes after each node, so the bar is redrawn several
    # times (each redraw begins with a carriage return) rather than once at the end.
    assert stream.getvalue().count("\r") > 1
    assert "5/5 computed" in stream.getvalue()


def test_computation_helpers_attach_reporters(caplog):
    """The convenience methods on Computation wire the same reporters up."""
    comp = _make_computation()
    stream = io.StringIO()
    logger = logging.getLogger("test.progress.method")
    comp.log_progress(logger=logger)
    with caplog.at_level(logging.INFO, logger="test.progress.method"), comp.console_progress(stream=stream):
        comp.compute_all()
    assert "3/3 computed, 1 failed" in caplog.messages[-1]
    assert "3/3 computed, 1 failed" in stream.getvalue()


class _FakeEvent:
    """Minimal stand-in carrying just the state_counts a reporter reads."""

    def __init__(self, counts):
        """Store the counts mapping the reporter will format."""
        self.state_counts = counts

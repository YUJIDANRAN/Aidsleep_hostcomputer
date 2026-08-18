"""Background serial drain: keep the OS buffer empty without blocking the UI."""

from __future__ import annotations

import queue
import threading
import time
from typing import Callable, List, Optional, TypeVar

T = TypeVar("T")


class BackgroundQueuePump(threading.Thread):
    """Call drain() on a worker thread and hand chunks to the UI via a queue."""

    def __init__(
        self,
        drain: Callable[[], List[T]],
        *,
        name: str = "serial-pump",
        idle_sleep_sec: float = 0.001,
    ) -> None:
        super().__init__(name=name, daemon=True)
        self._drain: Optional[Callable[[], List[T]]] = drain
        self._queue: "queue.Queue[List[T]]" = queue.Queue()
        # Do NOT name this `_stop`: threading.Thread already uses `_stop` as a method.
        self._stop_event = threading.Event()
        self._paused = threading.Event()
        self._paused.set()
        self._busy = threading.Event()
        self._idle_sleep_sec = float(idle_sleep_sec)
        self._error: Optional[BaseException] = None

    def run(self) -> None:
        while not self._stop_event.is_set():
            if self._paused.is_set():
                if self._stop_event.wait(0.02):
                    break
                continue
            drain = self._drain
            if drain is None:
                break
            self._busy.set()
            try:
                items = drain()
            except Exception as exc:
                self._error = exc
                items = []
                time.sleep(0.02)
            finally:
                self._busy.clear()
            if self._stop_event.is_set() or self._paused.is_set():
                continue
            if items:
                self._queue.put(items)
            else:
                time.sleep(self._idle_sleep_sec)

    def resume(self) -> None:
        self._error = None
        self._paused.clear()

    def pause(self) -> None:
        self._paused.set()

    def wait_until_idle(self, timeout: float = 2.0) -> bool:
        """Block until the current drain call finishes (or timeout)."""
        deadline = time.monotonic() + max(0.0, float(timeout))
        while self._busy.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(0.005, remaining))
        return True

    def shutdown(self, timeout: float = 3.0) -> None:
        self._paused.set()
        self._stop_event.set()
        self._drain = None
        self.wait_until_idle(timeout=min(1.0, timeout))
        if self.is_alive():
            self.join(timeout=timeout)

    def take_all(self) -> List[T]:
        out: List[T] = []
        while True:
            try:
                chunk = self._queue.get_nowait()
            except queue.Empty:
                break
            out.extend(chunk)
        return out

    def clear(self) -> None:
        self.take_all()

    def pending_chunks(self) -> int:
        return self._queue.qsize()

    def pop_error(self) -> Optional[BaseException]:
        error = self._error
        self._error = None
        return error

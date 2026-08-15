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
        self._drain = drain
        self._queue: "queue.Queue[List[T]]" = queue.Queue()
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._paused.set()
        self._idle_sleep_sec = float(idle_sleep_sec)
        self._error: Optional[BaseException] = None

    def run(self) -> None:
        while not self._stop.is_set():
            if self._paused.is_set():
                self._stop.wait(0.02)
                continue
            try:
                items = self._drain()
            except Exception as exc:
                self._error = exc
                time.sleep(0.02)
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

    def shutdown(self, timeout: float = 1.0) -> None:
        self._stop.set()
        self._paused.clear()
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

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class RolloutPool(Protocol[InputT, OutputT]):
    def map(
        self, function: Callable[[InputT], OutputT], jobs: Iterable[InputT]
    ) -> list[OutputT]: ...

    def iter_map(
        self, function: Callable[[InputT], OutputT], jobs: Iterable[InputT]
    ) -> Iterator[OutputT]: ...


@dataclass(frozen=True)
class ThreadRolloutPool(Generic[InputT, OutputT]):
    workers: int = 1

    def iter_map_resumable(self, function, jobs):
        """Run generators yielding backoff seconds; return their terminal values.

        Each admitted generator owns one trajectory slot until it terminates.
        Sleeping continuations consume no thread, but a replacement trajectory is
        admitted only after the current trajectory releases its slot.
        """
        if self.workers <= 0:
            raise ValueError("workers must be positive")
        queued = deque(function(job) for job in jobs)
        ready = []
        futures = {}

        def advance(iterator):
            try:
                return False, max(0.0, float(next(iterator)))
            except StopIteration as done:
                return True, done.value

        executor = ThreadPoolExecutor(max_workers=self.workers)
        try:
            while queued or ready or futures:
                while queued and len(ready) + len(futures) < self.workers:
                    ready.append((time.monotonic(), queued.popleft()))
                ready.sort(key=lambda item: item[0])
                while ready and len(futures) < self.workers and ready[0][0] <= time.monotonic():
                    _, iterator = ready.pop(0)
                    futures[executor.submit(advance, iterator)] = iterator
                timeout = None
                if ready and len(futures) < self.workers:
                    timeout = max(0.0, ready[0][0] - time.monotonic())
                if not futures:
                    time.sleep(min(timeout or 0.01, 0.1))
                    continue
                completed, _ = wait(futures, timeout=timeout, return_when=FIRST_COMPLETED)
                for future in completed:
                    iterator = futures.pop(future)
                    done, value = future.result()
                    if done:
                        yield value
                    else:
                        ready.append((time.monotonic() + value, iterator))
        finally:
            for future in futures:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
            for iterator in queued:
                iterator.close()
            for _, iterator in ready:
                iterator.close()
            for iterator in futures.values():
                iterator.close()

    def map(self, function: Callable[[InputT], OutputT], jobs: Iterable[InputT]) -> list[OutputT]:
        if self.workers <= 0:
            raise ValueError("workers must be positive")
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            return list(executor.map(function, jobs))

    def iter_map(
        self, function: Callable[[InputT], OutputT], jobs: Iterable[InputT]
    ) -> Iterator[OutputT]:
        """Yield each result immediately when that job completes."""

        if self.workers <= 0:
            raise ValueError("workers must be positive")
        executor = ThreadPoolExecutor(max_workers=self.workers)
        pending_jobs = iter(jobs)
        futures: set[Future[OutputT]] = set()
        for _ in range(self.workers):
            try:
                job = next(pending_jobs)
            except StopIteration:
                break
            futures.add(executor.submit(function, job))
        try:
            while futures:
                completed, _pending = wait(futures, return_when=FIRST_COMPLETED)
                completed_count = 0
                while completed:
                    for future in completed:
                        futures.remove(future)
                        completed_count += 1
                        # Yield a burst before refilling. The short grace drain
                        # prevents a near-simultaneous circuit-opening failure
                        # from racing one unnecessary replacement submission.
                        yield future.result()
                    if not futures:
                        break
                    completed, _pending = wait(
                        futures,
                        timeout=0.01,
                        return_when=FIRST_COMPLETED,
                    )
                for _ in range(completed_count):
                    try:
                        job = next(pending_jobs)
                    except StopIteration:
                        break
                    futures.add(executor.submit(function, job))
        finally:
            # A rollout interruption must not drain every queued API call before
            # returning control. Running calls finish cleanly; jobs that have not
            # started are cancelled and remain exact resume holes.
            for future in futures:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)

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

    def iter_map_resumable(self, function, jobs, *, group_key, max_active_groups):
        """Run generators yielding backoff seconds; return their terminal values.

        Sleeping continuations retain group ownership but consume no worker.
        A generator is resumed by only one worker at a time.
        """
        if self.workers <= 0 or max_active_groups <= 0:
            raise ValueError("worker and group limits must be positive")
        queued = deque((str(group_key(job)), function(job)) for job in jobs)
        ready = []
        active = {}
        futures = {}

        def advance(iterator):
            try:
                return False, max(0.0, float(next(iterator)))
            except StopIteration as done:
                return True, done.value

        executor = ThreadPoolExecutor(max_workers=self.workers)
        try:
            while queued or ready or futures:
                for _ in range(len(queued)):
                    key, iterator = queued.popleft()
                    if key in active or len(active) < max_active_groups:
                        active[key] = active.get(key, 0) + 1
                        ready.append((time.monotonic(), key, iterator))
                    else:
                        queued.append((key, iterator))
                ready.sort(key=lambda item: item[0])
                while ready and len(futures) < self.workers and ready[0][0] <= time.monotonic():
                    _, key, iterator = ready.pop(0)
                    futures[executor.submit(advance, iterator)] = (key, iterator)
                timeout = None
                if ready and len(futures) < self.workers:
                    timeout = max(0.0, ready[0][0] - time.monotonic())
                if not futures:
                    time.sleep(min(timeout or 0.01, 0.1))
                    continue
                completed, _ = wait(futures, timeout=timeout, return_when=FIRST_COMPLETED)
                for future in completed:
                    key, iterator = futures.pop(future)
                    done, value = future.result()
                    if done:
                        active[key] -= 1
                        if not active[key]:
                            del active[key]
                        yield value
                    else:
                        ready.append((time.monotonic() + value, key, iterator))
        finally:
            for future in futures:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
            for _, iterator in queued:
                iterator.close()
            for _, _, iterator in ready:
                iterator.close()
            for _, iterator in futures.values():
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

    def iter_map_grouped(
        self,
        function: Callable[[InputT], OutputT],
        jobs: Iterable[InputT],
        *,
        group_key: Callable[[InputT], str],
        max_active_groups: int,
    ) -> Iterator[OutputT]:
        """Continuously refill workers while bounding concurrently active groups."""

        if self.workers <= 0:
            raise ValueError("workers must be positive")
        if max_active_groups <= 0:
            raise ValueError("max_active_groups must be positive")
        grouped: dict[str, deque[InputT]] = {}
        for job in jobs:
            grouped.setdefault(str(group_key(job)), deque()).append(job)
        waiting = deque(grouped)
        active: list[str] = []
        inflight = {key: 0 for key in grouped}

        def admit_groups() -> None:
            while waiting and len(active) < max_active_groups:
                active.append(waiting.popleft())

        def next_job() -> tuple[str, InputT] | None:
            for _ in range(len(active)):
                key = active.pop(0)
                active.append(key)
                if grouped[key]:
                    return key, grouped[key].popleft()
            return None

        def release_groups() -> None:
            active[:] = [key for key in active if grouped[key] or inflight[key] > 0]
            admit_groups()

        admit_groups()
        executor = ThreadPoolExecutor(max_workers=self.workers)
        futures: dict[Future[OutputT], str] = {}

        def refill() -> None:
            while len(futures) < self.workers:
                selected = next_job()
                if selected is None:
                    break
                key, job = selected
                inflight[key] += 1
                futures[executor.submit(function, job)] = key

        refill()
        try:
            while futures:
                completed, _pending = wait(futures, return_when=FIRST_COMPLETED)
                completed_results: list[OutputT] = []
                while completed:
                    for future in completed:
                        key = futures.pop(future)
                        inflight[key] -= 1
                        completed_results.append(future.result())
                    if not futures:
                        break
                    completed, _pending = wait(
                        futures,
                        timeout=0.01,
                        return_when=FIRST_COMPLETED,
                    )
                # Preserve the ordinary iterator's circuit-breaker boundary: the
                # caller sees every already-completed result before new work starts.
                yield from completed_results
                release_groups()
                refill()
        finally:
            for future in futures:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)

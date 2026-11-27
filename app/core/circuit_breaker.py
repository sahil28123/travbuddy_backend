"""
Per-provider circuit breakers with concurrency limiting + hard timeouts.

Each breaker gives one external provider three protections in one wrapper:

  * Circuit breaking — closed → open (after N failures) → half-open (after a
    recovery timeout) → closed. While open, calls short-circuit to a fallback
    instead of hammering a provider that's already down.
  * Concurrency limiting — a per-provider semaphore caps how many calls run at
    once ACROSS ALL TRIPS. 100 trips can't fire 100 simultaneous Booking calls;
    the surplus queues on the semaphore. The semaphore is loop-scoped (rebuilt
    per event loop) so Celery's per-task `asyncio.run()` loops don't reuse a
    semaphore bound to a closed loop — same reasoning as curl_pool.
  * Hard timeout — no provider call waits forever; a timeout trips the breaker
    and returns the fallback.

Module-level singletons are shared across the process lifetime so failure
counts, in-flight counts, and semaphores persist across requests (the whole
point). `stats()` feeds the health dashboard.
"""
import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Callable

from app.config import settings

logger = logging.getLogger(__name__)


class CircuitOpenError(Exception):
    """Raised by CircuitBreaker.guard() when the breaker is open — the caller
    decides how to degrade (vs .call(), which silently returns a fallback)."""
    def __init__(self, name: str):
        super().__init__(f"circuit '{name}' is open")
        self.name = name


class CircuitBreaker:
    def __init__(
        self,
        name: str,
        failure_threshold: int = 3,
        recovery_timeout: float = 60.0,
        max_concurrency: int | None = None,
        timeout: float | None = None,
    ):
        self.name = name
        self._failures = 0
        self._threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._last_failure: float = 0.0
        self._state = "closed"  # closed | open | half-open
        self._max_concurrency = max_concurrency or settings.provider_max_concurrency
        self._timeout = timeout if timeout is not None else settings.provider_timeout_seconds
        # Loop-scoped semaphore: (loop_id, semaphore). Rebuilt when the running
        # loop changes so a semaphore never outlives its loop.
        self._sem: tuple[int, asyncio.Semaphore] | None = None
        # ── observability counters (lifetime, process-local) ──
        self._calls = 0
        self._successes = 0
        self._failures_total = 0
        self._timeouts = 0
        self._short_circuited = 0   # skipped because breaker was open
        self._in_flight = 0
        self._peak_in_flight = 0

    # ── circuit state ────────────────────────────────────────────────────────
    @property
    def state(self) -> str:
        if self._state == "open":
            if time.monotonic() - self._last_failure > self._recovery_timeout:
                self._state = "half-open"
        return self._state

    @property
    def is_open(self) -> bool:
        return self.state == "open"

    def _semaphore(self) -> asyncio.Semaphore:
        loop_id = id(asyncio.get_running_loop())
        if self._sem is None or self._sem[0] != loop_id:
            self._sem = (loop_id, asyncio.Semaphore(self._max_concurrency))
        return self._sem[1]

    def _record_failure(self, exc: BaseException, *, timed_out: bool = False) -> None:
        self._failures += 1
        self._failures_total += 1
        if timed_out:
            self._timeouts += 1
        self._last_failure = time.monotonic()
        if self._failures >= self._threshold:
            prev = self._state
            self._state = "open"
            if prev != "open":
                logger.error("[circuit] %s OPENED after %d failures: %s", self.name, self._failures, exc)
        else:
            logger.warning("[circuit] %s failure %d/%d: %s", self.name, self._failures, self._threshold, exc)

    def _record_success(self) -> None:
        self._successes += 1
        if self._state == "half-open":
            logger.info("[circuit] %s RECOVERED → closed", self.name)
        self._state = "closed"
        self._failures = 0

    async def call(self, coro_fn: Callable, *args, fallback: Any = None, **kwargs) -> Any:
        """Run `coro_fn(*args, **kwargs)` under the breaker + concurrency gate +
        timeout. Returns `fallback` (never raises) when the breaker is open, the
        call times out, or the call errors — the scraper aggregators rely on this
        so one dead provider never fails the whole search."""
        self._calls += 1
        if self.state == "open":
            self._short_circuited += 1
            logger.warning("[circuit] %s OPEN — skipping call, returning fallback", self.name)
            return fallback

        # Concurrency gate: surplus callers queue here rather than stampeding the
        # provider. Acquired AFTER the open-check so an open breaker never queues.
        sem = self._semaphore()
        async with sem:
            # The breaker may have opened while this call waited for a slot.
            if self.state == "open":
                self._short_circuited += 1
                logger.warning("[circuit] %s OPEN — skipping queued call, returning fallback", self.name)
                return fallback

            self._in_flight += 1
            self._peak_in_flight = max(self._peak_in_flight, self._in_flight)
            try:
                result = await asyncio.wait_for(coro_fn(*args, **kwargs), timeout=self._timeout)
                self._record_success()
                return result
            except asyncio.TimeoutError as e:
                logger.warning("[circuit] %s TIMEOUT after %.0fs — returning fallback", self.name, self._timeout)
                self._record_failure(e, timed_out=True)
                return fallback
            except Exception as e:
                self._record_failure(e)
                return fallback
            finally:
                self._in_flight -= 1

    @asynccontextmanager
    async def guard(self):
        """Concurrency gate + circuit + stats for callers that must handle their
        own errors/timeouts and keep RAISING (e.g. Vertex, which has its own
        region-retry loop). Raises CircuitOpenError when open; records and
        re-raises the caller's exceptions. No timeout is imposed here — the
        caller owns that (a wrapping wait_for would cut a legitimate retry loop)."""
        self._calls += 1
        if self.state == "open":
            self._short_circuited += 1
            raise CircuitOpenError(self.name)
        sem = self._semaphore()
        async with sem:
            # Reject queued callers before counting them as provider attempts.
            if self.state == "open":
                self._short_circuited += 1
                raise CircuitOpenError(self.name)

            self._in_flight += 1
            self._peak_in_flight = max(self._peak_in_flight, self._in_flight)
            try:
                yield
                self._record_success()
            except Exception as e:
                self._record_failure(e)
                raise
            finally:
                self._in_flight -= 1

    def stats(self) -> dict:
        """Snapshot for the health dashboard. success_pct is over completed calls
        (excludes short-circuited skips, which aren't provider attempts)."""
        completed = self._successes + self._failures_total
        success_pct = round(100.0 * self._successes / completed, 1) if completed else 100.0
        return {
            "name": self.name,
            "state": self.state,
            "success_pct": success_pct,
            "calls": self._calls,
            "successes": self._successes,
            "failures": self._failures_total,
            "timeouts": self._timeouts,
            "short_circuited": self._short_circuited,
            "consecutive_failures": self._failures,
            "in_flight": self._in_flight,
            "peak_in_flight": self._peak_in_flight,
            "max_concurrency": self._max_concurrency,
            "timeout_sec": self._timeout,
        }


# ── Module-level singletons — one per external provider ──────────────────────
# Concurrency caps reflect each provider's fragility/limits: Booking's WAF is the
# touchiest (keep it low), the Indian OTAs tolerate more. Tune via the numbers here.
# Gemini opening blocks ALL generation (no LLM = no trip), and vertex_generate
# already rotates through 11 regions before it raises — so only a genuine,
# sustained outage (many trips failing after full region cycling) should open it.
breaker_gemini   = CircuitBreaker("Gemini",     failure_threshold=6, recovery_timeout=20.0,
                                  max_concurrency=settings.vertex_max_concurrency, timeout=180.0)
breaker_places   = CircuitBreaker("Places",     failure_threshold=3, recovery_timeout=60.0,  max_concurrency=10)
breaker_booking  = CircuitBreaker("Booking",    failure_threshold=2, recovery_timeout=120.0, max_concurrency=5)
breaker_emt      = CircuitBreaker("EaseMyTrip", failure_threshold=3, recovery_timeout=60.0,  max_concurrency=8)
breaker_redbus   = CircuitBreaker("RedBus",     failure_threshold=3, recovery_timeout=60.0,  max_concurrency=8)
breaker_flixbus  = CircuitBreaker("FlixBus",    failure_threshold=3, recovery_timeout=60.0,  max_concurrency=6)
breaker_agoda    = CircuitBreaker("Agoda",      failure_threshold=3, recovery_timeout=60.0,  max_concurrency=8)
breaker_ixigo    = CircuitBreaker("Ixigo",      failure_threshold=3, recovery_timeout=60.0,  max_concurrency=8)
breaker_gotogate = CircuitBreaker("Gotogate",   failure_threshold=3, recovery_timeout=60.0,  max_concurrency=6)

# All breakers, for the health dashboard.
ALL_BREAKERS = (
    breaker_gemini, breaker_places, breaker_booking, breaker_emt, breaker_redbus,
    breaker_flixbus, breaker_agoda, breaker_ixigo, breaker_gotogate,
)

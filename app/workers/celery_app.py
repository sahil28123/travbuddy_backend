"""
Celery worker configuration.
Jobs are stored in Redis so API endpoints can poll status.
"""
import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

import sentry_sdk
from celery import Celery
from celery.signals import celeryd_init, worker_process_init

from app.config import settings

logger = logging.getLogger(__name__)


# ── which schedule this beat process owns ────────────────────────────────────
#
# **The sweep was scheduled and never ran, for four months.**
#
# `stayos-waiting-guests` sits in the schedule below with a comment saying beat
# enqueues it and "only the StayOS worker — the one with StayOS's SUPABASE_URL —
# ever runs it". That reasoning assumed one Redis. It is not one Redis:
# `docker-compose.stayos.yml` brings up its own (`stayos_redis_data`), and its
# api and worker both point at `redis://redis:6379/0` inside their own network.
#
# So the marketplace beat enqueues the sweep onto the *marketplace* broker's
# `hotelos` queue, and the StayOS worker is listening to the *StayOS* broker's
# `hotelos` queue. The two never meet. No error, no dead letter, no symptom —
# the guest simply waits, and the notification that was built to reach a
# manager's phone reaches nothing. Confirmed against production: zero rows in
# `agent_escalations`, ever, and the StayOS compose has no beat service at all.
#
# The comment that justified having no beat service — "every scheduled entry is
# a marketplace task" — was true when it was written and stopped being true when
# the sweep was added, which is exactly the kind of drift a comment cannot
# catch and a split cannot hide.
#
# So the schedule is split by role rather than shared. A StayOS beat runs the
# StayOS entries and *only* those: `revert_intro_commission` writes hourly to
# `partners`, a table that also exists and means something entirely different
# in the hotel OS, and pointing it at a hotel database is the one outcome worse
# than the sweep not running.
#
# Default is `all`, so the marketplace deployment is unchanged by this and
# needs no redeploy.

_MARKETPLACE_SCHEDULE = {
    "reco-expected-price-nightly": {
        "task": "travbuddy.reco.refresh_expected_price", "schedule": 24 * 3600,
    },
    "reco-destination-priors-nightly": {
        "task": "travbuddy.reco.refresh_destination_priors", "schedule": 24 * 3600,
    },
    "reco-fit-centroids-weekly": {
        "task": "travbuddy.reco.fit_centroids", "schedule": 7 * 24 * 3600,
    },
    # Hourly, not nightly: an intro rate ends at a wall-clock moment, and a
    # partner who watches their dashboard flip a rate the morning after it
    # expired asks why. Cheap — it touches only rows that have lapsed.
    #
    # **Never on a StayOS beat.** It writes to `partners`.
    "partner-intro-commission-hourly": {
        "task": "travbuddy.partners.revert_intro_commission", "schedule": 3600,
    },
}

_STAYOS_SCHEDULE = {
    # Every five minutes rather than every minute: `WAITING_AFTER_MINUTES` is
    # ten, so five bounds the worst case at fifteen minutes from a guest's
    # message to a manager's phone, and a sweep that found nothing is one
    # indexed query.
    "stayos-waiting-guests": {
        "task": "travbuddy.messaging.sweep_waiting_guests", "schedule": 300,
    },
    # Inventory drift is the one class of defect here that does not announce
    # itself. An oversold night sells a room the hotel does not have and nobody
    # learns until two guests arrive for it; a stranded night takes a sellable
    # room off the market and looks like a slow week.
    #
    # Hourly, and it reports rather than corrects — see the task. The
    # reconciliation has existed all along and nothing ran it, which is how both
    # of the real incidents were found by hand.
    "stayos-inventory-drift": {
        "task": "travbuddy.pms.reconcile_inventory", "schedule": 3600,
    },
    # The calendar out to Airbnb, Booking.com, Vrbo and tiket.com, in full, once
    # a night. Every edit already pushes its own nights; this is what moves the
    # far edge of the sold window forward a day and repairs any push lost to a
    # broker outage. Enqueues one push per mapped room type and does nothing at
    # all unless Repull is enabled — see services/repull/ari_push.py.
    "stayos-repull-resync": {
        "task": "travbuddy.pms.repull_resync_all", "schedule": 24 * 3600,
    },
}


def _beat_schedule() -> dict:
    """
    The entries this beat process is responsible for.

    `CELERY_BEAT_ROLE` is read from the environment rather than from `settings`
    because it is a property of *this process's job in the deployment*, not of
    the product — the same image runs as api, worker and beat, and only the
    compose file knows which.

        marketplace   the reco and partner tasks (the historical default)
        stayos        the waiting-guest sweep, and nothing that writes to a
                      marketplace table
        all           both — the default, so nothing changes for a deployment
                      that does not set it
    """
    import os

    role = (os.environ.get("CELERY_BEAT_ROLE") or "all").strip().lower()
    if role == "stayos":
        return dict(_STAYOS_SCHEDULE)
    if role == "marketplace":
        return dict(_MARKETPLACE_SCHEDULE)
    if role != "all":
        logger.warning(
            "CELERY_BEAT_ROLE=%r is not a role; scheduling everything", role,
        )
    return {**_MARKETPLACE_SCHEDULE, **_STAYOS_SCHEDULE}


celery_app = Celery(
    "travbuddy",
    broker=settings.redis_url,
    backend=settings.redis_url,
    # Task modules the worker must import so their @task functions register. Without
    # this, tasks defined outside celery_app.py (e.g. the recommendation workers) are
    # never discovered and beat's schedule fires into the void.
    include=[
        "app.workers.reco_tasks",
        "app.workers.partner_tasks",
        "app.workers.messaging_tasks",
        "app.workers.marketing_tasks",
        "app.workers.repull_tasks",
    ],
)


# ── Sentry ────────────────────────────────────────────────────────────────────
# Initialize Sentry only when running AS a worker — not on mere import (the API
# imports this module to enqueue tasks and must keep its own "api" Sentry init).
# celeryd_init covers the main worker process; worker_process_init covers each
# forked prefork child so every task-executing process reports to Sentry.
@celeryd_init.connect
def _sentry_celeryd_init(**_kwargs):
    from app.observability import init_sentry
    from app.services import trace
    init_sentry("worker")
    trace.install()


@worker_process_init.connect
def _sentry_worker_process_init(**_kwargs):
    from app.observability import init_sentry
    from app.services import trace
    init_sentry("worker")
    trace.install()

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,           # one task at a time per worker process
    worker_concurrency=settings.celery_worker_concurrency,  # cap concurrent workers (rate-limit protection)
    result_expires=86400,                   # results kept for 24 hours
    # ── which worker may run which task ──────────────────────────────────────
    #
    # This codebase is deployed **twice**: once against the marketplace Supabase
    # project and once against the hotel OS's. One `SUPABASE_URL` per process,
    # so a task executed by the wrong deployment reads the wrong database.
    #
    # Both deployments enqueue through Celery, and without routing every task
    # lands on the one default queue that both workers consume. Whichever worker
    # grabs it first runs it: a guest's WhatsApp reply picked up by the
    # marketplace worker looks for `conversations` in a project that has never
    # had that table, and fails in a service nobody was looking at. Sharing one
    # Redis makes that certain rather than likely — and the StayOS service is
    # meant to be configured by copying the marketplace's environment, `REDIS_URL`
    # included.
    #
    # **Only the hotel-OS tasks are routed.** Everything else keeps landing on
    # the default `celery` queue, because the marketplace worker is already
    # running in production with no `-Q` and would stop consuming anything the
    # moment its tasks moved. Renaming its queue to be symmetrical would be a
    # tidier diagram and an outage.
    #
    # The StayOS worker therefore starts with `-Q hotelos`, and the marketplace
    # worker's command does not change. Beat belongs to the marketplace
    # deployment: every scheduled entry below is a marketplace task, and
    # `revert_intro_commission` runs hourly and *writes* to `partners` — a table
    # that also exists, meaning something quite different, in the hotel OS.
    task_routes={
        "travbuddy.messaging.*": {"queue": "hotelos"},
        "travbuddy.marketing.*": {"queue": "hotelos"},
        # The hotel's own operational tasks. Same deployment as messaging —
        # `pms` is the hotel side of this codebase, and a reconciliation that
        # ran on the marketplace worker would be reading a hotel database from
        # a process that has no business with one.
        "travbuddy.pms.*": {"queue": "hotelos"},
    },
    # Recommendation-engine nightly/weekly maintenance (docs/recommendation_engine.md
    # Part 9.3 / 14.3). Defined here — not in reco_tasks — so `celery beat` sees the
    # schedule from conf regardless of task-module import order. Task functions are
    # registered via `include` above and executed by the worker.
    beat_schedule=_beat_schedule(),
)


@celery_app.task(bind=True, name="travbuddy.analyze_reel", max_retries=0)
def analyze_reel_task(
    self,
    job_id: str,
    reel_url: str,
    options: dict,
    metadata: Optional[dict],
    user_id: Optional[str] = None,
):
    """
    Celery task that runs the full pipeline.
    Stores progress and result in Redis (API polling) and mirrors job state
    into the Supabase `reel_jobs` table (app's developing-reels feed).
    """
    from app.pipeline.orchestrator import run_pipeline, PipelineEvent
    from app.workers.job_store import update_job_progress, store_job_result, _get_redis

    # Tag Sentry with the Supabase user + job so worker errors are traceable to a
    # specific user/row. Set on the global scope because the pipeline offloads
    # work (e.g. frame uploads) to executor threads that don't inherit the
    # task's scope; the worker runs one task at a time so there's no cross-talk.
    _scope = sentry_sdk.get_global_scope()
    _scope.set_tag("job_id", job_id)
    _scope.set_user({"id": user_id} if user_id else None)

    # Idempotency guard: with acks_late a worker crash redelivers the task —
    # never re-run a job that already started (each run costs Gemini/Places $).
    try:
        claimed = _get_redis().set(f"job:{job_id}:claimed", "1", nx=True, ex=3600)
        if not claimed:
            logger.warning(f"Job {job_id} already claimed by another run — skipping redelivery")
            return {"job_id": job_id, "status": "duplicate"}
    except Exception as e:
        logger.warning(f"Idempotency check unavailable for {job_id} (continuing): {e}")

    update_job_progress(job_id, stage="download", percent=0)
    _update_supabase_job(job_id, status="processing", progress=0)

    events_buffer: list[dict] = []

    def on_event(event: PipelineEvent):
        events_buffer.append({"type": event.event_type, "data": event.data})
        _handle_progress_event(job_id, event)

    try:
        result = asyncio.run(
            run_pipeline(
                job_id=job_id,
                reel_url=reel_url,
                options=options,
                metadata=metadata,
                user_id=user_id,
                on_event=on_event,
            )
        )
    except Exception as e:
        logger.exception(f"Pipeline crashed for {job_id}: {e}")
        _update_supabase_job(job_id, status="failed", error_message=str(e)[:500])
        raise

    store_job_result(job_id, result, events_buffer)
    _finalize_supabase_job(job_id, result)
    return {"job_id": job_id, "status": result.status}


def _update_supabase_job(job_id: str, **fields):
    """Mirror job state into the reel_jobs table; never fail the pipeline over it."""
    from app.config import settings
    if not (settings.supabase_url and settings.supabase_service_key):
        return
    try:
        from app.storage.supabase_client import update_reel_job
        update_reel_job(job_id, **fields)
    except Exception as e:
        logger.warning(f"reel_jobs mirror update failed for {job_id}: {e}")


def _finalize_supabase_job(job_id: str, result):
    """Write final status + display fields (title/thumbnail/video/category) to reel_jobs."""
    from collections import Counter

    fields: dict = {"progress": 100, "status": result.status}

    if result.status == "completed":
        first_place = next(
            (seg.place.name for seg in result.timeline if seg.place and seg.place.name),
            None,
        )
        fields["title"] = result.summary.dominant_city or first_place or "Imported Reel"
        fields["thumbnail_url"] = next(
            (seg.frame_thumbnail_url for seg in result.timeline if seg.frame_thumbnail_url),
            None,
        )
        fields["video_url"] = result.reel.video_url
        # Dominant place category drives the app's "Plan trip" CTA
        # (shown for tourist places, hidden for cafes/restaurants/etc.)
        categories = Counter(
            seg.place.category for seg in result.timeline
            if seg.place and seg.place.category
        )
        if categories:
            fields["place_category"] = categories.most_common(1)[0][0]
    elif result.error:
        fields["error_message"] = result.error.message[:500]

    _update_supabase_job(job_id, **fields)

    # Completed reels land in the app's Recent Reels automatically
    if result.status == "completed":
        from app.config import settings
        if settings.supabase_url and settings.supabase_service_key:
            try:
                from app.storage.supabase_client import auto_save_completed_job
                auto_save_completed_job(job_id)
            except Exception as e:
                logger.warning(f"saved_reels auto-save failed for {job_id}: {e}")


def _handle_progress_event(job_id: str, event):
    """Update Redis job progress based on pipeline events."""
    from app.workers.job_store import update_job_progress

    stage_order = {
        "download": (0, 20),
        "scene_detection": (1, 40),
        "frame_extraction": (2, 55),
        "vision_analysis": (3, 80),
        "places_enrichment": (4, 95),
    }

    if event.event_type == "stage_start":
        stage = event.data.get("stage", "")
        if stage in stage_order:
            idx, pct = stage_order[stage]
            update_job_progress(job_id, stage=stage, percent=pct)
            # Mirror to reel_jobs — only on stage transitions (~5 writes/job)
            _update_supabase_job(job_id, status="processing", progress=pct)

    elif event.event_type == "completed":
        update_job_progress(job_id, stage="completed", percent=100)

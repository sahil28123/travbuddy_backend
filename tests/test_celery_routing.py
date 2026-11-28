"""
Which deployment's worker may run which task.

This codebase runs twice — once against the marketplace Supabase project, once
against the hotel OS's — and one process has one `SUPABASE_URL`. A task
executed by the wrong deployment reads the wrong database.

Celery decides that, not the code that calls `.delay()`. Without routing every
task lands on the single default queue both workers consume, and whichever
worker grabs it first runs it. Sharing one Redis makes the collision certain
rather than likely, and the StayOS service is meant to be configured by copying
the marketplace's environment — `REDIS_URL` included.

The asymmetry in these tests is deliberate and is the whole design: the
marketplace worker is **already running in production with no `-Q`**, so its
tasks must keep landing on `celery`. Moving them to a symmetrical
`marketplace` queue would read better in a diagram and would stop reels, reco
and partner commissions dead on the next deploy.
"""

import pytest

from app.workers.celery_app import celery_app

HOTEL_OS_TASKS = [
    "travbuddy.messaging.process_conversation",
    "travbuddy.marketing.run_campaign",
    # Scheduled by the marketplace's beat — the only one that runs a beat — and
    # executed only by the StayOS worker. It reads `conversations`, which the
    # marketplace project has never had.
    "travbuddy.messaging.sweep_waiting_guests",
    # Reads `room_inventory` and `reservations` per property and reports drift.
    # Hotel tables throughout, so it must reach the StayOS worker — a
    # reconciliation run against the marketplace database would find every
    # hotel's inventory missing and say so hourly.
    "travbuddy.pms.reconcile_inventory",
    # The calendar out to Repull, and the nightly pass that queues it. Both
    # read `room_inventory` / `repull_room_map`, which only the hotel OS has.
    "travbuddy.pms.repull_push",
    "travbuddy.pms.repull_resync_all",
]

MARKETPLACE_TASKS = [
    "travbuddy.analyze_reel",
    "travbuddy.partners.revert_intro_commission",
    "travbuddy.reco.refresh_expected_price",
    "travbuddy.reco.refresh_destination_priors",
    "travbuddy.reco.fit_centroids",
    "travbuddy.reco.embed_and_profile",
    "travbuddy.reco.embed_hotel",
    "travbuddy.reco.profile_hotel",
]


def queue_for(task_name: str) -> str:
    route = celery_app.amqp.router.route({}, task_name)
    queue = route.get("queue")
    return getattr(queue, "name", queue) or celery_app.conf.task_default_queue


@pytest.mark.parametrize("task", HOTEL_OS_TASKS)
def test_hotel_os_tasks_go_to_the_hotelos_queue(task):
    """
    The direction that hurts. A guest's WhatsApp reply run by the marketplace
    worker looks for `conversations` in a project that has never had that
    table, and fails in a service nobody is watching.
    """
    assert queue_for(task) == "hotelos"


@pytest.mark.parametrize("task", MARKETPLACE_TASKS)
def test_marketplace_tasks_stay_on_the_default_queue(task):
    """
    Not `marketplace`. The production worker consumes `celery` today and its
    start command is not part of this change.
    """
    assert queue_for(task) == "celery"


def test_the_two_sets_never_share_a_queue():
    assert not (
        {queue_for(t) for t in HOTEL_OS_TASKS}
        & {queue_for(t) for t in MARKETPLACE_TASKS}
    )


def test_everything_beat_schedules_is_routed_to_a_deployment_that_can_run_it():
    """
    Beat runs in the marketplace deployment, and *enqueueing* is not
    *executing*. `revert_intro_commission` is unrouted, so beat's own worker
    picks it up — correct, because it writes to `partners`, a table that also
    exists meaning something entirely different in the hotel OS. The waiting
    sweep is routed to `hotelos`, so beat publishes it and the StayOS worker —
    the one process with StayOS's `SUPABASE_URL` — is the only one that can
    take it.

    The rule this holds is therefore not "no hotel-OS task in the schedule" but
    the thing that rule was standing in for: **nothing may be scheduled that
    the worker consuming its queue cannot run.** An entry added here with no
    route is one the marketplace worker will execute against the marketplace
    database, whatever it was written for.

    One operational consequence, recorded in `DEPLOYING_STAYOS.md`: the StayOS
    sweep stops if the *marketplace* beat stops. It is a scheduled reminder
    rather than part of any guest's path, so that degrades to no reminders
    rather than to anything a guest sees.
    """
    for name, entry in celery_app.conf.beat_schedule.items():
        task = entry["task"]
        if task in HOTEL_OS_TASKS:
            assert queue_for(task) == "hotelos", (
                f"{name} would be run by the marketplace worker against the "
                "marketplace database"
            )
        else:
            assert task in MARKETPLACE_TASKS, f"{name} is scheduled and unclassified"


def test_every_registered_task_is_routed_somewhere_a_worker_listens():
    """
    A task routed to a queue nobody consumes never runs, and nothing errors.
    The two queues in production are `celery` (marketplace worker, no -Q) and
    `hotelos` (StayOS worker, -Q hotelos).
    """
    listened = {"celery", "hotelos"}
    for task in HOTEL_OS_TASKS + MARKETPLACE_TASKS:
        assert queue_for(task) in listened, f"{task} lands on a queue nobody reads"


# ── the scheduler that was never deployed ───────────────────────────────────

class TestTheBeatThatOwnsTheSweep:
    """
    A scheduled task is not a running task, and this pair was never connected.

    `stayos-waiting-guests` has been in `beat_schedule` since the escalation
    work, carrying a comment that the marketplace beat would enqueue it and the
    StayOS worker would run it off the `hotelos` queue. That reasoning assumed
    one broker.

    It is not one broker. `docker-compose.stayos.yml` brings up its own Redis
    and points its api and worker at it, so the marketplace beat enqueues onto
    the marketplace broker and the StayOS worker listens to the StayOS broker.
    The task has never been delivered. Nothing errors — the guest simply waits,
    and production had zero rows in `agent_escalations`, ever.

    Two things are asserted here and neither is about Celery: that a StayOS
    beat schedules the sweep, and that it schedules **nothing that writes to a
    marketplace table**. The second is why there was no beat service to begin
    with, and it is still the right worry.
    """

    def test_a_stayos_beat_schedules_the_sweep(self, monkeypatch):
        from app.workers import celery_app as mod

        monkeypatch.setenv("CELERY_BEAT_ROLE", "stayos")
        assert "stayos-waiting-guests" in mod._beat_schedule()

    def test_a_stayos_beat_schedules_nothing_that_writes_to_the_marketplace(
        self, monkeypatch,
    ):
        """
        `revert_intro_commission` writes hourly to `partners`, a table that also
        exists and means something entirely different in the hotel OS. A StayOS
        beat running it would point it at a hotel database.
        """
        from app.workers import celery_app as mod

        monkeypatch.setenv("CELERY_BEAT_ROLE", "stayos")
        scheduled = {e["task"] for e in mod._beat_schedule().values()}

        assert "travbuddy.partners.revert_intro_commission" not in scheduled
        for task in scheduled:
            # **Two prefixes, not one.** `travbuddy.messaging.*` was the
            # whole of StayOS on the beat until inventory reconciliation
            # joined it, and `travbuddy.pms.*` is equally the hotel side —
            # renaming a reconciliation task "messaging" to satisfy a
            # prefix check would be the dishonest way past this. What the
            # rule is actually for is keeping *marketplace* tasks off a
            # beat pointed at a hotel database, and that is unchanged.
            assert task.startswith(("travbuddy.messaging.", "travbuddy.pms.")), (
                f"{task} is on the StayOS beat and belongs to neither the\n"
                f"messaging nor the hotel side"
            )

    def test_a_marketplace_beat_is_unchanged_by_any_of_this(self, monkeypatch):
        """
        The deployment that is already running must not need a redeploy, and
        must not lose an entry it has been firing for months.
        """
        from app.workers import celery_app as mod

        monkeypatch.setenv("CELERY_BEAT_ROLE", "marketplace")
        scheduled = set(mod._beat_schedule())

        assert scheduled == {
            "reco-expected-price-nightly", "reco-destination-priors-nightly",
            "reco-fit-centroids-weekly", "partner-intro-commission-hourly",
        }

    def test_the_default_is_everything(self, monkeypatch):
        """
        Unset means "as it was". A deployment that never learns about this
        variable keeps the behaviour it had, which is what makes the change
        safe to ship before any compose file is updated.
        """
        from app.workers import celery_app as mod

        monkeypatch.delenv("CELERY_BEAT_ROLE", raising=False)
        every = set(mod._beat_schedule())
        assert "stayos-waiting-guests" in every
        assert "partner-intro-commission-hourly" in every

    def test_an_unrecognised_role_schedules_everything_rather_than_nothing(
        self, monkeypatch,
    ):
        """
        A typo in a compose file must not silently switch a scheduler off.
        Failing loud is not available to a process that has to boot, so it
        fails *safe* — everything runs, which is the old behaviour, and the
        warning says so.
        """
        from app.workers import celery_app as mod

        monkeypatch.setenv("CELERY_BEAT_ROLE", "stayoss")
        assert "stayos-waiting-guests" in mod._beat_schedule()

    def test_the_two_roles_together_are_the_whole_schedule(self, monkeypatch):
        """
        No entry may belong to neither role — that is the drift this split
        exists to prevent, and it is exactly how the sweep went missing: a task
        was added to a shared schedule that one deployment could not reach.
        """
        from app.workers import celery_app as mod

        monkeypatch.setenv("CELERY_BEAT_ROLE", "stayos")
        stayos = set(mod._beat_schedule())
        monkeypatch.setenv("CELERY_BEAT_ROLE", "marketplace")
        marketplace = set(mod._beat_schedule())
        monkeypatch.setenv("CELERY_BEAT_ROLE", "all")
        everything = set(mod._beat_schedule())

        assert stayos | marketplace == everything
        assert not (stayos & marketplace), "an entry is scheduled by both beats"

    def test_the_stayos_compose_actually_runs_a_beat(self):
        """
        The half that is not Python. A schedule nothing executes is what this
        whole class is about, so the compose file is read rather than trusted.
        """
        import pathlib

        import yaml

        root = pathlib.Path(__file__).resolve().parent.parent
        compose = yaml.safe_load((root / "docker-compose.stayos.yml").read_text())
        services = compose["services"]

        assert "beat" in services, (
            "the StayOS stack has no beat service, so nothing fires the "
            "waiting-guest sweep and a waiting guest reaches nobody"
        )
        beat = services["beat"]
        assert "beat" in beat["command"]
        assert "CELERY_BEAT_ROLE=stayos" in beat["environment"], (
            "the StayOS beat is not scoped, so it would also run "
            "revert_intro_commission against a hotel database"
        )
        # Same broker as the worker that has to receive the task.
        assert any("REDIS_URL" in e for e in beat["environment"])
        worker_redis = [e for e in services["worker"]["environment"] if "REDIS_URL" in e]
        beat_redis = [e for e in beat["environment"] if "REDIS_URL" in e]
        assert worker_redis == beat_redis, (
            "beat and worker are on different brokers, which is the original "
            "bug in a new place"
        )

"""
The tool registry, and the one path through it.

Every tool call in the system goes through `execute()`. Nothing calls a tool
function directly — not the runtime, not the API, not a test — because
`execute()` is where permission, autonomy, approval, validation and the audit
record all happen, and a second entrance would be a way to skip all five.

    execute()
      ├─ resolve the tool          unknown name → refused, told to the model
      ├─ permissions.check()       grant + scope
      ├─ validate arguments        pydantic, the same schema the model was shown
      ├─ precheck()                the property's own rules, before anything queues
      ├─ autonomy.decide()         execute · approve · refuse
      ├─ revalidate()              approval path only: is it still true?
      ├─ run the function          which does the work and nothing else
      └─ return a ToolOutcome      which the caller writes to agent_actions

Tool bodies contain no security logic at all. That is the point: a tool author
cannot forget a check that lives somewhere they never edit.
"""
import hashlib
import inspect
import json
import logging
import time
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from app.db.client import get_db, run_db
from app.services.ai import autonomy, permissions, toolsets
from app.services.ai.context import AgentContext
from app.services.ai.errors import (
    AgentError,
    ApprovalRequired,
    ToolInputInvalid,
    ToolNotFound,
    ToolUnavailable,
)
from app.services.ai.schemas import ToolOutcome, ToolSpec
from app.services.pms import money

logger = logging.getLogger(__name__)

REGISTRY: dict[str, ToolSpec] = {}


def tool(
    *,
    name: str,
    description: str,
    input_model: type[BaseModel],
    permission: str | None,
    autonomy_floor: int,
    mutating: bool,
    usage_type: str,
    proposal: bool = False,
    result_type: str = "none",
    creates_result: bool = False,
    always_requires_approval: bool = False,
    scoped_to_conversation: bool = False,
    requires: tuple[str, ...] = (),
    when_deferred: str | None = None,
    precheck: Callable | None = None,
    revalidate: Callable | None = None,
    verify: Callable | None = None,
) -> Callable:
    """
    Declare a tool.

    Every field is required rather than defaulted, `usage_type` included. A tool
    that ships without a metering dimension is one somebody has to retro-fit
    later against a rate card that already exists, and the retro-fit is where
    the billing gap comes from.
    """

    def decorator(fn):
        if name in REGISTRY:
            raise RuntimeError(f"tool {name} is already registered")
        REGISTRY[name] = ToolSpec(
            name=name,
            description=description,
            input_model=input_model,
            permission=permission,
            autonomy_floor=autonomy_floor,
            mutating=mutating,
            proposal=proposal,
            usage_type=usage_type,
            fn=fn,
            result_type=result_type,
            creates_result=creates_result,
            always_requires_approval=always_requires_approval,
            scoped_to_conversation=scoped_to_conversation,
            requires=tuple(requires),
            when_deferred=when_deferred,
            precheck=precheck,
            revalidate=revalidate,
            verify=verify,
        )
        return fn

    return decorator


def get(name: str) -> ToolSpec:
    spec = REGISTRY.get(name)
    if spec is None:
        raise ToolNotFound(f"no tool named {name!r}")
    return spec


# What a channel has to be able to do, for the tools that need it to.
#
# **Found in production, on a hotel's own booking page.** A visitor asked about
# five rooms through the website chat; the model reached for `send_message` to
# ask which dates, and the tool raised
#
#     ValueError: no messaging policy for channel 'website'
#
# `registry.execute` correctly turned that into "that lookup did not come back
# — call request_human_handoff", the model did, and the highest-intent guest on
# the highest-intent surface a hotel has was told somebody would get back to
# them. Only `whatsapp` and `instagram` have a policy; `website`, `internal`,
# `phone` and `email` do not, and the model was offered the tool on all six.
#
# It is not that the reply had nowhere to go — web chat returns `result.reply`
# in the HTTP response and the staff console renders it — it is that a *second*
# way of speaking was on the menu and only one of them worked. This module's
# own docstring says the model is never shown a tool it cannot call; this is
# that rule applied to a dimension it did not previously cover.
#
# **The first version of this fix was a set of tool names living here, and it
# was wrong on the day it was written.** It held `send_message`, because that
# is the tool the production incident named. `send_template` and
# `send_room_photos` reach `messaging.send` too, so both were still offered on
# all six channels — a guest on a hotel's booking page could be sent room
# photographs that raised instead of arriving. A list of names kept in the
# module that filters them is a list nobody updates when they add a tool
# somewhere else; the requirement now travels on the tool, next to the code
# that has it, and `tests/test_channel_capability.py` re-derives it from the
# tool bodies so a missing declaration fails rather than ships.

# The capability names a tool may declare in `requires`.
OUTBOUND = "outbound"      # reaches messaging.send; the channel needs a policy
TEMPLATES = "templates"    # sends an approved template; Instagram has none
# Takes the guest off the channel they wrote on: a direct booking, a payment
# link, a link to the hotel's own pages, a cancellation done here rather than
# where the booking lives. Every channel allows it except an OTA thread —
# Airbnb, Booking.com, Vrbo and tiket.com forbid moving a guest off-platform,
# Airbnb rejects messages carrying off-platform links outright, and a booking
# the OTA sold can only be cancelled on the OTA (cancelling it here would leave
# the guest booked there and the room free here).
OFF_PLATFORM = "off_platform"

CAPABILITIES = (OUTBOUND, TEMPLATES, OFF_PLATFORM)


def channel_capabilities(channel: str | None) -> frozenset[str]:
    """
    What this channel can actually carry.

    Derived from `messaging/policy` rather than restated, so a channel that
    gains a provider tomorrow gains its tools without anybody editing this file
    — and one that loses it loses them the same way.

    An unknown channel gets nothing, which is the conservative direction: a
    channel nobody has thought about is not offered a tool that would raise on
    it, and the agent answers in prose, which every channel can carry.
    """
    from app.services.messaging import policy as channel_policy

    name = channel or channel_policy.WHATSAPP
    if name not in channel_policy.channels():
        # No provider (the staff console, web chat, phone, email): nothing to
        # send through, but nothing forbids a direct booking either.
        return frozenset({OFF_PLATFORM})
    rules = channel_policy.for_channel(name)
    able = {OUTBOUND}
    if rules.supports_templates:
        able.add(TEMPLATES)
    if rules.allows_off_platform:
        able.add(OFF_PLATFORM)
    return frozenset(able)


def _usable_on(spec: ToolSpec, ctx: AgentContext) -> bool:
    """Whether this tool can actually run on this turn's channel."""
    if not spec.requires:
        return True
    return set(spec.requires) <= channel_capabilities(ctx.channel)


def available_for(ctx: AgentContext) -> list[ToolSpec]:
    """
    The tools this agent may call, in registration order.

    This is also what the model is shown. An agent is not told about a tool it
    has no grant for — describing a capability and then refusing it wastes a
    turn and teaches the model to keep trying — nor about one its *channel*
    cannot carry, which is the same rule and was the same bug.
    """
    return [
        spec for nm, spec in REGISTRY.items()
        if nm in ctx.tool_grants and ctx.tool_grants[nm].get("enabled", True)
        and _usable_on(spec, ctx)
    ]


def describe_for_provider(ctx: AgentContext, task: str | None = None,
                          *, opening: bool = False) -> list[dict]:
    """
    The tool list in the shape a chat-completions API expects.

    `task` narrows *what is described*, never what is permitted — see
    services/ai/toolsets.py. Left out, everything granted is described, which
    is what every caller did before the narrowing existed and what the
    unclassifiable case still gets.

    `opening` says the classifier recognised a bare greeting on an empty
    thread, which narrows harder than any task does.
    """
    specs = available_for(ctx)
    if task or opening:
        shown = toolsets.for_task(task, frozenset(s.name for s in specs),
                                  opening=opening)
        specs = [s for s in specs if s.name in shown]
    return [
        {
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.json_schema(),
            },
        }
        for spec in specs
    ]


# ── one guest message, one effect ────────────────────────────────────────────
#
# `runtime` already refuses to run the same mutating call twice inside one turn
# (`called_writes`), and that set is a local variable which dies with the turn.
# Across turns there was nothing, and there is a real ordering that needs it:
#
#     run_turn  →  create_booking_payment_link  →  a Razorpay order exists
#               →  _mark_answered  ✗  (a reset connection, an OOM, a redeploy)
#               →  Celery redelivers (acks_late, autoretry_for=Exception)
#               →  the message is still pending, so the whole turn runs again
#               →  a second Razorpay order for the same stay
#
# The stamp cannot simply be made unfailable: if `_mark_answered` swallowed its
# error the same message would come straight back from `_pending` on the next
# iteration of the drain loop and be answered again inside the same task. So the
# mark is allowed to fail, and the tools underneath it are made safe when it
# does.
#
# **Scoped to the guest's message, not to a clock.** A time window would have to
# choose between covering a slow retry and honouring a guest who genuinely asks
# for the same thing twice. The message id needs no such choice: answering
# message X again is a replay by definition, and a guest's next message is a new
# scope however fast it arrives.
#
# Reads are deliberately not guarded. Re-running `get_quote` costs a moment and
# returns the truth; returning the first run's price would hand a replay a
# number fetched before whatever went wrong in between.

def idempotency_key(scope: str, tool: str, arguments: dict) -> str:
    """
    The identity of one mutating call made in service of one guest message.

    Built from the *validated* arguments rather than the model's raw ones, so
    `{"units": "5"}` and `{"units": 5}` are the same call — which they are, and
    which the model is entirely capable of alternating between across a retry.
    """
    canonical = json.dumps(arguments or {}, sort_keys=True, default=str)
    return hashlib.sha256(
        "\u0000".join((scope, tool, canonical)).encode("utf-8")
    ).hexdigest()


def _already_done(key: str, ctx: AgentContext) -> ToolOutcome | None:
    """
    The recorded result of this exact call, if it already succeeded.

    Only `executed` rows. A refusal must stay reachable — the second attempt may
    come from a run at a different autonomy level or after a grant was fixed,
    and answering it from a stale refusal would make the fix invisible until the
    guest wrote again. An `awaiting_approval` row is likewise not a result: the
    work has not happened, and the replay should queue behind the same human.
    """
    rows = (
        get_db().table("agent_actions")
        .select("tool,output,result_type,result_id")
        .eq("property_id", ctx.property_id)
        .eq("idempotency_key", key)
        .eq("status", "executed")
        .limit(1)
        .execute()
        .data
        or []
    )
    if not rows:
        return None
    row = rows[0]
    return ToolOutcome(
        tool=row["tool"],
        status="replayed",
        output=row.get("output"),
        result_type=row.get("result_type") or "none",
        result_id=row.get("result_id"),
        idempotency_key=key,
        # The model is told it is done, and not told it was done twice. Anything
        # else invites it to explain a retry to a guest who never saw one.
        model_message="",
    )


async def execute(
    name: str,
    arguments: dict[str, Any],
    ctx: AgentContext,
    *,
    approved: bool = False,
) -> ToolOutcome:
    """
    The only way to run a tool. Never raises for an expected refusal — a refusal
    is an outcome the model must see and answer, not an exception that ends the
    turn.

    `approved=True` means a human with the 'agents' permission has stood in for
    the autonomy level on this specific action. It skips **only** the autonomy
    gate. Permission, grant, argument validation, the property's own business
    rules and the tool's revalidation all still run — a manager can lend their
    judgement, not their access, and cannot approve a tool the agent was never
    granted or a price the property never authorised.
    """
    started = time.monotonic()

    # Computed once the arguments are validated, and stamped on *every* outcome
    # from that point — refusals and failures included, not only successes.
    #
    # **Found by mutation testing.** Deleting `.eq("status", "executed")` from
    # `_already_done` broke no test, because no refused outcome carried a key
    # for the lookup to find: the filter was redundant by accident rather than
    # by design. That is a guard nobody would notice losing.
    #
    # Recording it on a refusal is also the better audit trail. "Was this the
    # same call we turned down two minutes ago" is a question the action rows
    # could not answer, and it is exactly the question somebody asks when a
    # permission looks wrong.
    key: str | None = None

    def elapsed() -> int:
        return int((time.monotonic() - started) * 1000)

    def refusal(exc: AgentError, tool_name: str = name) -> ToolOutcome:
        # `refusal_reason` keeps the whole truth for the audit row; the model
        # gets `model_message`, which is the same sentence for every refusal
        # somebody wrote deliberately and a safe one for the catch-all below.
        # See `errors.AgentError` for why those had to stop being one field.
        return ToolOutcome(
            tool=tool_name,
            status=exc.action_status,
            input=arguments,
            refusal_reason=exc.reason,
            model_message=exc.model_message,
            guest_message=exc.guest_message,
            latency_ms=elapsed(),
            error=exc.reason if exc.action_status == "failed" else None,
            idempotency_key=key,
        )

    try:
        spec = get(name)
    except AgentError as exc:
        return refusal(exc)

    try:
        permissions.check(spec, ctx)
    except AgentError as exc:
        return refusal(exc)

    # Validated against the same model the provider was shown, so an argument
    # the model hallucinated fails here rather than three layers down as a 500.
    try:
        args = spec.input_model(**(arguments or {}))
    except ValidationError as exc:
        detail = "; ".join(
            f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:4]
        )
        return refusal(ToolInputInvalid(f"{spec.name} arguments rejected — {detail}"))

    # Business validation, before the autonomy gate and therefore before
    # `approved` can matter. A discount above what the property authorises is
    # not something a manager should be offered the chance to approve — it is
    # something the agent must be told it cannot ask for, while the guest is
    # still in the conversation.
    if spec.precheck is not None:
        try:
            await spec.precheck(args, ctx)
        except AgentError as exc:
            return refusal(exc)

    # Named before the gates so a refusal carries it; looked up after them, so
    # a replay is a shortcut past the work and never past a rule.
    if spec.mutating and ctx.answering:
        key = idempotency_key(ctx.answering, spec.name, args.model_dump(mode="json"))

    verdict = autonomy.EXECUTE if approved else autonomy.decide(spec, ctx)
    if verdict == autonomy.REFUSE:
        return refusal(
            AgentError(
                autonomy.explain(spec, ctx),
                "That's not something I can do here — let me pass you to the team.",
            )
        )
    if verdict == autonomy.APPROVE:
        # Recorded, queued, and reported honestly to both the model and the
        # guest. Nothing has happened yet and nobody is told it has.
        #
        # The audit row and the model are told different things, deliberately.
        # `refusal_reason` stays the system's own sentence, because that is what
        # a person reviewing the queue needs. `model_message` carries the
        # tool's `when_deferred` instead, because the model's only useful
        # response to "this is queued" is to take the other road — and told the
        # internal rule instead, it apologises to the guest for it.
        exc = ApprovalRequired(autonomy.explain(spec, ctx))
        out = refusal(exc)
        out.input = args.model_dump(mode="json")
        out.model_message = spec.when_deferred or (
            "Not done yet — someone from the hotel has to approve this, which "
            "is not instant. Do not apologise and do not say it is done. Tell "
            "the guest it has gone to the team, and offer anything else you "
            "can do right now."
        )
        return out

    # Between the proposal and the approval, the hotel moved. Checked here
    # rather than inside the tool so a tool author cannot forget it, and only on
    # the approval path — a live turn was never stale to begin with.
    if approved and spec.revalidate is not None:
        try:
            await spec.revalidate(args, ctx)
        except AgentError as exc:
            return refusal(exc)

    # **Below every gate, and that placement is the point.**
    #
    # Permission, argument validation, the precheck and the autonomy verdict
    # have all already run on this attempt. A replay is a shortcut past the
    # *work*, never past a rule — so a grant revoked between the two attempts
    # still refuses, and the guard can only ever answer a call that would have
    # been allowed to run anyway.
    if key is not None:
        try:
            done = await run_db(_already_done, key, ctx)
        except Exception:  # noqa: BLE001
            # The lookup is a guard, not the guest's answer. If it cannot run,
            # the turn proceeds as it did before this existed — which is the
            # behaviour it improves on, not one it depends on.
            logger.warning("idempotency lookup failed for %s", spec.name, exc_info=True)
            done = None
        if done is not None:
            logger.info("%s was already done for message %s — returning its result",
                        spec.name, ctx.answering)
            done.input = args.model_dump(mode="json")
            done.latency_ms = elapsed()
            return done

    try:
        result = spec.fn(args, ctx)
        if inspect.isawaitable(result):
            result = await result
    except AgentError as exc:
        return refusal(exc)
    except Exception as exc:  # noqa: BLE001 — the boundary is the point
        # A tool that breaks must never look like a tool that answered. The
        # model is told the system is unavailable, which routes to a human,
        # rather than being handed nothing and left to guess.
        #
        # **And it is told that in English, not in Python.** This used to hand
        # the model `f"{spec.name} failed: {type(exc).__name__}: {exc}"` — a
        # tool name, an exception class and whatever a Supabase client had to
        # say about a schema cache. The model has no way to act on any of that
        # and every incentive to relay it, which is how a paying guest came to
        # be told *"quote generate karne mein takleef ho rahi hai"*. The whole
        # string is still on the action row, where somebody can debug it.
        logger.exception("tool %s failed", spec.name)
        return refusal(ToolUnavailable(
            f"{spec.name} failed: {type(exc).__name__}: {exc}",
            model_message=(
                "That lookup did not come back. This is our system, not "
                "anything the guest did. Do not describe the error, do not "
                "guess the answer and do not try again — tell them you are "
                "checking and call request_human_handoff."
            ),
        ))

    # A tool may return (payload, result_id) when it created something the trace
    # should point at.
    result_id: str | None = None
    if isinstance(result, tuple) and len(result) == 2:
        result, result_id = result

    # **Every paise the model can see, beside the words for it.**
    #
    # Thirty guardrail-blocked production runs traced back to the model
    # formatting money itself, and the shape was always the same: a tool
    # returned `balancePaise: 3440640` and nothing that said `₹34,406`, so the
    # model divided by a hundred and wrote the paise out as decimals. The figure
    # was right; it was the *rendering* it had to invent, and inventing a
    # rendering is one keystroke from inventing a number.
    #
    # `get_quote` was given display strings by hand after the first of these,
    # and the other money-returning tools were not — `frontdesk` had ten paise
    # fields and no display calls at all. Doing it here rather than in each tool
    # is the difference between fixing seven files and fixing the next tool
    # somebody writes. See services/pms/money.with_display.
    result = money.with_display(result)

    # **Did it actually happen?**
    #
    # The last gate, and the only one that runs after the write. Everything
    # above this line decides whether the tool *may* run; this asks whether it
    # *did*. See `ToolSpec.verify` — the rule the agent must never break is
    # "reason → mutate → verify → respond", and a rule enforced only by a system
    # prompt is a rule the model may skip on the turn it matters.
    #
    # A verification failure is `failed`, not `refused`: nothing was denied, the
    # world is in a state nobody chose, and the model is handed the observed
    # state so it reports the truth instead of the request.
    verified: bool | None = None
    if spec.verify is not None:
        try:
            evidence = await spec.verify(args, ctx, result_id)
        except AgentError as exc:
            logger.error(
                "unverified mutation: %s claimed %s and the re-read disagreed (%s)",
                spec.name, result_id or "a change", exc.reason,
            )
            out = refusal(exc)
            out.input = args.model_dump(mode="json")
            out.verified = False
            out.result_type = spec.result_type
            out.result_id = str(result_id) if result_id else None
            return out
        except Exception as exc:  # noqa: BLE001
            # The check itself broke. That is not proof the write failed, and
            # treating it as such would hand off every turn the moment a read
            # replica hiccups — so the mutation stands and the evidence is
            # absent, which is exactly what `verified=None` means.
            logger.warning("verification of %s could not run: %s", spec.name, exc)
        else:
            verified = True
            if isinstance(result, dict) and isinstance(evidence, dict) and evidence:
                # Shown to the model, so it answers from the record rather than
                # from what it asked for.
                result = {**result, "verified": evidence}

    return ToolOutcome(
        tool=spec.name,
        status="executed",
        input=args.model_dump(mode="json"),
        output=result,
        result_type=spec.result_type,
        result_id=str(result_id) if result_id else None,
        model_message="",
        latency_ms=elapsed(),
        verified=verified,
        # Written to the action row, which is where the next attempt looks.
        idempotency_key=key,
    )


def load_all() -> None:
    """
    Import the tool modules so their decorators run.

    Explicit rather than a package scan: the registry is a security surface and
    what is in it should be a list somebody reads, not whatever happened to be
    on the import path.
    """
    from app.services.ai.tools import (  # noqa: F401
        booking,
        communication,
        frontdesk,
        hotel,
        marketing,
        operations,
        owner,
        pricing,
        read,
        sales,
        stay,
    )

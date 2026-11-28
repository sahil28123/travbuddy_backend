"""
A tool the channel cannot carry is a tool the model must not be shown.

**Found by probing the deployed backend, not by any test here.** A visitor
asked the website chat on a hotel's own booking page for five rooms. The model
called `get_group_rates`, got its answer, and then reached for `send_message`
to ask which dates — and the tool raised

    ValueError: no messaging policy for channel 'website'

`registry.execute` did exactly the right thing with that: the model was told the
lookup had not come back and to call `request_human_handoff` rather than guess.
So the highest-intent guest on the highest-intent surface a hotel has was told
somebody would get back to them, on a turn where nothing was actually wrong with
the hotel, the rooms or the price.

Only `whatsapp` and `instagram` have an outbound policy. `website`, `internal`,
`phone` and `email` do not — and `send_message` was on the menu for all six,
because the menu had never been filtered by anything except grants and task.

`registry`'s own docstring already stated the rule this violates:

    **The model never sees a tool it cannot call.**

That was true of permissions and false of channels. This file is that rule
applied to the dimension it had missed, in both directions — because the fix
that removes `send_message` everywhere would be worse than the bug.
"""
import textwrap

import pytest

from app.services.ai import registry
from app.services.ai.context import AgentContext, AgentPrincipal
from app.services.messaging import policy

registry.load_all()

PROPERTY = "cccccccc-cccc-4ccc-8ccc-cccccccccccc"
PARTNER = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"


def ctx_on(channel: str) -> AgentContext:
    principal = AgentPrincipal(
        agent_id="agent-1", agent_type="guest", partner_id=PARTNER,
        property_id=PROPERTY, display_name="Mandala", autonomy_level=3,
        scopes=frozenset({"tasks", "messaging", "rates", "frontdesk"}),
    )
    return AgentContext(
        principal=principal, property_id=PROPERTY, partner_id=PARTNER,
        channel=channel, actor="guest",
        tool_grants={t: {"tool_name": t, "enabled": True} for t in registry.REGISTRY},
    )


def shown_on(channel: str) -> set[str]:
    return {s.name for s in registry.available_for(ctx_on(channel))}


# ── the audit §8 asked for: every tool, every channel ───────────────────────
#
# The first version of this file tested one tool against six channels, because
# `send_message` was the tool the production incident named. That was the same
# mistake in miniature as the one it was fixing: a rule stated about the case
# somebody happened to hit rather than about the class it belongs to.
#
# `send_template` and `send_room_photos` both reach `messaging.send` and both
# were still on the menu everywhere. Neither was found by reading this file —
# they were found by writing the derivation below and letting it disagree with
# the hand-maintained list.

ALL_CHANNELS = ("whatsapp", "instagram", "website", "internal", "phone", "email", "ota")


def sends_outbound(spec) -> bool:
    """
    Whether this tool's own body reaches `messaging.send`.

    Read out of the source rather than off the declaration, so the two are
    independent answers to the same question. A declaration that agrees with
    the code proves nothing on its own; a declaration checked against the code
    is what stops the next sending tool shipping without one.
    """
    import ast
    import inspect

    try:
        source = inspect.getsource(spec.fn)
    except (OSError, TypeError):  # pragma: no cover
        return False
    tree = ast.parse(textwrap.dedent(source))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            rendered = ast.unparse(node.func)
            if rendered.endswith("messaging.send") or rendered.endswith("service.send"):
                return True
    return False


@pytest.mark.parametrize("name", sorted(registry.REGISTRY))
def test_every_tool_that_sends_declares_that_it_needs_a_channel(name):
    """
    The derivation, run against every tool in the registry.

    This is the test that found the two the hand-list had missed. It fails on a
    tool that calls `messaging.send` without `requires=(OUTBOUND,)`, which is
    the only way a new sending tool can be written — so the next one cannot
    ship on six channels the way these three did.
    """
    spec = registry.REGISTRY[name]
    if sends_outbound(spec):
        assert registry.OUTBOUND in spec.requires, (
            f"{name} calls messaging.send and does not declare "
            f"requires=(registry.OUTBOUND,) — it will be offered on website, "
            f"internal, phone and email, where messaging.send raises"
        )


@pytest.mark.parametrize("channel", ALL_CHANNELS)
@pytest.mark.parametrize("name", sorted(registry.REGISTRY))
def test_every_tool_on_every_channel_is_either_shown_or_impossible(name, channel):
    """
    §8, stated exhaustively: **every tool × every channel**. Around 240 pairs,
    and each one has to be one of two things — shown, and its channel able to
    carry what it needs; or hidden, and unable.

    A pair that is shown *and* unable is the production bug. A pair that is
    hidden *and* able is a capability quietly lost, which is the failure mode
    of over-correcting and would be much harder to notice.
    """
    spec = registry.REGISTRY[name]
    shown = name in shown_on(channel)
    able = set(spec.requires) <= registry.channel_capabilities(channel)

    assert shown == able, (
        f"{name} on {channel}: shown={shown} able={able}. A tool that is shown "
        f"and unable raises in front of a guest; one that is hidden and able is "
        f"a capability the channel silently lost."
    )


def test_the_capability_of_a_channel_comes_from_the_messaging_policy():
    """
    Derived, not restated — the property the first version of this file had and
    that is worth keeping. Nobody edits `registry` when a channel gains a
    provider.
    """
    assert registry.channel_capabilities("whatsapp") == {
        registry.OUTBOUND, registry.TEMPLATES, registry.OFF_PLATFORM}
    # Instagram has no approved-template mechanism, so a tool needing one is
    # not offered there — the second dimension, which the name-list could not
    # express at all.
    assert registry.channel_capabilities("instagram") == {
        registry.OUTBOUND, registry.OFF_PLATFORM}
    # OTA threads (through Repull): sendable, no templates — and no window, which
    # is *not* WINDOWLESS. See `ChannelPolicy.windowed`. And never off-platform.
    assert registry.channel_capabilities("ota") == {registry.OUTBOUND}
    for channel in policy.WINDOWLESS:
        assert registry.channel_capabilities(channel) == {registry.OFF_PLATFORM}


OFF_PLATFORM_TOOLS = ("create_booking", "create_booking_payment_link", "cancel_booking",
                      "create_payment_link", "get_check_in_link", "get_hotel_page_link")


def test_an_ota_guest_is_never_taken_off_the_channel():
    """
    Airbnb, Booking.com, Vrbo and tiket.com forbid moving a guest off-platform,
    and Airbnb rejects a message carrying an off-platform link. A booking the OTA
    sold is cancelled on the OTA, not here. So on `ota` the agent answers, quotes
    and helps — and is not offered the tools that book, charge, link or cancel
    outside the channel.
    """
    on_ota = shown_on("ota")
    for name in OFF_PLATFORM_TOOLS:
        assert name not in on_ota, f"{name} offered in an OTA thread"
        assert name in shown_on("whatsapp"), f"{name} lost on WhatsApp"
    assert "send_message" in on_ota
    assert "get_booking_status" in on_ota


def test_the_template_tool_is_not_offered_where_there_are_no_templates():
    """
    Instagram. `send_template` there raises TEMPLATE_NOT_FOUND — a clean refusal
    rather than a crash, which is why it survived: the agent was told "this
    channel has no approved-template mechanism" and wasted a turn learning it,
    every time, on the channel where the window closing is most likely.
    """
    assert "send_template" not in shown_on("instagram")
    assert "send_template" in shown_on("whatsapp")
    assert "send_message" in shown_on("instagram"), (
        "narrowing templates must not take ordinary replies with it"
    )


# ── the bug ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("channel", sorted(policy.WINDOWLESS))
@pytest.mark.parametrize("tool", ["send_message", "send_template", "send_room_photos"])
def test_a_channel_that_cannot_send_is_not_offered_any_sending_tool(channel, tool):
    """
    All three, on all four. `send_template` and `send_room_photos` are the two
    the original fix missed.
    """
    assert tool not in shown_on(channel)


@pytest.mark.parametrize("channel", sorted(policy.WINDOWLESS))
def test_a_channel_that_cannot_send_is_not_offered_the_sending_tool(channel):
    """
    `website`, `internal`, `phone`, `email`. Every one of them would raise, and
    the model reaching for any of them ends the turn at a handoff.
    """
    assert "send_message" not in shown_on(channel), (
        f"send_message is on the menu for {channel!r}, where messaging.send "
        f"raises — the model calling it turns a working turn into a handoff"
    )


@pytest.mark.parametrize("channel", ["whatsapp", "instagram", "ota"])
def test_a_channel_that_can_send_still_gets_it(channel):
    """
    The half that would be a much worse regression. On the DM channels the
    agent replies *with the tool* on some paths, and removing it there would
    silence the product on its two busiest surfaces.
    """
    assert "send_message" in shown_on(channel)


def test_the_two_lists_agree_with_the_policy_module():
    """
    Derived, not restated. A channel that gains a policy tomorrow gains the
    tool without anybody editing this rule, and one that loses it loses the
    tool the same way.
    """
    can_send = set(policy.channels())
    for channel in can_send:
        assert "send_message" in shown_on(channel)
    for channel in policy.WINDOWLESS:
        assert channel not in can_send
        assert "send_message" not in shown_on(channel)


def test_every_channel_the_database_allows_is_decided_one_way_or_the_other():
    """
    The lesson `policy.WINDOWLESS` half-learned, finished.

    Any channel `conversations.channel` can hold must be either sendable or
    knowably not; a channel in neither list would raise inside
    `_usable_on` on a real guest turn, which is the failure this file exists
    for wearing a different hat.
    """
    stored = set(policy.channels()) | set(policy.WINDOWLESS)
    for channel in ("whatsapp", "instagram", "website", "internal", "phone", "email", "ota"):
        assert channel in stored, (
            f"{channel!r} can be stored on a conversation and is in neither "
            f"_POLICIES nor WINDOWLESS"
        )
        # And asking is safe rather than explosive.
        shown_on(channel)


# ── everything else is untouched ────────────────────────────────────────────

def test_only_the_sending_tool_is_affected():
    """
    Narrow on purpose. A website guest must still be able to have their rooms
    priced, their group checked and their booking read — the channel decides
    how the agent *speaks*, not what it may know.
    """
    website = shown_on("website")
    whatsapp = shown_on("whatsapp")

    assert whatsapp - website == {"send_message", "send_template",
                                  "send_room_photos"}, (
        f"the channel filter removed more than the sending tools: "
        f"{sorted(whatsapp - website)}"
    )
    for essential in ("get_group_rates", "negotiate_rate", "get_quote",
                      "get_booking_status", "request_human_handoff",
                      "cancel_booking", "get_cancellation_quote"):
        assert essential in website


def test_a_website_guest_can_still_be_answered():
    """
    The reply does not need the tool. `web_chat.ask` returns `result.reply` in
    the HTTP response and writes it to the thread — which is the designed path
    and the one that was already working on every turn where the model did not
    reach for `send_message`.
    """
    assert "request_human_handoff" in shown_on("website")
    assert "get_property_knowledge" in shown_on("website")


def test_the_filter_cannot_add_anything():
    """
    Same invariant `toolsets` holds: narrowing may only ever narrow. A filter
    that could widen would be a second permission system disagreeing with the
    first.
    """
    for channel in ("whatsapp", "instagram", "website", "internal", "phone", "email"):
        assert shown_on(channel) <= set(registry.REGISTRY)


def test_an_unknown_channel_is_treated_as_unable_to_send():
    """
    Conservative in the direction that cannot break a turn. A channel nobody
    has thought about does not get a tool that would raise on it; the agent
    answers in prose, which every channel can carry.
    """
    assert "send_message" not in shown_on("carrier-pigeon")

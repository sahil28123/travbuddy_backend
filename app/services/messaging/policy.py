"""
What a channel permits — asked, not assumed.

`service.send()` used to know three WhatsApp facts: that a closed window can be
reopened with an approved template, that the window is 24 hours, and that
connections live in `whatsapp_connections`. None of those is true of Instagram,
and none of them is a property of *messaging* — they are properties of a
channel. So they move here, and the shared path asks.

    service.send(conversation)
        └─ policy.for_channel(conversation["channel"])
               ├─ window_hours(conversation)   how long may we reply freely?
               ├─ supports_templates            is there a way to reopen it?
               └─ connections_table             where does the account live?

The alternative — `if channel == "instagram"` inside the send path — is the
thing this file exists to prevent. It reads fine with two channels and becomes
unmaintainable at four, and it puts channel rules in the one function that must
stay identical for every channel.

**Two genuinely different rules, encoded rather than commented:**

*Templates.* WhatsApp reopens a closed window with a template Meta approved in
advance. Instagram has no such mechanism. A caller asking for a template on
Instagram gets a refusal naming the reason, not a silent fallback to free text —
which would be a message sent outside a window Meta will reject anyway.

*The human-agent window.* Meta gives a **person** answering an Instagram
conversation seven days, against twenty-four hours for an automated reply. That
is not a quirk to paper over: it is exactly the difference between a bot and a
human agent, and this codebase already tracks which one is driving in
`conversations.handler`. So the Instagram window is a function of the
conversation, not a constant. WhatsApp's is a constant, and stays one.
"""
from dataclasses import dataclass, field
from typing import Callable

from app.config import settings

WHATSAPP = "whatsapp"
INSTAGRAM = "instagram"
# The staff console. A channel with no provider and nothing to send to —
# see WINDOWLESS below for why it is named here but has no policy row.
INTERNAL = "internal"
# The chat on the hotel's own booking page. Also windowless, and for a
# different reason worth stating: there is no provider *and* no rule to
# inherit. A 24-hour window exists because Meta bills by one; a guest sitting
# on a web page can be replied to for as long as the page is open, and after
# that there is nobody to deliver to at all.
WEBSITE = "website"
# A call and an email a member of staff logged against a guest. The database
# has always allowed both on `conversations.channel`; nothing writes one today,
# and the moment something does it must not be the Inbox that finds out.
PHONE = "phone"
EMAIL = "email"
# Guest messages on Airbnb, Booking.com, Vrbo and tiket.com, carried by Repull
# (https://repull.dev). One channel rather than four: the hotel's reply goes
# back through the same Repull conversation whichever OTA the guest booked on,
# and nothing about *sending* differs between them — Repull owns the per-OTA
# rules (Airbnb stripping a phone number, Vrbo queueing a busy thread).
OTA = "ota"


@dataclass(frozen=True)
class ChannelPolicy:
    channel: str

    # Where the account→property mapping lives, and the SECURITY DEFINER
    # function that reads it. Never a table read from the service: the resolver
    # is the only trusted route from an inbound payload to a hotel.
    #
    # `resolver_rpc=None` is the one exception, and it is the OTA channel:
    # `repull_connections` is written only by the server, keyed on a
    # `workspace_id` Repull minted, and the id arrives inside a webhook whose
    # HMAC has already been checked against our own secret. There is no
    # partner-supplied value in the lookup for a definer function to guard.
    connections_table: str
    resolver_rpc: str | None
    resolver_arg: str | None
    # The column on the connection row holding the provider's id for our end.
    account_column: str

    # Whether a closed window can be reopened at all.
    supports_templates: bool

    # What the hotel is charged for outbound. WhatsApp bills per *conversation*
    # (see usage.record_conversation); Instagram has no conversation billing
    # from Meta at all, so it is metered per message and the conversation-level
    # charge is opened by us on the first inbound rather than by a Meta
    # callback that never arrives.
    bills_per_conversation: bool
    conversation_usage_type: str | None = None

    # Given a conversation, how many hours of free-form reply are permitted.
    # A callable rather than an int because Instagram's answer depends on who
    # is answering.
    window_hours: Callable[[dict], int] = field(default=lambda _c: 24)

    # False for a channel with a provider and **no** reply window at all.
    #
    # Not the same thing as WINDOWLESS below, and the difference is the whole
    # reason this is a field. WINDOWLESS channels have no window because there
    # is nothing to send to — the staff console, a web page that has closed.
    # An OTA thread has no window because the OTA lets a host answer a guest
    # whenever they like, before, during and after the stay: it is *always*
    # open. Folding it into WINDOWLESS would have stripped the agent of its
    # sending tools on exactly the channel where it most needs them.
    windowed: bool = True

    # False where the channel's owner forbids taking the guest elsewhere — the
    # OTA thread. The agent is then not offered direct bookings, payment links,
    # links to the hotel's own pages or cancellations; see
    # `ai.registry.OFF_PLATFORM`.
    allows_off_platform: bool = True

    # Which `status` values on the connection row mean "send through this".
    # Meta connections say `connected`; a Repull workspace says `active`.
    live_statuses: tuple[str, ...] = ("connected", "pending")
    # The connection row's primary key. `repull_connections` is keyed on the
    # property it belongs to — one workspace per hotel — so it has no `id`.
    connection_key: str = "id"

    def human_is_handling(self, conversation: dict) -> bool:
        return (conversation.get("handler") or "ai") in ("human", "human_requested")


def _whatsapp_window(_conversation: dict) -> int:
    # Meta's rule, and a constant: 24 hours from the guest's last message,
    # whoever is replying. A human on WhatsApp gets no more room than the agent.
    return settings.whatsapp_service_window_hours


def _instagram_window(conversation: dict) -> int:
    """
    24 hours for an automated reply; 7 days once a person is handling it.

    This is Meta's human-agent allowance, and claiming it while an agent is
    still driving the conversation would be a policy violation dressed up as a
    longer window. `handler` is the server's own record of who took over — set
    by `take_over()` and `request_human()`, never by the model — which is why
    it is safe to key on.
    """
    return (settings.instagram_human_agent_window_hours
            if (conversation.get("handler") or "ai") in ("human", "human_requested")
            else settings.instagram_service_window_hours)


_POLICIES = {
    WHATSAPP: ChannelPolicy(
        channel=WHATSAPP,
        connections_table="whatsapp_connections",
        resolver_rpc="resolve_whatsapp_property",
        resolver_arg="p_phone_number_id",
        account_column="phone_number_id",
        supports_templates=True,
        bills_per_conversation=True,
        window_hours=_whatsapp_window,
    ),
    INSTAGRAM: ChannelPolicy(
        channel=INSTAGRAM,
        connections_table="instagram_connections",
        resolver_rpc="resolve_instagram_property",
        resolver_arg="p_account_id",
        account_column="instagram_account_id",
        supports_templates=False,
        # Meta does not bill Instagram per conversation, so nothing external
        # ever tells us a window opened. We open ours on the first inbound
        # message, which is also the only honest moment: it is our pricing
        # decision, not a pass-through of someone else's.
        bills_per_conversation=False,
        conversation_usage_type="instagram.conversation",
        window_hours=_instagram_window,
    ),
    OTA: ChannelPolicy(
        channel=OTA,
        connections_table="repull_connections",
        resolver_rpc=None,
        resolver_arg=None,
        account_column="workspace_id",
        # An OTA has no approved-template mechanism, and needs none: nothing
        # ever closes the thread.
        supports_templates=False,
        # Repull does not bill per conversation and neither does any OTA. The
        # agent's own turns are metered by the pipeline as they are everywhere.
        bills_per_conversation=False,
        conversation_usage_type=None,
        windowed=False,
        allows_off_platform=False,
        # `pending` is a workspace whose hotel has started Connect: a guest
        # message can only arrive once an account is attached, and refusing to
        # answer it over a status that lags by one webhook would be worse.
        live_statuses=("active", "pending"),
        connection_key="property_id",
    ),
}


# Channels that carry no window, as opposed to channels nobody has configured.
#
# The difference matters and is the reason this is a set rather than a policy
# row. `internal` is the staff console: no provider, no Meta, no 24-hour rule,
# nothing that leaves the process. Giving it a `ChannelPolicy` would mean a fake
# connection row and a window that is always open — a provider abstraction
# pretending to carry something that never travels, which `services/ai/console`
# refused to build for exactly this reason.
#
# But `for_channel` raising for it was wrong too. `list_conversations` calls
# `window_open()` on every row in the inbox, so one staff-console thread made
# the whole Inbox screen 500 with `no messaging policy for channel 'internal'`.
# The caller was asking a reasonable question — "is the window open" — about a
# channel whose honest answer is "there is no window", and got an exception.
#
# So the two cases are now distinguishable: a channel in here has no window and
# says so; a channel in neither place is one nobody has thought about, and that
# still raises.
# **`website` was the second one, and it broke the same screen.** The website
# chat was added after this set was written, its conversations carry
# `channel='website'`, and `list_conversations` walks every row — so a single
# guest asking a question on a hotel's own booking page made that hotel's
# Inbox 500 with `no messaging policy for channel 'website'`. The hotel could
# not read its own guest messages, and the failure was in the screen rather
# than in the channel that caused it, which is why it survived a launch audit:
# the conversation worked, the reply was sent, and the Inbox was the only thing
# that stopped.
#
# `phone` and `email` are here for the same reason and were found by writing
# that test rather than by anybody hitting them: the CHECK on
# `conversations.channel` has allowed both since it was written, nothing
# creates one today, and the first hotel to log a phone enquiry would have
# taken its own Inbox down. Neither carries a window — a call is over when it
# is over, and an email has no 24-hour rule to inherit.
#
# The lesson the `internal` fix half-learned: any channel the *database* can
# store has to be in `_POLICIES` or in here, and adding one without doing that
# is a screen-level 500 nobody sees until a guest triggers it.
# `tests/test_messaging_policy.py` asserts exactly that, against the CHECK
# constraint's own list.
WINDOWLESS = frozenset({INTERNAL, WEBSITE, PHONE, EMAIL})


def has_window(channel: str | None) -> bool:
    """Whether free-form outbound on this channel is time-limited at all."""
    name = channel or WHATSAPP
    if name in WINDOWLESS:
        return False
    pol = _POLICIES.get(name)
    return pol.windowed if pol is not None else True


def always_open(channel: str | None) -> bool:
    """
    A channel we can send on whenever we like: a provider, and no window.

    The OTA channel today. Distinct from `not has_window()`, which is also true
    of the staff console — where there is nothing to send to at all.
    """
    pol = _POLICIES.get(channel or WHATSAPP)
    return pol is not None and not pol.windowed


def for_channel(channel: str | None) -> ChannelPolicy:
    """
    The policy for a channel.

    Unknown channels raise rather than defaulting to WhatsApp. A channel we have
    not thought about must not silently inherit another channel's window rule
    and template behaviour — that is how a message goes out under rules Meta
    never agreed to.
    """
    policy = _POLICIES.get(channel or WHATSAPP)
    if policy is None:
        raise ValueError(f"no messaging policy for channel {channel!r}")
    return policy


def channels() -> tuple[str, ...]:
    return tuple(_POLICIES)

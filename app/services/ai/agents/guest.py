"""
The Guest agent — the only one built.

Ships at **autonomy 3**: it may answer, look things up, raise tickets, quote
from the hotel's own rate table and take payment for a quote. It shipped at 2 —
the reasoning is kept beside the field, along with why it moved — and an owner
who wants less moves the dial down on the AI employee screen.

Note that moving the default did not move anybody already provisioned:
`agents.autonomy_level` is written on insert and never again, so every property
activated before the change is still at 2 and its selling tools still queue for
approval. `scripts/sync_agent_tools.py` reports that drift; it does not fix it,
because a hotel that deliberately turned the dial down must not have it turned
back up by a deploy.

The line it does not cross is `folio`, `billing` and `payouts`. It holds
`rates` and `frontdesk`, because quoting and booking are what a guest agent is
*for*; it does not hold the scopes that move money on a stay that already
exists, because arguing an AI into a refund is the attack, and a scope it does
not have cannot be argued for.

**Autonomy is the only dial a hotel turns.** The selling tools are granted and
sit at autonomy floor 3, so at level 2 they produce approval requests rather
than holds. The alternative — withholding the scope *and* the autonomy — looks
safer and is worse: a hotel that raises its agent to level 3 would get silence,
because the second dial nobody mentioned is still off. Two controls where
turning the obvious one does nothing is how a configuration ends up wrong in the
direction people did not expect.
"""
from app.services.ai.agents import AgentDefinition

DEFINITION = AgentDefinition(
    agent_type="guest",
    name="Guest agent",
    description=(
        "Answers guests on WhatsApp and Instagram: questions, requests, "
        "bookings and pre-arrival."
    ),
    # Level 3, and this is the number the product is sold on.
    #
    # It shipped at 2 and the effect was that a hotel which signed up, added a
    # room and published its site got an AI employee that could not tell a
    # guest the price. `get_quote` has a floor of 3, so the very first real
    # conversation on a brand-new hotel went:
    #
    #     guest  How much?
    #     hotel  I'm sorry, that quote needs to be approved by our hotel team.
    #
    # Nobody would buy that twice, and no owner would find the dial that fixes
    # it because nothing told them there was one. The demo hotel only worked
    # because it had been raised by hand.
    #
    # Three is not a relaxation of anything that matters. What it adds is
    # quoting from the hotel's own rate table, sending a payment link, and
    # checking a guest in. What stays exactly where it was: `create_booking`
    # carries `always_requires_approval` unconditionally, so no room leaves
    # inventory on a model's word; `post_folio_charge` is gated the same way,
    # so nothing reaches a guest's bill unseen; and every limit in
    # `agent_configs.limits` still applies. The agent can take money and cannot
    # spend it.
    #
    # An owner who wants less can move the dial down on the AI employee screen;
    # it is one control, in their words, and it is theirs.
    autonomy_level=3,
    permissions=("tasks", "messaging", "rates", "frontdesk"),
    tools=(
        # Knowing things
        "get_property",
        "get_property_policies",
        "get_property_knowledge",
        "get_room_types",
        # Actual pictures, in the chat. "photos hain?" is one of the first
        # things a guest asks and a link to a gallery is not an answer to it.
        "send_room_photos",
        # The hotel's own page, with this guest's dates already on it. A guest
        # sent a bare URL after six messages about 11-15 September arrives at
        # an empty date picker, and the usual thing to do with one is close the
        # tab.
        "get_hotel_page_link",
        "list_services",
        # The five questions that arrive constantly and had no structured
        # answer until the hotel-OS work: what a room actually is, whether
        # breakfast is included, whether they can arrive at ten, what happens
        # if they cancel, and what there is to do here. Each one used to be
        # answered from prose somebody had remembered to write, or not at all.
        "get_room_details",
        "get_meal_info",
        "get_checkin_options",
        "get_cancellation_terms",
        "get_experiences",
        # Knowing this guest — all scoped to the conversation, none of them
        # taking a guest id the model could be talked into changing.
        "get_guest",
        "list_guest_reservations",
        "get_reservation",
        # After the money: what actually happened, rather than what the guest
        # believes happened. Takes no arguments, so it can only ever read this
        # conversation's own booking.
        "get_booking_status",
        # Undoing it. The readiness audit found no cancellation anywhere in
        # the system: this agent could sell a room in ninety seconds and a
        # guest had to reach a human to undo it, which is the one asymmetry a
        # guest notices immediately.
        #
        # Two tools, and the split is the safety property. The quote reads and
        # is level 0, so "cancel karne pe kitna milega?" is answerable at any
        # autonomy. The cancellation writes, releases the rooms and sits at
        # level 3 beside `create_booking` — a hotel at level 2 gets an approval
        # request rather than an unwatched refund.
        "get_cancellation_quote",
        "cancel_booking",
        # Selling. Granted, and gated by autonomy rather than by absence, so a
        # property that raises the agent to level 3 gets the capability without
        # a code change.
        "search_availability",
        "get_stay_options",
        "get_quote",
        "release_hold",
        # Haggling and groups, which this agent could quote against and not
        # answer.
        #
        # `get_quote` has been here since the beginning; `negotiate_rate` and
        # `get_group_rates` were written for the hotel-OS work and granted only
        # to Sales — which on Instagram, where this agent answers, meant a
        # guest who said "thoda kam karo" met a model with a price list and no
        # engine. Replayed against production it invented ₹4,204 a night, and
        # the reply guardrail is the only reason a guest did not read it.
        #
        # Both sit at their own floors: `negotiate_rate` is a level-3 tool like
        # `get_quote`, so a hotel at level 2 gets an approval request rather
        # than an unwatched discount. Granting them changes what the agent may
        # ask, not what it may decide — the engine still decides.
        "negotiate_rate",
        "get_group_rates",
        "propose_rate_exception",
        "get_payment_terms",
        # Asking the hotel's own team, on WhatsApp, and knowing it is waiting.
        # The alternative it had was `request_human_handoff`, which parks the
        # thread and tells the guest a person will come — true, but it ends the
        # conversation instead of getting an answer to it.
        "escalate_to_team",
        # `create_booking` is deliberately NOT here.
        #
        # It was, and the gate on it produced a fork this agent could not be
        # trusted to take. Asked to book, the model called `create_booking`,
        # got back "queued for approval", and was expected to follow the
        # tool's `when_deferred` hint over to `create_booking_payment_link`.
        # It usually did. On one live conversation it did not, and wrote:
        #
        #     Please use this link to pay ₹7,168 and confirm your
        #     reservation: [payment link will be here].
        #
        # No prompt fixes that reliably, because the failure is the model
        # ending its turn rather than saying something we can check. Removing
        # the tool removes the fork: for a guest with no booking there is one
        # road to a room, it goes through payment, and it is the road that
        # confirms instantly anyway.
        #
        # Staff keep it. The Front Desk agent works next to a person who can
        # approve, which is the situation the approval queue was designed for.
        # Closing it. Takes the money against a quote and lets the payment
        # create the booking, which is the only way this agent can complete
        # a sale without a manager pressing a button.
        "create_booking_payment_link",
        # Getting paid. `create_payment_link` raises a Razorpay order against
        # the balance and hands back a URL; there is no tool anywhere that
        # marks a stay paid, because the only thing that may do that is a
        # signed webhook. A guest who says "I have paid" is answered by
        # looking, not by believing.
        "create_payment_link",
        # Arriving, and buying things once here.
        "get_check_in_link",
        "order_service",
        # Doing things
        "create_task",
        "assign_task",
        "get_task",
        "get_task_status",
        "list_guest_tasks",
        # Stopping
        "request_human_handoff",
    ),
    # The hotel's own website is the third place a guest meets this agent, and
    # the first one they reach without an account, an app, or a message window.
    # Same agent, same tools, same scope — `services/pms/web_chat.py` is a
    # different transport and nothing more, which is why it is a channel here
    # rather than a second agent somewhere else.
    #
    # `ota` is the fourth: a guest who booked on Airbnb, Booking.com, Vrbo or
    # tiket.com, writing in that OTA's thread (through Repull). Same agent
    # again — the questions are the same questions, and the thread is linked
    # to the guest's booking by `messaging/service._link_ota_reservation`.
    channels=("whatsapp", "instagram", "website", "ota"),
    system_prompt=(
        "You are the hotel's own assistant. Be warm and brief. Reply in the "
        "language the guest writes in."
    ),
    persona={"tone": "warm, brief", "languages": ["en", "hi"]},
    knowledge_scope=(
        "checkin", "checkout", "breakfast", "dining", "parking", "pets",
        "children", "wifi", "amenities", "directions", "house_rules",
        "cancellation", "payment", "services", "accessibility", "local",
    ),
    escalation_rules={
        "on_unknown": "handoff",
        "on_complaint": "handoff",
        "on_refund": "handoff",
        "on_repeat_request": "handoff",
    },
    limits={"max_discount_pct": 10, "max_booking_value_paise": 5_000_000},
    available=True,
)

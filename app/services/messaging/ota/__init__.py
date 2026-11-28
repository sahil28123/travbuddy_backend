"""
OTA guest messages — Airbnb, Booking.com, Vrbo and tiket.com, through Repull.

    adapter.py   sending, through the Repull SDK, into a Repull conversation
    fake.py      a deterministic stand-in implementing the same protocol
    webhook.py   `reservation.message.received` → the shared inbound path

One channel (`ota`) for every OTA. The guest's reply goes back through the
same Repull conversation whichever OTA they booked on, and the rules that do
differ per OTA — Airbnb stripping a phone number, Vrbo queueing a busy thread —
are Repull's to apply. What differs from the Meta channels is expressed as
code in `messaging/policy.py`, not as comments:

  · **no window** — an OTA lets a host answer whenever they like, so the
    thread is always open (`windowed=False`), and the agent keeps its sending
    tools
  · **no templates** — and none needed; `send_template` refuses in both the
    real adapter and the fake
  · **no conversation billing** — nobody bills per OTA conversation; the
    agent's turns are metered as they are on every channel
"""

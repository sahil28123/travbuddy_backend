# Selling on Airbnb, Booking.com, Vrbo and more with Repull

HostOS takes bookings from the hotel's own website, WhatsApp, Instagram and the
front desk. With the Repull integration switched on, a hotel can also sell its
rooms on the big booking channels:

- **Rates and availability** you set in HostOS go out to every connected channel.
- **Channel bookings** (new, changed, cancelled) arrive in HostOS as reservations,
  and the room is taken off sale everywhere else.
- **Guest messages** from Airbnb, Booking.com, Vrbo and tiket.com reach the same
  AI front desk and guest agents that answer WhatsApp and Instagram, and their
  replies go back to the guest on the channel they wrote from.
- **A hotel moving from another PMS** can bring its room types and upcoming
  bookings into HostOS in one step.

The integration is optional and off by default. Nothing changes for a deployment
that does not set `ENABLE_REPULL=true`.

---

## What Repull is

[Repull](https://repull.dev) is one API for the booking channels and the
property management systems around them:

| Kind | Connected through Repull |
| --- | --- |
| **Booking channels (OTAs)** | Airbnb, Booking.com, Vrbo, tiket.com. Expedia is coming soon. |
| **PMSs** | Mews, Cloudbeds, Guesty, Hostaway, Hospitable, OwnerRez, Lodgify, Smoobu, Beds24, iGMS, BookingSync, Track, Zeevou, RentalReady, CiiRUS |

More channels and PMSs are added regularly. HostOS talks to all of them through
one API key, one webhook and the official
[`repull-sdk`](https://pypi.org/project/repull-sdk/) Python package. When Repull
adds a channel, HostOS hotels can use it without any change to this code.

---

## How it works inside HostOS

```text
                         ┌──────────────── Repull ────────────────┐
  HostOS calendar        │                                        │   Airbnb
  (rates, rooms,   ──────┼─► PUT /v1/availability/{listing} ──────┼─► Booking.com
   stop-sells)    push   │                                        │   Vrbo
                         │                                        │   tiket.com
  HostOS reservations ◄──┼── webhook: reservation.created ◄───────┼── a guest books
  (room taken off sale   │           reservation.updated          │
   on every channel)     │           reservation.cancelled        │
                         │                                        │
  HostOS AI agents   ◄───┼── webhook: reservation.message.received┼── a guest writes
  (front desk, guest) ───┼─► POST /v1/conversations/{id}/messages ┼─► the reply
                         │                                        │
  HostOS room types  ◄───┼── GET /v1/listings, /v1/reservations ◄─┼── Mews, Cloudbeds,
  + bookings (import)    │                                        │   Guesty, …
                         └────────────────────────────────────────┘
```

**Each hotel is its own Repull workspace.** One HostOS deployment serves many
hotels with one API key. The first time a hotel connects, Repull creates a
sub-workspace for it. Every later call for that hotel is scoped to it, so one
hotel's channel accounts are never visible to another.

**Room types map to Repull listings.** When a hotel connects Booking.com or
tiket.com, Repull imports each room type with its real number of rooms. Airbnb
and Vrbo listings arrive the same way. HostOS matches its own room types to
those listings automatically where the names agree; you map the rest by hand.

**One rate plan per room type is sold on the channels.** An OTA listing has one
nightly price. HostOS uses the room type's public rate plan unless you choose
another when mapping.

Code map:

| File | What it does |
| --- | --- |
| `app/services/repull/client.py` | Builds every Repull request through `repull-sdk`; turns Repull errors into readable ones |
| `app/services/repull/connections.py` | Hotel ⇄ workspace, room type ⇄ listing, the Connect link |
| `app/services/repull/ari_push.py`, `app/workers/repull_tasks.py` | Pushes the calendar out (on every change, plus a nightly full resync) |
| `app/services/repull/reservations.py` | Turns channel bookings into HostOS reservations |
| `app/services/repull/pms_import.py` | Imports room types and bookings from another PMS |
| `app/services/messaging/ota/` | The `ota` messaging channel the AI agents answer on |
| `app/services/repull/signature.py`, `events.py` | Verifies and de-duplicates webhooks |
| `app/api/v1/pms/repull.py` | The HTTP routes below |
| `migrations/2026-11-28_a_hotel_can_sell_on_the_channels.sql` | The tables |

---

## 1. Get a Repull API key

1. **Register** at **[repull.dev/register](https://repull.dev/register)**.
2. **Create an API key** at **[repull.dev/dashboard/keys](https://repull.dev/dashboard/keys)**.
   There is one key type. It starts with `sk_live_` and acts on whatever
   channels your workspace has connected.
3. Plans and pricing are at [repull.dev/pricing](https://repull.dev/pricing).

Keep the key on the server only. It goes in `REPULL_API_KEY` and is never sent
to the partner app or a browser. Authentication details:
[repull.dev/docs/api-keys](https://repull.dev/docs/api-keys).

---

## 2. Configure HostOS

Install the dependencies. `repull-sdk` is in `requirements.txt`.

```bash
pip install -r requirements.txt
```

Apply the migration to the HostOS Supabase project:

```bash
psql "$HOSTOS_DATABASE_URL" -f migrations/2026-11-28_a_hotel_can_sell_on_the_channels.sql
```

Set the environment variables (in `.env.hotelos` for a StayOS deployment):

| Variable | Description |
| --- | --- |
| `ENABLE_REPULL` | `true` mounts the Repull routes. Needs `ENABLE_HOTEL_OS=true`. Default `false`. |
| `REPULL_API_KEY` | Your `sk_live_…` key from [repull.dev/dashboard/keys](https://repull.dev/dashboard/keys) |
| `REPULL_WEBHOOK_SECRET` | The `whsec_…` secret from step 3 below |
| `REPULL_CONNECT_REDIRECT_URL` | Where a hotelier lands after connecting their channels, usually a page of the partner app |
| `REPULL_BASE_URL` | Optional. Default `https://api.repull.dev` |
| `REPULL_RESYNC_DAYS` | Optional. How far ahead a full calendar push goes. Default `365` |

Run a Celery worker on the `hotelos` queue (calendar pushes run there) and the
StayOS beat (it runs the nightly full resync), as for the rest of HostOS.

---

## 3. Register the webhook

Repull tells HostOS about bookings, guest messages and connection changes with
signed webhooks. Register your endpoint once, with your API key:

```bash
curl -X POST https://api.repull.dev/v1/webhooks \
  -H "Authorization: Bearer $REPULL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://<your-hostos-host>/api/v1/pms/public/repull/webhook",
    "description": "HostOS",
    "events": [
      "reservation.created",
      "reservation.updated",
      "reservation.cancelled",
      "reservation.message.received",
      "connect.session.completed",
      "account.created",
      "account.disconnected"
    ]
  }'
```

The response contains `secret` (`whsec_…`) **once**. Put it in
`REPULL_WEBHOOK_SECRET`. HostOS rejects every delivery whose signature does not
match (see
[repull.dev/docs/webhooks/verify-signatures](https://repull.dev/docs/webhooks/verify-signatures)).

To check the endpoint, send a sample event:

```bash
curl -X POST https://api.repull.dev/v1/webhooks/<webhook-id>/test/reservation.created \
  -H "Authorization: Bearer $REPULL_API_KEY"
```

---

## 4. Connect a hotel

All routes take the property id in the path and use the partner app's normal
login. Connecting needs the owner role; mapping and resyncing need `rates`.

**Get a Connect link** and open it for the hotelier:

```http
POST /api/v1/pms/properties/{property_id}/repull/connect
{ "purpose": "connect" }
```

```json
{ "url": "https://connect.repull.dev/…", "workspaceId": "…", "providers": ["airbnb", "booking", "vrbo-login", "tiket"] }
```

On Repull's hosted page the hotelier picks a channel and signs in. Airbnb is a
one-click sign-in. For Booking.com and tiket.com they choose Repull as their
connectivity provider in the extranet and enter their hotel ID. They then land
on `REPULL_CONNECT_REDIRECT_URL`.

**Map room types to listings.** Try the automatic match first:

```http
POST /api/v1/pms/properties/{property_id}/repull/automap
```

```json
{ "mapped": [ … ], "unmatched": [ { "roomTypeId": "…", "name": "Deluxe Double", "candidates": 0 } ] }
```

For anything unmatched, list the listings and map by hand:

```http
GET /api/v1/pms/properties/{property_id}/repull/listings

PUT /api/v1/pms/properties/{property_id}/repull/map
{ "mappings": [ { "roomTypeId": "…", "listingId": 12345, "ratePlanId": "…" } ] }
```

`ratePlanId` is optional; the public plan is used when it is left out.
`"listingId": null` removes a mapping. Mapping a room type pushes its calendar
straight away.

**Check the connection** at any time:

```http
GET /api/v1/pms/properties/{property_id}/repull/status
```

The response lists each connected channel with its status. If a channel needs
the hotelier to sign in again, it includes a `fixUrl` to send them to.

---

## 5. What syncs, and when

**Calendar: HostOS → channels.** Pushed after each of these:

- an edit in the rate calendar (prices, rooms for sale, stop-sell, minimum and
  maximum stay, closed to arrival or departure);
- a booking or a cancellation that moves rooms (a quote hold counts as taken
  while it is active; when it expires, the next push or the nightly resync puts
  the room back on sale);
- a room type being mapped;
- every night, a full resync of the next `REPULL_RESYNC_DAYS` days as a safety
  net. You can also run it by hand:

```http
POST /api/v1/pms/properties/{property_id}/repull/resync
{ "days": 365 }
```

Editing a rate plan that is not the one sold on the channels (a B2B rate, say)
sends nothing.

For each night HostOS sends: rooms left to sell (rooms for sale minus booked,
held and blocked; capped at the listing's room count on the channel; rooms the
hotel only agreed to oversell are never offered to the OTAs), whether the
night is open, the price from the mapped rate plan, minimum and maximum stay,
and closed-to-arrival or departure.
Consecutive nights with identical values go in one request. Pushes run in the
background, so a calendar edit or a booking never waits on a channel. They are
retried automatically if Repull or a channel is briefly unavailable.

**Bookings: channels → HostOS.**

- **New booking:** a confirmed HostOS reservation is created with the channel as
  its `source` and the OTA confirmation code in `external_ref`. The guest is
  matched to an existing guest record by phone or email. The room comes off sale
  on every other channel.
- **Booking requests** (an Airbnb request-to-book, for example) are stored as
  pending and take no room until the channel confirms them, which is HostOS's
  normal rule for pending reservations.
- **Changed dates or room:** the old nights are released and the new ones booked.
- **Cancelled:** the reservation is cancelled through HostOS's normal
  cancellation path and the room goes back on sale.
- **No room left in HostOS** (an overbooking): the booking is still recorded,
  because the guest holds a confirmed reservation on the channel, and an
  escalation is raised for staff.
- The channel's total is stored as-is; HostOS does not re-price it. INR totals
  fill the reservation's amounts. Other currencies are kept in
  `external_payload` with a note.

Webhooks are de-duplicated on Repull's event id. If HostOS fails to handle one,
it answers 500 and Repull retries.

**Guest messages: channels ⇄ HostOS AI.** Each OTA conversation becomes a
HostOS conversation on the `ota` channel and is linked to its reservation. The
guest's message goes through the same pipeline as WhatsApp, so the hotel's AI
agents and staff answer it in the same inbox. Replies are sent back through
Repull to the channel the guest wrote on. OTAs have no 24-hour window and no
message templates, so neither applies on this channel.

---

## 6. Importing a hotel from another PMS

For a hotel arriving from Mews, Cloudbeds, Guesty or any PMS above:

```http
POST /api/v1/pms/properties/{property_id}/repull/connect
{ "purpose": "import" }
```

The hotelier connects their PMS on the hosted page. Then run:

```http
POST /api/v1/pms/properties/{property_id}/repull/import
```

```json
{ "roomTypesCreated": 4, "roomTypesMatched": 1, "reservationsImported": 212, "reservationsSkipped": 0 }
```

HostOS creates a room type with the right number of rooms and a default public
rate plan for each PMS room type it does not already have. It then imports
every booking that has not yet checked out. The import is safe to run again:
anything already imported is left as it is.

---

## Limits and troubleshooting

- **One rate plan per room type goes to the channels.** Rate-plan-level pricing
  on Booking.com and tiket.com is a possible later addition.
- **Rooms owned by a PMS.** If a room type's rooms are managed in a PMS that is
  still connected (Mews, Cloudbeds), Repull refuses room-count changes from
  HostOS. HostOS then pushes prices and restrictions only.
- **A channel says "failed" after a push.** Each push logs the per-channel result
  Repull returns. A channel that needs the hotelier to reconnect shows a
  `fixUrl` in `/repull/status`.
- **Webhooks answered 401.** `REPULL_WEBHOOK_SECRET` does not match the webhook,
  or the server clock is more than five minutes off.
- **Webhook ignored as "unknown workspace".** The event is for a hotel this
  deployment has not connected, for example another deployment using the same
  key.
- Repull errors carry a `fix` sentence and a `docs_url`. HostOS passes both
  through in its logs and API errors.

Repull documentation:

- Quickstart: https://repull.dev/docs/quickstart
- Connect: https://repull.dev/docs/connect
- Hotels and multi-unit room types: https://repull.dev/docs/hotels
- Webhooks: https://repull.dev/docs/webhooks/event-types and https://repull.dev/docs/webhooks/verify-signatures
- Conversations: https://repull.dev/docs/conversations
- Rate limits: https://repull.dev/docs/rate-limits

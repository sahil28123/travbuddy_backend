from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Optional


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Vertex AI — Gemini 2.5 Flash via Vertex (uses ADC / service account)
    gcp_project: Optional[str] = None
    gcp_location: str = "us-central1"
    # Fallback regions tried (in order) when the primary region returns 429/503.
    # Each Vertex region has its own quota, so rotating spreads load and lets a
    # rate-limited request retry immediately against a fresh region instead of
    # sleeping through a long backoff. Comma-separated; primary is always first.
    # `global` and `asia-south1` are both here for reasons worth stating.
    #
    # **`global` was missing and it cost the Gemini 3 upgrade.** Almost the whole
    # 3.x family is served from `global` and from nowhere else, so a universe
    # that did not contain it could not reach any of them however the model ids
    # were spelled.
    #
    # **`asia-south1` is where this application runs.** Mumbai serves
    # `gemini-2.5-flash` at ~0.7s against us-central1's ~2.2s, and
    # `gemini-3.5-flash` at ~1.8s against global's ~2.4s. It was absent from a
    # list of eleven regions on three continents.
    #
    # This is the universe, not the itinerary: which of these can serve a given
    # model is `app/services/model_catalog.py`, which was measured per model.
    # Several models 404 in several of these, and the catalogue is what keeps a
    # call from being sent somewhere it cannot be answered.
    gcp_locations: str = (
        "global,asia-south1,asia-southeast1,"
        "us-central1,us-east1,us-east4,us-east5,us-west1,us-west4,"
        "us-south1,northamerica-northeast1,europe-west1,europe-west4"
    )
    # Path to a service account JSON key; leave unset to use ADC
    google_application_credentials: Optional[str] = None

    # LLM gateway tiers — model IDs live in config ONLY (never hardcode at call
    # sites). "brain" handles reasoning-heavy calls (itinerary, journey tradeoffs,
    # re-ranking); "muscle" handles high-volume schema-bound transforms
    # (extraction, tagging, onboarding). Swap to gemini-3.5-flash /
    # gemini-3.1-flash-lite via env once regional availability is confirmed.
    vertex_model_brain: str = "gemini-2.5-flash"
    vertex_model_muscle: str = "gemini-2.5-flash-lite"

    # Embeddings for the recommendation engine (docs/recommendation_engine.md).
    # 384-dim is deliberate: ~4k hotels, not 100M docs — half the RAM/index, no
    # measurable recall loss at this scale.
    vertex_model_embedding: str = "text-embedding-005"
    embedding_dim: int = 384

    # Vertex fallback account — tried only when the primary project returns 429
    # (RESOURCE_EXHAUSTED). Lets a rate-limited Gemini call retry instantly against
    # a second project's independent quota. Both optional; omit to disable fallback.
    vertex_fallback_project: Optional[str] = None
    vertex_fallback_service_account_json: Optional[str] = None  # full SA JSON string

    # ── Hotel OS ─────────────────────────────────────────────────────────────
    #
    # Whether this process serves the hotel OS at all — `/api/v1/pms/*` and the
    # messaging webhooks. **Off by default, and the default is the whole point.**
    #
    # The hotel OS is a different Supabase project to the marketplace
    # (`rdlozqvuywtgsodunait` against `uabxtdhxbamuhthxnhid`) and `get_db()`
    # builds one client from one `SUPABASE_URL`. A marketplace deployment with
    # these routes mounted answers 500 to every one of them, because the tables
    # are not there — so StayOS is its own deployment of this codebase with
    # `.env.hotelos`, and the marketplace stays correct without anybody having
    # to remember to switch something off.
    enable_hotel_os: bool = False

    # ── Hotel OS agent runtime (services/ai) ────────────────────────────────
    # Gemini on Vertex, through the same `services/vertex.py` the traveller-side
    # pipeline uses — same project, same credentials, same region rotation, same
    # circuit breaker. One AI provider architecture, not two.
    #
    # Model ids live here and nowhere else, matching the rule above, and the
    # runtime asks `model_router` for a *capability* rather than naming any of
    # these. Three tiers because guest traffic is not one workload:
    #
    #   conversation  the workhorse. A guest turn is a short reasoning step over
    #                 twenty tool schemas, run thousands of times a month.
    #   reasoning     the exception. A rate negotiation, a cancellation dispute,
    #                 an owner asking why revenue moved — where a wrong answer
    #                 costs more than the tokens.
    #   classify      routing, intent, language, spam. Never worth the others.
    #
    # **The 3.x family is reachable, and the note that said otherwise was wrong
    # in a way worth recording.** It read: "gemini-3-flash, gemini-3.1-pro and
    # gemini-3.1-flash-lite all return 404 … the models are not enabled for this
    # project." Two mistakes compounded into that conclusion. `gemini-3-flash`
    # and `gemini-3.1-pro` are not model ids — the real ones are
    # `gemini-3-flash-preview` and `gemini-3.1-pro-preview`. And every 3.x model
    # is served from `global`, whose endpoint has no region prefix, which the
    # URL builder could not construct. Wrong id and unreachable host both return
    # 404, which reads exactly like no access. Every model below was verified by
    # live `generateContent`, including function calling and responseSchema.
    #
    # Four tiers now, because "conversation" was doing two jobs. A guest asking
    # for the wifi password and a guest disputing a cancellation fee were both
    # the workhorse; splitting them lets the cheap model take the volume while
    # the brain keeps the work that needs it.
    #
    #   classify      the router: intent, language, spam. Never reasons.
    #   fast          FAQs, lookups, simple CRUD. High volume, thinking off.
    #   conversation  the agent brain: tool selection, operations, money.
    #   deep          escalation only, when the brain has already failed.
    #
    # Ids only. Locations, thinking budgets and fallbacks are in
    # `app/services/model_catalog.py`, because they are properties of the model
    # rather than choices this deployment makes.
    hotel_model_classify: str = "gemini-3.5-flash-lite"
    hotel_model_fast: str = "gemini-3.5-flash"
    hotel_model_conversation: str = "gemini-3.6-flash"
    hotel_model_reasoning: str = "gemini-3.6-flash"
    hotel_model_deep: str = "gemini-3.1-pro-preview"

    # **The rollback, and it is one variable.**
    #
    # False puts every tier back on the 2.5 models that have been serving
    # production, without a deploy and without touching the router. That is what
    # makes this not a big-bang migration: 3.x is the default because it is
    # better, and it is one environment variable away from not being.
    #
    # The 2.5 ids are kept as their own settings rather than hardcoded in the
    # rollback path, so the floor stays configurable too.
    hotel_models_v3: bool = True

    # Whether PATCH /website refuses a field it does not recognise.
    #
    # False, for now, and deliberately. The builder that calls it lives in
    # another repository, so nothing here can prove no client has been sending a
    # stray key for months — and a 422 to one of them turns a cosmetic bug into
    # a hotelier who cannot edit their own page. While it is false the endpoint
    # accepts the request, saves only the fields it knows, logs the names it did
    # not, and returns them as `ignoredFields`. When that log has been quiet
    # long enough, this flips and the same information becomes a 422.
    website_strict_fields: bool = False
    hotel_model_classify_stable: str = "gemini-2.5-flash-lite"
    hotel_model_fast_stable: str = "gemini-2.5-flash"
    hotel_model_conversation_stable: str = "gemini-2.5-flash"
    # 2.5-pro, so a rollback restores the exact configuration that was in
    # production before any of this — not an approximation of it.
    hotel_model_reasoning_stable: str = "gemini-2.5-pro"
    hotel_model_deep_stable: str = "gemini-2.5-pro"

    # Per request, not per turn. A guest waiting on WhatsApp will forgive thirty
    # seconds; a hung socket holding a worker is a different problem.
    #
    # **The fallback, not the number.** `model_router` sets a real timeout per
    # tier — a classification is not a negotiation — and this is what a tier
    # nobody has listed falls back to. Left at 45 deliberately: a value that is
    # only reached by a mistake should be forgiving.
    hotel_model_timeout_seconds: float = 45.0
    # How long a whole turn may take, tool rounds included. The clock that did
    # not exist: the timeout above is per HTTP request and a turn makes up to
    # `agent_max_tool_turns` of them, so nothing bounded what a guest actually
    # waits. One real turn took 107 seconds. Per tier in `model_router`; this is
    # the fallback. See `runtime.run_turn`, which stops asking for another tool
    # round once it is spent rather than cutting the guest off.
    hotel_turn_budget_seconds: float = 60.0
    # How many times the model may call tools before the runtime stops it. Six
    # is generous for "look up the booking, raise a ticket, assign it"; a loop
    # that needs more is a loop that has stopped making progress.
    agent_max_tool_turns: int = 6

    # ── WhatsApp Cloud API ──────────────────────────────────────────────────
    # Credentials live here and nowhere else. `whatsapp_connections` holds a
    # `credentials_ref` naming which entry to use, because a token in a database
    # row is a token in every backup, replica and accidental `select *`.
    #
    # `app_secret` verifies Meta's X-Hub-Signature-256 on every inbound webhook;
    # `access_token` sends. They are separate secrets doing opposite jobs and
    # are never interchangeable.
    whatsapp_app_secret: Optional[str] = None
    whatsapp_access_token: Optional[str] = None
    # The string Meta echoes during the GET handshake. Ours to choose, and
    # compared in constant time — see services/messaging/whatsapp/webhook.py.
    whatsapp_verify_token: Optional[str] = None
    whatsapp_api_base: str = "https://graph.facebook.com/v21.0"
    whatsapp_timeout_seconds: float = 15.0

    # Connections whose `credentials_ref` is this one are wired to the
    # deterministic fake rather than to Meta. Named rather than inferred from a
    # hostname, so nothing can ship believing it is talking to Meta when it is
    # not.
    whatsapp_sandbox_ref: str = "sandbox"

    # Meta permits free-form replies only inside a service window that opens on
    # each inbound message. 24 hours is Meta's number, not ours; it is here so a
    # test can shorten it rather than sleep through it.
    whatsapp_service_window_hours: int = 24

    # ── Embedded Signup (Tech Provider) ─────────────────────────────────────
    #
    # What turns "we have a WhatsApp number" into "a hotel can connect its own".
    # A hotel completes Meta's Embedded Signup, we exchange the code it returns
    # for a business token scoped to that customer, and from then on the hotel
    # has its own WABA, its own number and its own templates.
    #
    # `whatsapp_app_id` is the **Facebook** app id from Basic settings — the
    # same app the webhook signature already uses, and NOT the Instagram app id
    # a few lines below. They are both ~16 digits and only one of them works
    # here; `scripts/meta_config_check.py` exists because of that trap.
    whatsapp_app_id: Optional[str] = None

    # The Embedded Signup configuration created under Facebook Login for
    # Business. Required, not optional: without it the popup falls back to a
    # generic login dialogue that asks a hotelier for permissions we do not
    # need and never creates a WABA.
    whatsapp_config_id: Optional[str] = None

    # Which Embedded Signup variant the popup should run, and it must match the
    # configuration named above.
    #
    #   ""                              a business getting a new number
    #   "whatsapp_business_app_onboarding"
    #                                   a business already using the WhatsApp
    #                                   Business App, migrating that number
    #
    # Server-driven rather than hardcoded in the PWA: the answer is a property
    # of the Meta configuration, and a popup asking for one flow against a
    # config built for the other fails inside Meta's own UI, where we cannot see
    # it. Read it out of the Meta-hosted link's `extras.featureType`.
    whatsapp_signup_feature_type: str = ""

    # Ours, from the App Dashboard. Rides along on the signup so Meta attributes
    # the WABA to us as its Tech Provider, and comes back on the PARTNER_ADDED
    # webhook.
    whatsapp_solution_id: Optional[str] = None

    # The Meta-hosted landing page, for onboarding a hotel by sending them a
    # link rather than sitting them in front of the PWA. Optional — the in-app
    # popup is the primary path and the only one that returns a code.
    whatsapp_meta_hosted_url: Optional[str] = None

    # Off by default, like `enable_hotel_os`. A deployment with the flag unset
    # shows WhatsApp as not-yet-available rather than offering a Connect button
    # that leads to a 503 — which is what a half-configured Embedded Signup
    # looks like to a hotelier.
    whatsapp_onboarding_enabled: bool = False

    # ── Repull (Airbnb, Booking.com, Vrbo, tiket.com and PMS connectivity) ────
    #
    # Repull is the channel manager behind "sell this hotel on the OTAs": one
    # API key, one webhook, and every connected channel. See docs/repull.md.
    #
    # Off by default, like `enable_hotel_os`, and only meaningful inside it — the
    # Repull routes mount in the hotel-OS block of `api/v1/router.py`, because
    # every one of them reads `properties`, `room_types` and `reservations`.
    #
    # `repull_api_key` is the operator's `sk_live_…` key from
    # https://repull.dev/dashboard/keys. Each hotel gets its own Repull
    # sub-workspace under that key, so one HostOS deployment serves many hotels
    # without their channel accounts ever seeing each other.
    #
    # `repull_webhook_secret` is the `whsec_…` value Repull returns once, when
    # the webhook is registered. Without it the webhook refuses everything —
    # an unsigned reservation is not one this system will book a room for.
    enable_repull: bool = False
    repull_api_key: Optional[str] = None
    repull_base_url: str = "https://api.repull.dev"
    repull_webhook_secret: Optional[str] = None
    # Where Repull's hosted Connect page sends the hotelier back to after they
    # pick their channels. Usually a page of the partner app.
    repull_connect_redirect_url: Optional[str] = None
    # How far ahead the calendar is pushed on a full resync. A year is what the
    # OTAs actually sell; the nightly beat keeps the far edge from going stale.
    repull_resync_days: int = 365

    # ── Instagram ────────────────────────────────────────────────────────────
    #
    # Two windows, because Meta gives two. An automated reply gets 24 hours from
    # the guest's last message, the same as WhatsApp. A conversation a *person*
    # has taken over gets seven days under Meta's human-agent allowance — which
    # is why `services/messaging/policy.py` reads the conversation's `handler`
    # rather than treating this as a constant. Claiming the longer window while
    # the agent is still driving would be a policy violation wearing a longer
    # window's clothes.
    instagram_service_window_hours: int = 24
    instagram_human_agent_window_hours: int = 168

    instagram_sandbox_ref: str = "sandbox"
    instagram_verify_token: Optional[str] = None
    instagram_access_token: Optional[str] = None
    instagram_app_secret: Optional[str] = None
    instagram_api_version: str = "v21.0"

    # ── Instagram Business Login (OAuth) ────────────────────────────────────
    #
    # Meta's "Instagram API with Instagram Login" flow, which is the one that
    # lets a hotel connect its *own* account. Note these are the Instagram
    # app's credentials (Meta app → Instagram → API setup with Instagram
    # login), not the Facebook app id/secret on the Basic settings page. They
    # look interchangeable and are not: the token exchange rejects the wrong
    # pair with an "Invalid platform app" that says nothing useful.
    instagram_app_id: Optional[str] = None
    instagram_oauth_client_secret: Optional[str] = None

    # Must byte-for-byte equal the "Redirect URL" configured in Meta. Meta
    # compares the string, not the URL: a trailing slash, http vs https, or a
    # different case in the host is a rejected callback. Set explicitly in
    # production rather than derived from the request, because deriving it from
    # a proxied Host header is how an open redirect gets built by accident.
    #
    # ⚠ Moved with the rename to `api.hostos.in`. Meta will not accept the new
    # string until it is added to the app's Redirect URL list in the Meta
    # console — the default changing here does not change what Meta compares
    # against, and a mismatch fails the callback, not the consent screen, so it
    # looks like the hotel refused.
    instagram_redirect_uri: str = (
        "https://api.hostos.in/api/v1/messaging/instagram/oauth/callback"
    )

    # ── Meta ads ─────────────────────────────────────────────────────────────
    #
    # **Declared because they were unreachable, not because they are set.**
    #
    # `marketing/meta_ads._missing_credentials` reads these two with
    # `getattr(settings, ..., None)` and reports `ad_account` / `ads_permission`
    # when they are absent — which is right, and was the *only* possible
    # outcome: neither field existed on Settings, and `model_config` above is
    # `extra="ignore"`. So a hotelier who obtained an ad account and set
    # META_AD_ACCOUNT_ID and META_ADS_ACCESS_TOKEN in the environment had them
    # silently dropped by pydantic, and the product went on saying "connect a
    # Meta ad account" at somebody who just had.
    #
    # Declaring them does not connect anything and does not weaken the check
    # constraint that keeps a campaign out of `published` without a Meta id.
    # It turns "cannot be configured" into "configured or not", which is the
    # difference between a gap and a trap.
    #
    # Still required externally, and none of it is a code change: a Meta
    # Business account, a funded ad account (`act_…`), `ads_management` granted
    # to the app through App Review, and a Facebook Page linked to the
    # Instagram professional account. The Instagram credentials this system
    # already holds are messaging scopes and do not carry any of it — see
    # services/marketing/meta_ads.py.
    meta_ad_account_id: Optional[str] = None
    meta_ads_access_token: Optional[str] = None
    # The Page an ad's destination resolves to when the campaign is not
    # click-to-WhatsApp. `build_request` already takes it per call; this is the
    # property-independent default for a single-tenant deployment.
    meta_page_id: Optional[str] = None

    # Minimum for an AI employee that reads and answers DMs. Anything more is a
    # permission the hotel is asked to grant and we never use, which is both a
    # worse consent screen and a longer app review.
    instagram_oauth_scopes: str = "instagram_business_basic,instagram_business_manage_messages"

    # Where the owner lands when the connection succeeds or fails. The callback
    # appends ?instagram=connected or ?instagram=error&reason=…
    pms_app_url: str = "https://app.hostos.in"

    # How long a signed OAuth `state` stays valid. It is a CSRF token and a
    # round trip through Meta's consent screen, so minutes, not hours.
    oauth_state_ttl_seconds: int = 600

    # Fernet key for messaging_credentials.access_token_ciphertext. Unset means
    # per-hotel tokens cannot be read or written — deliberately fatal at the
    # boundary rather than silently falling back to storing them in the clear.
    messaging_credentials_key: Optional[str] = None

    # ── Google Business Profile ─────────────────────────────────────────────
    #
    # **Off, and off is the launch decision rather than an accident.** StayOS
    # launches on two channels a hotel can switch on by itself today — its own
    # website and Instagram. Google is written, tested and dormant: it needs a
    # Cloud OAuth client and an approved Google access request before reviews
    # or posts work at all, and a Connect button that leads to a wall is worse
    # for a new hotel than a channel that is simply not there yet.
    #
    # The flag gates the routes (api/v1/router.py) and the Channels row
    # (api/v1/messaging/oauth.py) from one place, so switching it on is the
    # whole of turning the feature on. Nothing was deleted; nothing is shown.
    enable_google_business_profile: bool = False

    #
    # A standard Google Cloud OAuth **web application** client — not the service
    # account in GOOGLE_SERVICE_ACCOUNT_JSON. The two are not interchangeable:
    # a Business Profile is owned by a person, and only that person's consent
    # can let us answer a review under the hotel's name. A service account has
    # no listings and never will.
    google_oauth_client_id: Optional[str] = None
    google_oauth_client_secret: Optional[str] = None

    # Byte-for-byte equal to an "Authorised redirect URI" on the OAuth client,
    # for the same reason the Instagram one is: Google compares the string.
    # ⚠ Same as the Instagram one: add it to the OAuth client's "Authorised
    # redirect URIs" in the Google console before this deploys, or the callback
    # is rejected with `redirect_uri_mismatch`.
    google_oauth_redirect_uri: str = (
        "https://api.hostos.in/api/v1/google/oauth/callback"
    )

    # One scope, and it is the only one Business Profile publishes. There is no
    # narrower read-only variant to prefer — `business.manage` is least
    # privilege here because it is the whole list.
    google_oauth_scopes: str = "https://www.googleapis.com/auth/business.manage"

    # Google's access tokens last an hour. Refresh a little early so a call that
    # takes a moment to reach Google is not holding a token that expires in
    # flight.
    google_token_refresh_skew_seconds: int = 300

    # The controlled smoke test, and only the controlled smoke test.
    #
    # Deliberately singular. `scripts/meta_smoke.py` messages exactly this
    # number and has no way to be handed a list — a smoke test that could
    # quietly widen into production traffic would be the worst possible way to
    # discover a mistake. Read through settings like every other credential, so
    # `.env` is genuinely the one place any of this lives.
    whatsapp_test_recipient: Optional[str] = None
    whatsapp_test_phone_number_id: Optional[str] = None

    # The hotel-facing number this deployment actually sends from, and the WABA
    # that owns it. Distinct from `whatsapp_test_phone_number_id` above, which
    # is Meta's throwaway test sender.
    #
    # **The WABA id cannot be discovered from the token** — a system-user token
    # returns nothing from `me/businesses` or
    # `assigned_whatsapp_business_accounts`. It is on the App Dashboard under
    # WhatsApp → API Setup, and it has to be set by hand or passed with --waba.
    whatsapp_phone_number_id: Optional[str] = None
    whatsapp_waba_id: Optional[str] = None

    # The template the in-app "send a test message" button uses.
    #
    # `hello_world` is Meta's own, present on every WABA from the day it is
    # created, and is what their "Try it out" tool sends. A *template* rather
    # than free text because a hotel testing a new connection has no 24-hour
    # window open — nobody has written to it — and free text there is refused
    # with 131047, which reads as a broken integration.
    whatsapp_test_template: str = "hello_world"
    whatsapp_test_template_language: str = "en_US"

    # Run the agent inline in the webhook request instead of handing it to
    # Celery. Development and the E2E suite only: a real deployment must
    # acknowledge Meta quickly and do the model call elsewhere.
    messaging_inline_processing: bool = False

    # Google Maps — Places API for place enrichment (optional but recommended)
    google_maps_api_key: Optional[str] = None

    # Pipeline tuning
    scene_detection_threshold: float = 27.0
    scene_min_length_frames: int = 15
    max_unique_frames: int = 10
    min_vision_confidence: float = 0.60
    frame_max_dimension: int = 1024
    frame_jpeg_quality: int = 85

    # Limits
    max_video_duration_seconds: int = 300
    max_video_size_mb: int = 200
    max_locations: int = 20

    # Cache
    places_cache_dir: str = "/tmp/travbuddy/places_cache"
    places_cache_ttl_seconds: int = 604800  # 7 days

    # Job queue — and the Celery broker, and the rate limiter's storage, and
    # the messaging pipeline's lock. One URL behind all of them.
    #
    # **The localhost default is a development convenience and is always wrong
    # in a container.** Nothing listens on localhost inside one, so it produces
    # `Error 111 connecting to localhost:6379` — which reads like Redis is down
    # rather than like the variable was never set. Worse than the noise is the
    # silence: this is also `broker=` in `workers/celery_app.py`, so a worker
    # that starts on the default connects to nothing and every messaging and
    # marketing task simply never runs, with no error anywhere.
    #
    # `app/main.py` therefore logs an error at boot if production is still
    # holding this default. See `redis_target()` for the redacted form used in
    # logs and health output.
    # The two deployments share a Redis server and must not share a namespace:
    # the marketplace is on database 0, StayOS on 1. That is belt to the queue
    # routing's braces (`workers/celery_app.py`) — routing is what stops a task
    # crossing over; separate databases keep result keys and locks apart too.
    redis_url: str = "redis://localhost:6379/0"

    # GCP Cloud Storage — videos and frames
    gcs_bucket_name: Optional[str] = None

    # Supabase
    supabase_url: Optional[str] = None
    supabase_service_key: Optional[str] = None   # service_role key — DB writes, bypasses RLS
    supabase_anon_key: Optional[str] = None       # anon/public key — JWT verification
    # JWT secret (Dashboard → Settings → API → JWT Secret) — enables local HS256
    # token verification instead of a network round-trip to Supabase per request
    supabase_jwt_secret: Optional[str] = None

    # Vertex AI — service account JSON string (used by plan_trip, plan_day_out)
    google_service_account_json: Optional[str] = None  # full JSON as a single env var string

    # Bulk Explore Seeder — shared secret for POST /api/v1/bulk-generate-explore
    bulk_explore_api_key: Optional[str] = None

    # Internal admin API (travbuddy_dashboard) — shared passcode sent as
    # X-Admin-Passcode. Unset means every /api/v1/admin/* route 503s, so a
    # missing env var fails closed instead of opening the admin surface.
    admin_api_passcode: Optional[str] = None

    # Transactional email (Resend). Admin-triggered mail (approve/reject) is
    # sent from here; traveller/partner mail from the website. No key = no-op.
    resend_api_key: Optional[str] = None
    resend_from: str = "TravBuddy <hello@travbuddy.in>"
    site_url: str = "https://travbuddy.in"

    # Shared with the website's /api/revalidate route. Unset means the SEO ping
    # is skipped, which is the right default for a development box and for any
    # deployment with no website in front of it — see services/pms/seo_ping.py.
    revalidate_secret: Optional[str] = None

    # Rotating proxy — Instagram requests go through this FIRST so Instagram
    # never sees (and blocks) the server's own IP. Direct is the fallback.
    # Format: http://user:pass@host:port/
    instagram_proxy_url: Optional[str] = None

    # Celery concurrency — keep low to avoid Instagram rate limits per IP
    celery_worker_concurrency: int = 8

    # App
    app_env: str = "development"
    # Serve /docs, /redoc and /openapi.json. Off unless explicitly switched on:
    # the schema is a complete map of every route, parameter and model we have,
    # and it was being served to the public internet. Deliberately its own flag
    # rather than `app_env != "production"` — prod is currently deployed with
    # APP_ENV unset, so an env-name check would have left it exposed.
    expose_api_docs: bool = False
    log_level: str = "INFO"
    temp_dir: str = "/tmp/travbuddy"

    # Sentry — error tracking (project: travbuddy_backend, id: 4511325237346304)
    sentry_dsn: Optional[str] = None
    sentry_traces_sample_rate: float = 0.1
    sentry_profiles_sample_rate: float = 0.1

    # Razorpay — paid community plans (INR only for now)
    razorpay_key_id: Optional[str] = None
    razorpay_key_secret: Optional[str] = None
    razorpay_webhook_secret: Optional[str] = None

    # Booking.com affiliate ID (used in flight search headers + booking URLs)
    booking_affiliate_id: str = "304142"

    # ConfirmTKT — Indian Railway API credentials
    confirmtkt_api_key: str = "ct-web!2$"
    confirmtkt_device_id: str = "067d2368-0a5c-4372-abf7-fd33546f1dea"

    # CORS — comma-separated list of allowed origins, or "*" for all
    allowed_origins: str = "*"

    # Threads (and thus thread-local Supabase clients) for the dedicated DB
    # executor. Caps concurrent DB ops per process; raise to relieve /plan
    # contention, but each thread holds its own connection pool.
    db_executor_workers: int = 4

    # Hard ceiling (seconds) on any single request before the server returns 504.
    # Backstops per-endpoint timeouts so a wedged scraper/upstream can't pin a
    # worker connection open forever. 0 disables the middleware.
    request_timeout_seconds: float = 210.0

    # ── Resilience / load-shedding ───────────────────────────────────────────
    # Max trip generations running at once PER WORKER PROCESS. Beyond this, new
    # generations queue (they don't fail) — the trip row exists and the frontend
    # sees a "queued" event until a slot frees. Stops a burst of users from
    # saturating Vertex quota + the proxy + the TLS stack all at once. With
    # uvicorn --workers 4 the box-wide ceiling is 4× this.
    max_concurrent_generations: int = 6

    # Default hard timeout (seconds) for a single provider operation wrapped by a
    # circuit breaker — nothing waits forever. Per-provider overrides live on the
    # breaker singletons (circuit_breaker.py).
    provider_timeout_seconds: float = 25.0

    # Default max concurrent in-flight calls to any one provider, across all
    # trips. Per-provider overrides live on the breaker singletons.
    provider_max_concurrency: int = 8

    # Max concurrent Vertex (Gemini) generateContent calls per process — the
    # tightest external quota, so it gets its own gate.
    vertex_max_concurrency: int = 6


settings = Settings()


def redis_target(url: str | None = None) -> str:
    """
    The Redis this process is pointed at, safe to log.

    Scheme, host, port and database number only — a Redis URL carries its
    password inline (`redis://:secret@host:6379/0`) and this string goes into
    logs and into the health endpoint's error text, both of which get pasted
    into chats and issue trackers.

    Exists because the failure it describes is otherwise unreadable. An unset
    `REDIS_URL` leaves the localhost default in place, and the resulting
    `Error 111 connecting to localhost:6379` reads like Redis is down rather
    than like a variable was never set — which is what it actually means in a
    container, where nothing listens on localhost.
    """
    from urllib.parse import urlsplit

    raw = url if url is not None else settings.redis_url
    try:
        parts = urlsplit(raw)
        host = parts.hostname or "?"
        port = f":{parts.port}" if parts.port else ""
        db = parts.path or ""
        return f"{parts.scheme}://{host}{port}{db}"
    except Exception:  # noqa: BLE001 — a log line must never raise
        return "<unparseable REDIS_URL>"


# Whether anybody actually supplied `REDIS_URL`, recorded once from the settings
# object the app is really using.
#
# Asked of pydantic rather than answered by comparing the value to the default,
# because those differ: `redis://localhost:6379/0` is a legitimate thing to set
# — a compose stack whose Redis really is on localhost — and a value check calls
# that "unset" and prints a boot error about a variable somebody just set. Which
# it did, while verifying the compose stack. A warning that cries wolf once is a
# warning nobody reads twice.
#
# `model_fields_set` covers every source pydantic-settings reads: the
# environment, a `.env` file, an explicit argument. Only the field default is
# absent from it. Captured at import rather than recomputed per call, so the
# health endpoint does not re-read `.env` from disk on every hit.
_REDIS_URL_WAS_PROVIDED = "redis_url" in settings.model_fields_set


def redis_is_unconfigured() -> bool:
    """True when `REDIS_URL` was never supplied and the default survived."""
    return not _REDIS_URL_WAS_PROVIDED

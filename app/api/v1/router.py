import logging

from fastapi import APIRouter

from app.api.v1.admin.catalog import router as admin_catalog_router
from app.api.v1.admin.billing import router as admin_billing_router
from app.api.v1.admin.endpoint import router as admin_router
from app.api.v1.admin.trips import router as admin_trips_router
from app.api.v1.autocomplete.endpoint import router as autocomplete_router
from app.api.v1.discover.endpoint import router as discover_router
from app.api.v1.groups.endpoint import router as groups_router
from app.api.v1.health import router as health_router
from app.api.v1.hostels.endpoint import router as hostels_router
from app.api.v1.hotels.endpoint import router as hotels_router
from app.api.v1.images import router as images_router
from app.api.v1.marketplace.endpoint import router as marketplace_router
from app.api.v1.marketplace.partner_ops import router as marketplace_partner_router
from app.api.v1.marketplace.payments import router as marketplace_payments_router
from app.api.v1.google.endpoint import router as google_router
from app.api.v1.messaging.oauth import router as messaging_oauth_router
from app.api.v1.messaging.webhooks import router as messaging_webhooks_router
from app.api.v1.messaging.whatsapp_onboarding import (
    router as whatsapp_onboarding_router,
)
from app.api.v1.metrics import router as metrics_router
from app.api.v1.onboarding.endpoint import router as onboarding_router
from app.api.v1.plan_trip.endpoint import router as plan_trip_router
from app.api.v1.pms.endpoint import router as pms_router
from app.api.v1.pms.hotel_os import router as hotel_os_router
from app.api.v1.recommend.endpoint import router as recommend_router
from app.api.v1.reels.analyze import router as reels_analyze_router
from app.api.v1.reels.endpoint import router as reels_router
from app.api.v1.reels.jobs import router as reels_jobs_router
from app.api.v1.referral.endpoint import router as referral_router
from app.api.v1.search.endpoint import router as search_router
from app.api.v1.trips.endpoint import router as trips_router
from app.api.v1.users.endpoint import router as users_router
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1")

router.include_router(health_router, tags=["health"])
router.include_router(metrics_router)
router.include_router(plan_trip_router, tags=["ai"])
router.include_router(groups_router)
router.include_router(referral_router)
router.include_router(trips_router)
router.include_router(search_router)
router.include_router(recommend_router, tags=["recommend"])
router.include_router(autocomplete_router)
router.include_router(onboarding_router)
router.include_router(discover_router)
router.include_router(hostels_router)
router.include_router(hotels_router)
router.include_router(marketplace_router)
router.include_router(marketplace_partner_router)
router.include_router(marketplace_payments_router)
# ── the hotel OS, mounted only where its database is ─────────────────────────
#
# **These routes need a different Supabase project to every route above them.**
# The marketplace lives in `uabxtdhxbamuhthxnhid`; the hotel OS lives in
# `rdlozqvuywtgsodunait`, and `get_db()` builds one client from one
# `SUPABASE_URL`. A process configured for the marketplace has no `properties`,
# no `reservations` and no `hotel_websites` table, so every one of these routes
# answers a 500 from PostgREST — including the public ones a guest can reach
# without a login.
#
# That is exactly what happened when StayOS first reached production:
# `/api/v1/pms/public/hotels/{slug}` returned INTERNAL_SERVER_ERROR to anybody
# who asked, because the routes shipped with the marketplace image while the
# schema they read did not. Mounting them behind the flag turns that into an
# honest 404 — this deployment does not serve the hotel OS — and costs the
# marketplace nothing.
#
# StayOS therefore runs as its own deployment of this same codebase with
# `.env.hotelos`, which sets `ENABLE_HOTEL_OS=true` along with the project it
# points at. Default off, so the marketplace is correct with no environment
# change at all; a StayOS deployment that forgets the flag fails as a clean 404
# on every PMS route, which is diagnosable, rather than as a 500 storm.
if settings.enable_hotel_os:
    router.include_router(pms_router)
    # The hotel's own configuration — rooms in full, rules, team, readiness.
    # A second router on the same prefix rather than more of endpoint.py,
    # which is already 5,500 lines. Same flag, same limiter, same dependency.
    router.include_router(hotel_os_router)
    router.include_router(messaging_webhooks_router)
    router.include_router(messaging_oauth_router)
    router.include_router(whatsapp_onboarding_router)
    # Google Business Profile: hotel-OS *and* switched on.
    #
    # Two conditions, not one. The outer flag is about which database this
    # process talks to; the inner one is a launch decision — Google is not part
    # of the two-channel launch, needs a Cloud OAuth client and an approved
    # access request before reviews or posts work, and until then every route
    # here would answer a hotel owner with a wall. Unmounted, they 404, which
    # is the honest answer to "does this deployment do Google".
    if settings.enable_google_business_profile:
        router.include_router(google_router)
    # Repull: Airbnb, Booking.com, Vrbo and tiket.com, plus PMS imports.
    # Hotel-OS *and* switched on, for the same reason as Google: without a key
    # every route here would answer a hotelier with a configuration error.
    # Imported lazily so a deployment without repull-sdk installed still boots.
    if settings.enable_repull:
        from app.api.v1.pms.repull import router as repull_router
        router.include_router(repull_router)
    logger.info("Hotel OS routes mounted (ENABLE_HOTEL_OS=true, db=%s)",
                settings.supabase_url)
else:
    logger.info(
        "Hotel OS routes NOT mounted — set ENABLE_HOTEL_OS=true on a deployment "
        "whose SUPABASE_URL is the hotel-OS project."
    )

router.include_router(admin_router)
router.include_router(admin_billing_router)
router.include_router(admin_trips_router)
router.include_router(admin_catalog_router)
router.include_router(users_router)
router.include_router(reels_router)
router.include_router(reels_analyze_router, tags=["reels"])
router.include_router(reels_jobs_router, tags=["reels"])
router.include_router(images_router, tags=["media"])

"""
Every channel a conversation can carry has to be answerable.

This file exists because the same bug shipped twice.

`list_conversations` asks `window_open()` about every row in a hotel's inbox.
`for_channel` raises on a channel it does not know. So one conversation on an
unhandled channel does not degrade that conversation — it 500s the **whole
Inbox screen**, and the hotel cannot read any of its guest messages.

    internal   the staff console.  Fixed, with a comment explaining it.
    website    the chat on the hotel's own booking page. Added afterwards,
               and broke the same screen in the same way — because the fix
               had been to that one channel rather than to the rule.

The rule is: a channel the database will store is a channel this module has to
have an answer for. The database's own CHECK constraint is the list, which is
why the constant below is a copy of it rather than a hand-written set — a
migration that adds a channel and forgets the policy fails here, at the point
where somebody can still do something about it.
"""
import pytest

from app.services.messaging import policy


# Copied from the CHECK on `conversations.channel`. Keep in step: the point of
# this list is to be the same list.
#
#   check (channel in ('whatsapp','instagram','website','phone','email',
#                      'internal','ota'))
#
# `ota` since migrations/2026-11-28_a_hotel_can_sell_on_the_channels.sql.
STORABLE_CHANNELS = ("whatsapp", "instagram", "website", "phone", "email", "internal", "ota")


@pytest.mark.parametrize("channel", STORABLE_CHANNELS)
def test_every_storable_channel_has_an_answer(channel):
    """
    Either a policy or an explicit "no window". Never an exception.

    This is the assertion that would have caught the website bug the day the
    website chat was written, instead of a hotel's Inbox catching it.
    """
    if channel in policy.WINDOWLESS:
        assert policy.has_window(channel) is False
        return
    # A channel with a policy must produce one without raising.
    assert policy.for_channel(channel) is not None, channel


@pytest.mark.parametrize("channel", STORABLE_CHANNELS)
def test_asking_whether_the_window_is_open_never_raises(channel):
    """
    The exact call `list_conversations` makes on every inbox row.

    A boolean, always. The Inbox is the screen a hotel opens to answer guests,
    and it must not be possible for one guest's choice of channel to take it
    down.
    """
    assert isinstance(policy.has_window(channel), bool)


def test_the_website_chat_carries_no_window():
    """
    Not an oversight — there is no rule to inherit.

    A 24-hour window exists because Meta bills by one. A guest on a hotel's own
    booking page can be replied to while the page is open, and after that there
    is nobody to deliver to at all. Giving it WhatsApp's window would be
    inventing a constraint Meta never applied to a page we serve ourselves.
    """
    assert "website" in policy.WINDOWLESS
    assert policy.has_window("website") is False


def test_an_ota_thread_is_always_open_and_can_still_send():
    """
    No window is not the same as nothing to send to.

    An OTA lets a host answer a guest at any time, so `ota` has no window — but
    unlike the WINDOWLESS channels it has a provider, and folding it into that
    set would have stripped the agent of its sending tools on the one channel
    where every guest has already booked.
    """
    assert "ota" not in policy.WINDOWLESS
    assert policy.has_window("ota") is False
    assert policy.always_open("ota") is True
    pol = policy.for_channel("ota")
    assert pol.supports_templates is False
    assert pol.bills_per_conversation is False
    assert pol.conversation_usage_type is None
    # And the channels that do carry a window are not mistaken for it.
    assert policy.always_open("whatsapp") is False
    assert policy.always_open("internal") is False


def test_a_channel_nobody_has_thought_about_still_raises():
    """
    The original rule, and it stays.

    An unknown channel must not silently inherit WhatsApp's window and template
    behaviour — that is how a message goes out under rules Meta never agreed
    to. Unknown is an error; *known and windowless* is the thing that needed a
    way to be said.
    """
    with pytest.raises(ValueError):
        policy.for_channel("telegram")
    assert policy.has_window("telegram") is True


def test_the_billing_channels_and_the_policy_channels_agree():
    """
    A guest channel that can be charged for is one a conversation can be on.

    `billing.USAGE_BY_GUEST_CHANNEL` prices website, Instagram and WhatsApp.
    If one of those were not storable, the charge would be for a conversation
    that could never exist — and the mismatch would show up as revenue nobody
    could explain.
    """
    from app.services import billing

    for channel in billing.USAGE_BY_GUEST_CHANNEL:
        assert channel in STORABLE_CHANNELS, channel


# ── the key every channel connection depends on ──────────────────────────────


class TestCredentialsKeyIsVisible:
    """
    A missing `MESSAGING_CREDENTIALS_KEY` must be visible before a hotel finds it.

    In production a real hotel completed Instagram's consent screen three times
    in thirty-five seconds. Each time Meta handed us a valid token and each time
    we threw it away, because the key had never been pasted into the deployment.
    The service was healthy, every screen loaded, the OAuth round-trip worked —
    the failure was in the last step of a flow only a customer performs, and the
    screen told them to try again.
    """

    def test_a_missing_key_is_an_unhealthy_dependency(self, monkeypatch):
        from app.api.v1 import health
        from app.config import settings

        monkeypatch.setattr(settings, "messaging_credentials_key", "")
        result = health._check_credentials_key()
        assert result.startswith("error:")
        # The sentence has to say what it costs, or it reads as one more
        # optional integration nobody has got round to.
        assert "connect" in result.lower()

    def test_a_malformed_key_is_caught_too(self, monkeypatch):
        from app.api.v1 import health
        from app.config import settings

        monkeypatch.setattr(settings, "messaging_credentials_key", "not-a-fernet-key")
        assert health._check_credentials_key().startswith("error:")

    def test_a_real_key_passes(self, monkeypatch):
        from cryptography.fernet import Fernet

        from app.api.v1 import health
        from app.config import settings

        monkeypatch.setattr(settings, "messaging_credentials_key",
                            Fernet.generate_key().decode())
        assert health._check_credentials_key() == "ok"

    def test_the_key_itself_never_appears_in_the_answer(self, monkeypatch):
        """The health endpoint is public on some deployments. It says whether,
        never what."""
        from cryptography.fernet import Fernet

        from app.api.v1 import health
        from app.config import settings

        key = Fernet.generate_key().decode()
        monkeypatch.setattr(settings, "messaging_credentials_key", key)
        assert key not in health._check_credentials_key()
        monkeypatch.setattr(settings, "messaging_credentials_key", "bad" + key)
        assert key not in health._check_credentials_key()


def test_a_hotel_is_not_asked_to_retry_a_key_we_have_not_set():
    """
    The owner-facing half.

    `save_failed` says "please try again", which is right for a dropped
    connection and wrong for a missing key — retrying is futile and the hotel
    cannot fix it. The callback distinguishes the two, and this asserts the
    distinction exists rather than the wording, which is the PWA's.
    """
    import inspect

    from app.api.v1.messaging import oauth

    source = inspect.getsource(oauth.instagram_oauth_callback)
    assert "CredentialCryptoError" in source
    assert "save_not_configured" in source
    # And the generic handler still exists for everything else.
    assert "save_failed" in source

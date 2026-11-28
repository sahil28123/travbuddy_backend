"""
A deterministic OTA channel, for development and for the test suite.

The same argument as `instagram/fake.py`: a second implementation of
`ChannelAdapter`, selected when the hotel's Repull workspace is the sandbox
one, recording what it was asked to send so a test can assert on it.
Everything above it — the connection lookup, the queue row keyed on
`client_reference`, the agent, the audit trail, the meter — runs unchanged,
which is the only reason a test against it proves anything.

**What it does not prove.** That Repull accepts the payload, that the hotel's
Airbnb or Booking.com account is still connected, that the OTA delivered the
words unaltered. Those need Repull.

`send_template` raises here too. A fake that accepted one would let a caller
assuming WhatsApp's reopen-with-a-template behaviour pass every test and fail
the first time it met a real OTA thread.
"""
import logging
import uuid
from dataclasses import dataclass

from app.services.messaging.channels import (
    PROVIDER_FAILED,
    PROVIDER_UNAVAILABLE,
    TEMPLATE_NOT_FOUND,
    MessagingError,
    SendResult,
)

logger = logging.getLogger(__name__)


@dataclass
class SentOta:
    to: str
    kind: str = "text"           # text · image
    body: str | None = None
    image_url: str | None = None
    client_reference: str = ""
    external_id: str = ""


class FakeOtaAdapter:
    channel = "ota"
    is_sandbox = True

    # Class-level for the same reason the Instagram fake's is: the send path
    # builds its own adapter deep inside a request, and a fake whose output only
    # the constructing code can see is a fake nothing can assert on.
    outbox: list[SentOta] = []
    fail_next: MessagingError | None = None
    timeout_next: bool = False

    def __init__(self, *, workspace_id: str = "sandbox", **_kw):
        self.workspace_id = workspace_id

    @classmethod
    def reset(cls) -> None:
        cls.outbox = []
        cls.fail_next = None
        cls.timeout_next = False

    @classmethod
    def last(cls) -> SentOta | None:
        return cls.outbox[-1] if cls.outbox else None

    def _check_failures(self) -> None:
        if type(self).timeout_next:
            type(self).timeout_next = False
            raise MessagingError(
                PROVIDER_UNAVAILABLE, "fake Repull timed out", retryable=True
            )
        if type(self).fail_next is not None:
            exc = type(self).fail_next
            type(self).fail_next = None
            raise exc

    def _record(self, sent: SentOta) -> SendResult:
        # Repull message ids are numeric strings; these are not, so nothing
        # downstream can come to depend on the format of one.
        sent.external_id = f"ota_msg.FAKE{uuid.uuid4().hex[:16]}"
        type(self).outbox.append(sent)
        logger.debug("fake ota → %s: %s", sent.to, sent.body or sent.image_url)
        return SendResult(external_id=sent.external_id, status="sent",
                          raw={"sandbox": True, "clientReference": sent.client_reference})

    async def send_text(self, *, to: str, body: str, client_reference: str) -> SendResult:
        self._check_failures()
        return self._record(SentOta(to=to, body=body, client_reference=client_reference))

    async def send_image(self, *, to: str, image_url: str, caption: str | None,
                         client_reference: str) -> SendResult:
        """The real one carries the caption with the picture, so this does too."""
        self._check_failures()
        return self._record(SentOta(to=to, kind="image", body=caption, image_url=image_url,
                                    client_reference=client_reference))

    async def send_template(self, *, to: str, template_name: str, language: str,
                            parameters: list[str], client_reference: str) -> SendResult:
        raise MessagingError(
            TEMPLATE_NOT_FOUND,
            "OTA threads have no approved-template mechanism, in the fake as in the real one.",
        )


def fail_with(code: str = PROVIDER_FAILED, detail: str = "fake failure",
              *, retryable: bool = False) -> None:
    FakeOtaAdapter.fail_next = MessagingError(code, detail, retryable=retryable)

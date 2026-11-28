"""
OTA guest messages, out — through Repull.

A guest who booked on Airbnb, Booking.com, Vrbo or tiket.com writes to the
hotel inside that OTA's message thread. Repull (https://repull.dev) carries
those threads for every OTA behind one endpoint, so this adapter is one class
rather than four:

    WhatsApp / Instagram              OTA (via Repull)
    ─────────────────────────────     ──────────────────────────────────────
    to: a phone number / an IGSID     to: the Repull conversation id
    a 24-hour window                  no window — a host may answer any time
    templates reopen WhatsApp         no templates, and nothing to reopen
    the provider's message id         Repull's message id
    a Meta token per account          the operator's REPULL_API_KEY, scoped
                                        to the hotel's workspace

**Idempotent at the provider, not only here.** `service.send()` already writes
a queued row keyed on `client_reference` before the request leaves, so a retry
of a send we *know* finished returns early. What that cannot cover is the send
that reached the OTA and timed out on the way back. So the same reference goes
to Repull as `Idempotency-Key`: a retried request is answered with the first
one's result instead of becoming a second message to the guest.

**Queued is not failed.** Vrbo delivers through a thread that is sometimes
busy; Repull then records the message, answers `202` with `status: "queued"`
and finishes in the background. That is reported as `queued`, never retried —
a resend would be the guest's second copy.

**What the error means, not just that there was one.** Repull's error envelope
carries a `fix` sentence and whether the call is worth retrying. Both survive
into the `MessagingError` the agent and the inbox see: `PROVIDER_UNAVAILABLE`
(retryable) for a rate limit or an outage, `PROVIDER_FAILED` for a send that
will never work as written, `CHANNEL_NOT_CONFIGURED` when the deployment has no
key at all.

`send_template` refuses, as Instagram's does, and for the same reason: a caller
asking for a template here has made a mistake about the channel, and answering
with free text would hide it behind a successful-looking send.
"""
import logging

from app.services.messaging.channels import (
    NOT_CONFIGURED,
    PROVIDER_FAILED,
    PROVIDER_UNAVAILABLE,
    TEMPLATE_NOT_FOUND,
    MessagingError,
    SendResult,
)
from app.services.repull import client as repull_client
from app.services.repull.client import RepullError

logger = logging.getLogger(__name__)


def _as_messaging_error(exc: RepullError) -> MessagingError:
    """
    Repull's failure, in the vocabulary the agent branches on.

    Repull's own `fix` goes into the detail untouched — "Reconnect the Airbnb
    account at …" is better advice than anything a mapping here could invent,
    and it is what a person reading the failed message in the inbox needs.
    """
    detail = f"Repull {exc.code}: {exc.message}" + (f" — {exc.fix}" if exc.fix else "")
    if exc.code == "not_configured":
        return MessagingError(NOT_CONFIGURED, detail)
    if exc.retryable:
        return MessagingError(PROVIDER_UNAVAILABLE, detail, retryable=True)
    return MessagingError(PROVIDER_FAILED, detail)


def _conversation_id(to: str) -> int:
    """
    The Repull conversation id, which the API takes as an integer.

    A non-numeric `to` is a conversation row that was never an OTA thread;
    refused here rather than sent to Repull as a 404 that reads like a deleted
    conversation.
    """
    try:
        return int(str(to).strip())
    except (TypeError, ValueError):
        raise MessagingError(
            PROVIDER_FAILED, f"{to!r} is not a Repull conversation id"
        ) from None


class OtaAdapter:
    """Sends into a Repull conversation. Holds a workspace id; reads no database."""

    channel = "ota"
    is_sandbox = False

    def __init__(self, *, workspace_id: str):
        self.workspace_id = str(workspace_id)

    async def _send(self, *, to: str, message: str | None, attachments: list[dict],
                    client_reference: str) -> SendResult:
        from repull.api.conversations import send_conversation_message
        from repull.models.send_message_request import SendMessageRequest

        payload: dict = {}
        if message:
            payload["message"] = message
        if attachments:
            payload["attachments"] = attachments

        try:
            body = await repull_client.call(
                send_conversation_message.asyncio_detailed,
                workspace_id=self.workspace_id,
                id=_conversation_id(to),
                body=SendMessageRequest.from_dict(payload),
                idempotency_key=client_reference,
            )
        except RepullError as exc:
            mapped = _as_messaging_error(exc)
            logger.warning("ota send failed (ref=%s): %s", client_reference, mapped.detail)
            raise mapped from exc

        body = body if isinstance(body, dict) else {}
        status = "queued" if body.get("status") == "queued" else "sent"
        return SendResult(external_id=body.get("id"), status=status, raw=body)

    async def send_text(self, *, to: str, body: str, client_reference: str) -> SendResult:
        return await self._send(to=to, message=body, attachments=[],
                                client_reference=client_reference)

    async def send_image(self, *, to: str, image_url: str, caption: str | None,
                         client_reference: str) -> SendResult:
        """
        A photograph as an attachment, with the caption as the message text.

        Repull fetches the URL and uploads it to the OTA itself, so — as with
        Instagram — the bucket must be publicly readable and nothing here moves
        bytes. Unlike Instagram, the words and the picture travel in one call;
        Repull splits them into the OTA's own messages in the right order.
        """
        return await self._send(
            to=to, message=caption,
            attachments=[{"url": image_url, "contentType": _content_type(image_url)}],
            client_reference=client_reference,
        )

    async def send_template(self, *, to: str, template_name: str, language: str,
                            parameters: list[str], client_reference: str) -> SendResult:
        raise MessagingError(
            TEMPLATE_NOT_FOUND,
            "OTA threads have no approved-template mechanism, and need none: the "
            "thread never closes, so send the message as free text.",
        )


def _content_type(url: str) -> str:
    """A best guess from the extension. Repull re-detects it; this is a hint."""
    path = url.split("?", 1)[0].lower()
    if path.endswith(".png"):
        return "image/png"
    if path.endswith(".webp"):
        return "image/webp"
    if path.endswith(".gif"):
        return "image/gif"
    return "image/jpeg"

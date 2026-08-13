"""TelegramChannel — native Telegram Bot API adapter."""

from __future__ import annotations

import asyncio
import logging
import os
import textwrap
import threading
from typing import Any, Dict, List, Optional

from openjarvis.channels._stubs import (
    BaseChannel,
    ChannelHandler,
    ChannelMessage,
    ChannelStatus,
)
from openjarvis.core.events import EventBus, EventType
from openjarvis.core.registry import ChannelRegistry

logger = logging.getLogger(__name__)


@ChannelRegistry.register("telegram")
class TelegramChannel(BaseChannel):
    """Native Telegram channel adapter using the Bot API.

    Parameters
    ----------
    bot_token:
        Telegram Bot API token.  Falls back to ``TELEGRAM_BOT_TOKEN`` env var.
    allowed_chat_ids:
        Comma-separated list of chat IDs allowed to interact.
    parse_mode:
        Message parse mode (``Markdown``, ``HTML``, etc.).
    bus:
        Optional event bus for publishing channel events.
    """

    channel_id = "telegram"

    def __init__(
        self,
        bot_token: str = "",
        *,
        allowed_chat_ids: str = "",
        parse_mode: str = "Markdown",
        bus: Optional[EventBus] = None,
        control_plane_url: str = "",
        control_plane_shared_secret: str = "",
        control_plane_queue_poll_interval: float = 3.0,
    ) -> None:
        self._token = bot_token or os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self._allowed_chat_ids = allowed_chat_ids
        self._parse_mode = parse_mode
        self._bus = bus
        self._handlers: List[ChannelHandler] = []
        self._status = ChannelStatus.DISCONNECTED
        self._listener_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        # Phase B: when set, Telegram is expected to be webhooked to the
        # control plane (Telegram only allows one consumption mode at a
        # time), so incoming messages are pulled from its queue instead of
        # python-telegram-bot's getUpdates long polling. Empty (the
        # default) keeps today's polling behavior completely unchanged.
        self._control_plane_url = control_plane_url
        self._control_plane_secret = control_plane_shared_secret or os.environ.get(
            "CONTROL_PLANE_SHARED_SECRET", ""
        )
        self._control_plane_queue_poll_interval = control_plane_queue_poll_interval

    # -- connection lifecycle ---------------------------------------------------

    def connect(self) -> None:
        """Start listening for incoming messages: via the control plane's
        queue (Phase B, when configured) or Telegram long polling (default,
        unchanged behavior)."""
        if not self._token:
            logger.warning("No Telegram bot token configured")
            self._status = ChannelStatus.ERROR
            return

        self._stop_event.clear()
        self._status = ChannelStatus.CONNECTING

        if self._control_plane_url:
            self._listener_thread = threading.Thread(
                target=self._queue_poll_loop,
                daemon=True,
            )
            self._listener_thread.start()
            self._status = ChannelStatus.CONNECTED
            logger.info(
                "Telegram channel connected (control plane queue: %s)",
                self._control_plane_url,
            )
            return

        try:
            from telegram.ext import ApplicationBuilder  # noqa: F401

            self._listener_thread = threading.Thread(
                target=self._poll_loop,
                daemon=True,
            )
            self._listener_thread.start()
            self._status = ChannelStatus.CONNECTED
            logger.info("Telegram channel connected (long polling)")
        except ImportError:
            # python-telegram-bot not installed — send-only mode
            logger.info(
                "python-telegram-bot not installed; send-only mode",
            )
            self._status = ChannelStatus.CONNECTED

    def disconnect(self) -> None:
        """Stop the listener thread."""
        self._stop_event.set()
        if self._listener_thread is not None:
            self._listener_thread.join(timeout=5.0)
            self._listener_thread = None
        self._status = ChannelStatus.DISCONNECTED

    # -- send / receive --------------------------------------------------------

    def send(
        self,
        channel: str,
        content: str,
        *,
        conversation_id: str = "",
        metadata: Dict[str, Any] | None = None,
    ) -> bool:
        """Send a message to a Telegram chat via the Bot API."""
        if not self._token:
            logger.warning("Cannot send: no Telegram bot token")
            return False

        try:
            import httpx

            _TELEGRAM_MAX_LEN = 4096
            url = f"https://api.telegram.org/bot{self._token}/sendMessage"
            # Canonical channel send contract (see BaseChannel.send): the first
            # positional ``channel`` arg is the DESTINATION (the Telegram chat
            # id).  ``conversation_id`` is the inbound message id used as a
            # reply/thread reference (``reply_to_message_id``).  We fall back to
            # ``conversation_id`` as the chat id only when ``channel`` is empty,
            # for backwards compatibility with legacy callers that passed the
            # chat id via ``conversation_id``.
            chat_id = channel or conversation_id
            reply_to = conversation_id if (channel and conversation_id) else ""
            chunks = textwrap.wrap(
                content,
                width=_TELEGRAM_MAX_LEN,
                break_long_words=True,
                replace_whitespace=False,
            )
            for chunk in chunks:
                payload: Dict[str, Any] = {
                    "chat_id": chat_id,
                    "text": chunk,
                }
                if self._parse_mode:
                    payload["parse_mode"] = self._parse_mode
                if reply_to:
                    payload["reply_to_message_id"] = reply_to

                resp = httpx.post(url, json=payload, timeout=10.0)
                if resp.status_code >= 300:
                    logger.warning(
                        "Telegram API returned status %d: %s",
                        resp.status_code,
                        resp.text,
                    )
                    return False
            self._publish_sent(channel, content, conversation_id)
            return True
        except Exception:
            logger.debug("Telegram send failed", exc_info=True)
            return False

    def send_photo(
        self,
        channel: str,
        photo_path: str,
        *,
        caption: str = "",
        conversation_id: str = "",
    ) -> bool:
        """Send a local image file as a Telegram photo (``sendPhoto``).

        Added for the visual-proof pipeline (2026-08-13): missions and the
        ``show_current_state`` tool both need to deliver a screenshot as an
        actual photo, not a wall of text -- ``send()`` only ever does
        ``sendMessage``. Mirrors ``send()``'s chat-id/reply-to resolution.
        """
        if not self._token:
            logger.warning("Cannot send photo: no Telegram bot token")
            return False
        try:
            import httpx

            chat_id = channel or conversation_id
            reply_to = conversation_id if (channel and conversation_id) else ""
            url = f"https://api.telegram.org/bot{self._token}/sendPhoto"
            data: Dict[str, Any] = {"chat_id": chat_id}
            if caption:
                # Telegram photo captions are capped at 1024 chars (vs 4096
                # for text messages) -- truncate rather than fail the send.
                data["caption"] = caption[:1024]
            if reply_to:
                data["reply_to_message_id"] = reply_to
            with open(photo_path, "rb") as fh:
                resp = httpx.post(
                    url, data=data, files={"photo": fh}, timeout=30.0
                )
            if resp.status_code >= 300:
                logger.warning(
                    "Telegram sendPhoto returned status %d: %s",
                    resp.status_code,
                    resp.text,
                )
                return False
            return True
        except Exception:
            logger.debug("Telegram send_photo failed", exc_info=True)
            return False

    def status(self) -> ChannelStatus:
        """Return the current connection status."""
        return self._status

    def list_channels(self) -> List[str]:
        """Return available channel identifiers."""
        return ["telegram"]

    def on_message(self, handler: ChannelHandler) -> None:
        """Register a callback for incoming messages."""
        self._handlers.append(handler)

    # -- internal helpers -------------------------------------------------------

    def _poll_loop(self) -> None:
        """Long-poll for updates using python-telegram-bot."""
        try:
            from telegram.ext import ApplicationBuilder, MessageHandler, filters

            app = ApplicationBuilder().token(self._token).build()

            async def _handle_msg(update, context):
                msg = update.message
                if msg is None:
                    return
                cm = ChannelMessage(
                    channel="telegram",
                    sender=str(msg.from_user.id) if msg.from_user else "",
                    content=msg.text or "",
                    message_id=str(msg.message_id),
                    conversation_id=str(msg.chat.id),
                )
                # Enforce allow-list when configured
                if self._allowed_chat_ids:
                    _allowed = {
                        cid.strip()
                        for cid in self._allowed_chat_ids.split(",")
                        if cid.strip()
                    }
                    if cm.conversation_id not in _allowed:
                        logger.debug(
                            "Ignoring message from unlisted chat %s",
                            cm.conversation_id,
                        )
                        return
                for handler in self._handlers:
                    try:
                        await asyncio.to_thread(handler, cm)
                    except Exception:
                        logger.exception("Telegram handler error")
                if self._bus is not None:
                    self._bus.publish(
                        EventType.CHANNEL_MESSAGE_RECEIVED,
                        {
                            "channel": cm.channel,
                            "sender": cm.sender,
                            "content": cm.content,
                            "message_id": cm.message_id,
                        },
                    )

            app.add_handler(MessageHandler(filters.TEXT, _handle_msg))
            app.run_polling(stop_signals=None, drop_pending_updates=True)
        except Exception:
            logger.debug("Telegram poll loop error", exc_info=True)
            self._status = ChannelStatus.ERROR

    def _queue_poll_loop(self) -> None:
        """Phase B: pull pending messages from the control plane's queue
        instead of long-polling Telegram directly (fed by Telegram's
        webhook -> the Worker -> D1). Builds the exact same ChannelMessage
        and calls the exact same handlers as _poll_loop -- tools, mission
        engine, memory are all unchanged, only how the message arrives
        differs. Replies still go straight to Telegram's API via send(),
        not through the control plane.

        The allow-list gate already happened at the Worker (same secret,
        same TELEGRAM_OWNER_ID check) before a message ever reaches this
        queue, but it's re-checked here too -- defense in depth, and it's
        what keeps this method's behavior identical to _poll_loop's own
        allow-list check rather than blindly trusting the network hop.
        """
        import httpx

        url = self._control_plane_url.rstrip("/") + "/telegram-queue"
        headers = {
            "x-control-plane-secret": self._control_plane_secret,
            "user-agent": "JARVIS-PC-Worker/1.0",
        }
        allowed = {
            cid.strip() for cid in self._allowed_chat_ids.split(",") if cid.strip()
        }

        while not self._stop_event.is_set():
            try:
                resp = httpx.get(url, headers=headers, timeout=10.0)
                if resp.status_code == 200:
                    for m in resp.json().get("messages", []):
                        cm = ChannelMessage(
                            channel="telegram",
                            sender=str(m.get("sender_id", "")),
                            content=m.get("content", ""),
                            message_id=str(m.get("message_id", "")),
                            conversation_id=str(m.get("chat_id", "")),
                        )
                        if allowed and cm.conversation_id not in allowed:
                            logger.debug(
                                "Ignoring queued message from unlisted chat %s",
                                cm.conversation_id,
                            )
                            continue
                        for handler in self._handlers:
                            try:
                                handler(cm)
                            except Exception:
                                logger.exception("Telegram queue handler error")
                        if self._bus is not None:
                            self._bus.publish(
                                EventType.CHANNEL_MESSAGE_RECEIVED,
                                {
                                    "channel": cm.channel,
                                    "sender": cm.sender,
                                    "content": cm.content,
                                    "message_id": cm.message_id,
                                },
                            )
                else:
                    logger.debug(
                        "Telegram queue poll returned status %d", resp.status_code
                    )
            except Exception:
                logger.debug("Telegram queue poll failed", exc_info=True)
            self._stop_event.wait(self._control_plane_queue_poll_interval)

    def _publish_sent(self, channel: str, content: str, conversation_id: str) -> None:
        """Publish a CHANNEL_MESSAGE_SENT event on the bus."""
        if self._bus is not None:
            self._bus.publish(
                EventType.CHANNEL_MESSAGE_SENT,
                {
                    "channel": channel,
                    "content": content,
                    "conversation_id": conversation_id,
                },
            )


__all__ = ["TelegramChannel"]

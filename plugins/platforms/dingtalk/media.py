"""DingTalk inbound media: downloadCode → temporary URL → local cache file.

Official contract (``POST /v1.0/robot/messageFiles/download``):

* Body fields are ``downloadCode`` (the callback's code) and ``robotCode`` (the bot that
  received that message). Header ``x-acs-dingtalk-access-token`` carries the app token.
* The response ``downloadUrl`` is a short-lived http(s) link. Bytes are fetched from that
  link; DingTalk does not return the file in the API body.
* Native admission keeps only absolute regular files under the platform cache. A remote
  URL left in ``media_urls`` is dropped, so a resolved URL must be cached before the event
  is built.

HTTP 500 ``unknownError`` is a server-side response. The documented client errors for an
empty or mismatched ``robotCode`` / ``downloadCode`` are HTTP 400, so 500 is not evidence
of a missing permission. This module never logs the download code, access token, or the
signed download URL.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from plugins.platforms.dingtalk.inbound import EXT_MAP

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
_AUDIO_EXTS = {".ogg", ".mp3", ".wav", ".m4a", ".aac"}
_QUERY_SECRET = re.compile(r"(?i)(signature|sig|token|access_token|downloadcode)=([^&\s]+)")
_ITEM_KIND = {
    "picture": "image", "image": "image", "voice": "audio", "audio": "audio",
    "video": "video", "file": "document",
}
_KEPT = " The message text was kept and the attachment was not admitted."


def robot_code_for(message: Any, configured: str, client_id: str) -> str:
    """Bot that received the message. The download API rejects a code paired with another bot.

    Callback ``robotCode`` wins. Otherwise the configured ``extra.robot_code`` (which already
    defaults to the app key / client id), then the client id.
    """
    extensions = getattr(message, "extensions", None)
    from_extensions = ""
    if isinstance(extensions, dict):
        raw = extensions.get("robotCode") or extensions.get("robot_code") or ""
        from_extensions = raw if isinstance(raw, str) else ""
    for value in (getattr(message, "robot_code", None), from_extensions, configured, client_id):
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def redact_code(code: str) -> str:
    """Correlate a code in logs without keeping a usable download credential."""
    raw = code or ""
    digest = hashlib.sha256(raw.encode("utf-8", "replace")).hexdigest()[:8]
    return f"len={len(raw)} sha256={digest}"


def is_remote_url(value: str) -> bool:
    """True for an http(s) URL with a host. ``file:`` and bare paths are not download URLs."""
    parsed = urlsplit((value or "").strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def read_field(obj: Any, key: str) -> str:
    value = obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)
    return value.strip() if isinstance(value, str) else ""


def write_field(obj: Any, key: str, value: str) -> None:
    if isinstance(obj, dict):
        obj[key] = value
    elif hasattr(obj, key):
        setattr(obj, key, value)


def filename_of(obj: Any) -> str:
    if not isinstance(obj, dict):
        return ""
    name = obj.get("fileName") or obj.get("filename") or ""
    return name if isinstance(name, str) else ""


def kind_for(message: Any, obj: Any, key: str) -> str:
    """Cache bucket for one code. Unknown files stay documents so a PDF is not image-checked."""
    if key == "download_code" and not isinstance(obj, dict):
        return "image"
    if isinstance(obj, dict):
        item = str(obj.get("type") or "").strip()
        if item in _ITEM_KIND:
            return _ITEM_KIND[item]
        fname = filename_of(obj)
        ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else ""
        mime = EXT_MAP.get(ext, "")
        if mime.startswith("image/"):
            return "image"
        if mime.startswith("audio/"):
            return "audio"
        if mime.startswith("video/"):
            return "video"
        if fname:
            return "document"
    msg_type = str(getattr(message, "message_type", "") or "")
    if msg_type in {"picture", "image"}:
        return "image"
    if msg_type == "video":
        return "video"
    if msg_type in {"audio", "voice"}:
        return "audio"
    return "document"


def needs_download_api(code: str) -> bool:
    """True when ``code`` is still a download code, not a URL, cache file, or local path."""
    return not (is_remote_url(code) or looks_like_local_path(code) or bool(cache_path_if_safe(code)))


def looks_like_local_path(code: str) -> bool:
    """An existing local file (or a ``file:`` URL), not a DingTalk download code.

    Codes may contain ``/`` and occasionally start with ``/``. Only an existing file, or a
    ``file:`` URL, is treated as a path so those codes still go to the download API.
    """
    raw = (code or "").strip()
    if raw.lower().startswith("file:"):
        return True
    if not raw or not os.path.isabs(raw):
        return False
    try:
        return Path(raw).is_file()
    except OSError:
        return False


def cache_roots() -> list[Path]:
    from gateway.platforms.base import (
        get_audio_cache_dir, get_document_cache_dir, get_image_cache_dir, get_video_cache_dir,
    )
    return [get_image_cache_dir(), get_audio_cache_dir(), get_video_cache_dir(), get_document_cache_dir()]


def cache_path_if_safe(path: str) -> str:
    """Absolute regular file inside a platform cache directory, else ``""``.

    Symlinks and paths outside the cache (including ``/etc/passwd``) are refused. The returned
    path is the resolved file, which is what admission may open.
    """
    if not isinstance(path, str) or not path or not os.path.isabs(path):
        return ""
    candidate = Path(path)
    try:
        if candidate.is_symlink() or not candidate.is_file():
            return ""
        resolved = candidate.resolve()
    except OSError:
        return ""
    if not resolved.is_file():
        return ""
    for root in cache_roots():
        try:
            root_resolved = root.resolve()
        except OSError:
            continue
        if root_resolved == resolved or root_resolved in resolved.parents:
            return str(resolved)
    return ""


def download_url_from(response: Any) -> str:
    """``downloadUrl`` from the robot file-download response body (model or dict)."""
    body = getattr(response, "body", None) if response is not None else None
    if body is None and isinstance(response, dict):
        body = response
    if isinstance(body, dict):
        value = body.get("downloadUrl") or body.get("download_url") or ""
    else:
        value = getattr(body, "download_url", None) or getattr(body, "downloadUrl", None) or ""
    return value.strip() if isinstance(value, str) else ""


def _suffix(filename: str) -> str:
    return Path(filename).suffix.lower() if filename else ""


def _strip_secrets(text: str, secrets: tuple[str, ...]) -> str:
    for secret in secrets:
        if isinstance(secret, str) and len(secret) >= 8 and secret in text:
            text = text.replace(secret, "[redacted]")
    return _QUERY_SECRET.sub(r"\1=[redacted]", text)


def scrub_exception(exc: BaseException, secrets: tuple[str, ...]) -> None:
    """Drop download codes, tokens and signed URLs from an exception before it is logged."""
    if exc.args:
        exc.args = tuple(_strip_secrets(arg, secrets) if isinstance(arg, str) else arg for arg in exc.args)
    message = getattr(exc, "message", None)
    if isinstance(message, str):
        setattr(exc, "message", _strip_secrets(message, secrets))
    data = getattr(exc, "data", None)
    if isinstance(data, dict):
        setattr(exc, "data", {
            key: _strip_secrets(val, secrets) if isinstance(val, str) else val for key, val in data.items()
        })


def describe_failure(exc: BaseException, *, secrets: tuple[str, ...] = ()) -> str:
    """Short public summary. Never the raw exception text (it may quote the request)."""
    try:
        import httpx
    except ImportError:  # pragma: no cover - httpx is a DingTalk extra
        httpx = None
    if httpx is not None and isinstance(exc, httpx.TimeoutException):
        return "download timed out"
    if httpx is not None and isinstance(exc, httpx.HTTPStatusError):
        status = getattr(getattr(exc, "response", None), "status_code", None)
        return f"download HTTP {status}" if status else "download HTTP error"
    if isinstance(exc, ValueError):
        lower = str(exc).lower()
        if "too large" in lower:
            return "file exceeds the inbound size limit"
        if any(token in lower for token in ("unsafe", "ssrf", "blocked", "private")):
            return "download URL blocked by safety check"
        if "non-image" in lower:
            return "payload is not an image"
        return "local cache rejected the payload"
    status = getattr(exc, "status_code", None) or getattr(exc, "statusCode", None)
    code = getattr(exc, "code", None)
    message = getattr(exc, "message", None)
    data = getattr(exc, "data", None)
    request_id = data.get("requestid") or data.get("requestId") if isinstance(data, dict) else None
    parts: list[str] = []
    if status:
        parts.append(f"HTTP {status}")
    if isinstance(code, str) and code and len(code) <= 80 and " " not in code:
        parts.append(code)
    if isinstance(message, str) and message and len(message) <= 80:
        parts.append(_strip_secrets(message, secrets))
    if isinstance(request_id, str) and 8 <= len(request_id) <= 80:
        safe_id = _strip_secrets(request_id, secrets)
        if safe_id == request_id:
            parts.append(f"requestId={request_id}")
    if parts:
        return _strip_secrets(" ".join(parts), secrets)[:240]
    if type(exc) is RuntimeError:
        text = _strip_secrets(str(exc), secrets)
        if text and len(text) <= 160 and "http" not in text.lower():
            return text
    return type(exc).__name__


def failure_note(stage: str, summary: str = "") -> str:
    """One actionable sentence for the agent. No codes, tokens, or URLs."""
    detail = f" ({summary})" if summary else ""
    if stage == "token":
        text = "DingTalk attachment was not downloaded: access token unavailable."
    elif stage == "robot":
        text = ("DingTalk attachment was not downloaded: robotCode is empty. "
                "Use the callback robotCode, or platforms.dingtalk.extra.robot_code.")
    elif stage == "sdk":
        text = "DingTalk attachment was not downloaded: robot SDK is not initialized."
    elif stage == "empty":
        text = "DingTalk file-download API returned no downloadUrl."
    elif stage == "api":
        text = (f"DingTalk file-download API failed{detail}. HTTP 500 alone does not show a missing "
                "permission; a bad or mismatched robotCode/downloadCode is documented as HTTP 400.")
    elif stage == "unsafe":
        text = "DingTalk media was rejected by the local cache safety check and was not admitted."
    else:
        text = f"DingTalk download URL could not be saved locally{detail}."
    return f"[{text}{_KEPT}]"


async def cache_resolved_url(url: str, *, kind: str, filename: str = "") -> str:
    """Download ``url`` with the shared size-, time- and SSRF-bounded cache helpers.

    Images and audio use the public ``cache_*_from_url`` helpers (30s client timeout, inbound
    size cap, unsafe-URL rejection). Video and other files use the same downloader, then the
    matching ``cache_*_from_bytes`` writer. The signed query string is never logged here.
    """
    if not is_remote_url(url):
        raise ValueError("Blocked unsafe URL (not an http(s) download)")
    from gateway.platforms.base import (
        _cache_media_from_url, cache_audio_from_url, cache_document_from_bytes,
        cache_image_from_url, cache_video_from_bytes,
    )
    if kind == "image":
        ext = _suffix(filename)
        return await cache_image_from_url(url, ext=ext if ext in _IMAGE_EXTS else ".jpg")
    if kind == "audio":
        ext = _suffix(filename)
        return await cache_audio_from_url(url, ext=ext if ext in _AUDIO_EXTS else ".ogg")
    if kind == "video":
        ext = _suffix(filename) or ".mp4"

        def _store_video(data: bytes, ext_: str) -> str:
            return cache_video_from_bytes(data, ext_ or ext)

        return await _cache_media_from_url(
            url, ext, 2, media_type="video", accept="video/*,*/*;q=0.8",
            cache_fn=_store_video, log_label="DingTalk")
    ext = _suffix(filename) or ".bin"

    def _store_doc(data: bytes, ext_: str) -> str:
        return cache_document_from_bytes(data, filename or f"dingtalk-file{ext_ or '.bin'}")

    return await _cache_media_from_url(
        url, ext, 2, media_type="document", accept="*/*",
        cache_fn=_store_doc, log_label="DingTalk")

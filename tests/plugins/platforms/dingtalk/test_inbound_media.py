"""Synthetic offline tests for DingTalk inbound images and shared files.

No live DingTalk API, credentials, or network. The robot SDK and the HTTP download
are fakes. Admission is the real local-file guard: remote URLs stay dropped, and a
cached regular file is what a native turn is allowed to open.
"""

import asyncio
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from gateway.platforms.event import MessageType

TOKEN = "token-SECRET-do-not-log-aaaa"
SIG = "SUPERSECRETVALUE"
CODE = "DLCODE-secret-download-code-value"
OTHER = "DLCODE-second-secret-download-value"
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 32
PDF = b"%PDF-1.4\n%synthetic\n"


class _Model:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _Image:
    def __init__(self, code):
        self.download_code = code


class _ApiFailure(Exception):
    def __init__(self, download_code, token):
        super().__init__(
            f"Error: unknownError code: 500 for {download_code} Authorization {token} Signature={SIG}"
        )
        self.code = "unknownError"
        self.status_code = 500
        self.message = "未知错误"
        self.data = {"requestid": "REQ-SYNTH-1", "downloadCode": download_code}


def _message(**kwargs):
    msg = SimpleNamespace(
        message_type=kwargs.get("message_type", "picture"),
        text=kwargs.get("text"),
        image_content=kwargs.get("image_content"),
        rich_text_content=kwargs.get("rich_text_content"),
        rich_text=kwargs.get("rich_text"),
        extensions=kwargs.get("extensions") or {},
        robot_code=kwargs.get("robot_code", "robot-from-callback"),
        message_id=kwargs.get("message_id", "m-1"),
        conversation_id=kwargs.get("conversation_id", "cid-1"),
        conversation_type=kwargs.get("conversation_type", "1"),
        sender_id="user-1",
        sender_nick="Ada",
        sender_staff_id="staff-1",
        session_webhook="",
        session_webhook_expired_time=0,
        create_at=None,
        conversation_title="chat",
        is_in_at_list=False,
    )
    return msg


def _signed_url(name="img.jpg"):
    return f"https://files.example.test/{name}?Signature={SIG}"


def _stream_client(body=None, error=None):
    client = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)

    def _stream(method, url, **kwargs):
        if error is not None:
            raise error
        resp = AsyncMock()
        resp.raise_for_status = lambda: None
        resp.headers = {}

        async def _chunks():
            yield body

        resp.aiter_bytes = _chunks
        cm = AsyncMock()
        cm.__aenter__ = AsyncMock(return_value=resp)
        cm.__aexit__ = AsyncMock(return_value=False)
        return cm

    client.stream = _stream
    return client


def _install_http(monkeypatch, *, body=None, error=None, recorded=None):
    def factory(**kwargs):
        if recorded is not None:
            recorded.append(kwargs)
        return _stream_client(body=body, error=error)

    monkeypatch.setattr("tools.url_safety.is_safe_url", lambda url: True)
    monkeypatch.setattr("tools.url_safety.create_ssrf_safe_async_client", factory)


@pytest.fixture
def cache_dirs(tmp_path, monkeypatch):
    dirs = {}
    for name in ("IMAGE_CACHE_DIR", "AUDIO_CACHE_DIR", "VIDEO_CACHE_DIR", "DOCUMENT_CACHE_DIR"):
        path = tmp_path / name.lower()
        path.mkdir()
        monkeypatch.setattr(f"gateway.platforms.base.{name}", path)
        dirs[name] = path
    return dirs


@pytest.fixture
def adapter(monkeypatch, cache_dirs):
    import plugins.platforms.dingtalk.adapter as dt

    monkeypatch.setattr(dt, "dingtalk_robot_models", SimpleNamespace(
        RobotMessageFileDownloadRequest=_Model, RobotMessageFileDownloadHeaders=_Model))
    monkeypatch.setattr(dt, "tea_util_models", SimpleNamespace(RuntimeOptions=_Model))
    monkeypatch.setenv("DINGTALK_CLIENT_ID", "app-key-client")
    monkeypatch.setenv("DINGTALK_CLIENT_SECRET", "secret-not-used")
    from gateway.config import PlatformConfig
    from plugins.platforms.dingtalk.adapter import DingTalkAdapter

    built = DingTalkAdapter(PlatformConfig(
        enabled=True, extra={"client_id": "app-key-client", "robot_code": "robot-from-config"}))
    built._get_access_token = AsyncMock(return_value=TOKEN)
    built.handled = []

    async def _handle(event):
        built.handled.append(event)

    built.handle_message = _handle
    return built


def _sdk(adapter, handler):
    adapter._robot_sdk = SimpleNamespace(robot_message_file_download_with_options_async=handler)


def _assert_clean(text):
    for secret in (TOKEN, SIG, CODE, OTHER, "secret-not-used"):
        assert secret not in text


async def _deliver(adapter, message):
    await adapter._on_message(message)
    assert adapter.handled, "message was dropped"
    return adapter.handled[-1]


class TestDownloadRequest:
    @pytest.mark.asyncio
    async def test_request_uses_callback_robot_code_and_download_code(self, adapter, monkeypatch, caplog):
        """Official body is downloadCode + the receiving bot's robotCode; the token is only a header."""
        calls = []
        url = _signed_url()

        async def handler(request, headers, options):
            calls.append((request, headers))
            return SimpleNamespace(body=SimpleNamespace(download_url=url))

        _sdk(adapter, handler)
        _install_http(monkeypatch, body=JPEG)
        caplog.set_level(logging.DEBUG)
        msg = _message(image_content=_Image(CODE), text=SimpleNamespace(content="caption"))
        event = await _deliver(adapter, msg)
        request, headers = calls[0]
        assert request.download_code == CODE
        assert request.robot_code == "robot-from-callback"
        assert not hasattr(request, "pictureDownloadCode")
        assert headers.x_acs_dingtalk_access_token == TOKEN
        assert event.text.splitlines()[0] == "caption"
        assert event.message_type == MessageType.PHOTO
        assert event.media_types == ["image"]
        assert len(event.media_urls) == 1
        assert event.media_urls[0].endswith(".jpg")
        _assert_clean(caplog.text)
        _assert_clean(event.text)

    @pytest.mark.asyncio
    async def test_configured_robot_code_when_callback_omits_it(self, adapter, monkeypatch):
        calls = []

        async def handler(request, headers, options):
            calls.append(request)
            return SimpleNamespace(body={"downloadUrl": _signed_url()})

        _sdk(adapter, handler)
        _install_http(monkeypatch, body=JPEG)
        msg = _message(robot_code="", image_content=_Image(CODE))
        await _deliver(adapter, msg)
        assert calls[0].robot_code == "robot-from-config"
        assert calls[0].robot_code != "app-key-client"

    @pytest.mark.asyncio
    async def test_extension_robot_code_before_client_id(self, adapter, monkeypatch):
        calls = []

        async def handler(request, headers, options):
            calls.append(request)
            return SimpleNamespace(body=SimpleNamespace(download_url=_signed_url("sheet.pdf")))

        _sdk(adapter, handler)
        _install_http(monkeypatch, body=PDF)
        msg = _message(
            message_type="file", robot_code=None, image_content=None,
            extensions={"robotCode": "robot-from-extension", "content": {
                "downloadCode": CODE, "fileName": "report.pdf"}},
        )
        event = await _deliver(adapter, msg)
        assert calls[0].robot_code == "robot-from-extension"
        assert event.message_type == MessageType.DOCUMENT
        assert event.media_types == ["application/pdf"]
        assert event.text.startswith("[文件] report.pdf")
        assert event.media_urls[0].endswith("report.pdf")
        assert ".." not in event.media_urls[0]


class TestImageAdmission:
    @pytest.mark.asyncio
    async def test_cached_image_passes_admission_remote_url_does_not(self, adapter, monkeypatch):
        from gateway.session_envelope import _drop_remote_media
        from gateway.session_ingress_media import capture_native_media

        async def handler(request, headers, options):
            return SimpleNamespace(body=SimpleNamespace(download_url=_signed_url()))

        _sdk(adapter, handler)
        _install_http(monkeypatch, body=JPEG)
        event = await _deliver(adapter, _message(image_content=_Image(CODE), text=SimpleNamespace(content="look")))
        kept = _drop_remote_media(event)
        assert kept.media_urls == event.media_urls
        assert "could not be retrieved" not in kept.text
        refs = capture_native_media(kept.media_urls)
        assert Path(refs[0]["path"]).read_bytes() == JPEG
        remote = _drop_remote_media(type(event)(
            text="look", message_type=MessageType.PHOTO, source=event.source,
            media_urls=[_signed_url()], media_types=["image"]))
        assert remote.media_urls == []
        assert "could not be retrieved" in remote.text

    @pytest.mark.asyncio
    async def test_rich_text_caption_and_picture_download_code_fallback(self, adapter, monkeypatch):
        calls = []

        async def handler(request, headers, options):
            calls.append(request.download_code)
            return SimpleNamespace(body=SimpleNamespace(download_url=_signed_url("rich.png")))

        _sdk(adapter, handler)
        _install_http(monkeypatch, body=JPEG)
        msg = _message(
            message_type="richText", image_content=None, text=None,
            rich_text_content=SimpleNamespace(rich_text_list=[
                {"text": "see this"},
                {"type": "picture", "downloadCode": CODE, "pictureDownloadCode": "do-not-send"},
            ]),
        )
        event = await _deliver(adapter, msg)
        assert calls == [CODE]
        assert event.text == "see this"
        assert event.message_type == MessageType.PHOTO
        assert event.media_urls and event.media_urls[0].endswith(".jpg")

    @pytest.mark.asyncio
    async def test_legacy_rich_text_list_and_picture_download_code_only(self, adapter, monkeypatch):
        calls = []

        async def handler(request, headers, options):
            calls.append(request.download_code)
            return SimpleNamespace(body=SimpleNamespace(download_url=_signed_url()))

        _sdk(adapter, handler)
        _install_http(monkeypatch, body=JPEG)
        msg = _message(
            message_type="richText", image_content=None, rich_text_content=None, text=None,
            rich_text=[{"text": "legacy"}, {"pictureDownloadCode": CODE}],
        )
        event = await _deliver(adapter, msg)
        assert calls == [CODE]
        assert event.text == "legacy"
        assert event.media_types == ["image"]

    @pytest.mark.asyncio
    async def test_picture_extension_without_image_content(self, adapter, monkeypatch):
        async def handler(request, headers, options):
            return SimpleNamespace(body=SimpleNamespace(download_url=_signed_url()))

        _sdk(adapter, handler)
        _install_http(monkeypatch, body=JPEG)
        msg = _message(
            message_type="picture", image_content=None,
            extensions={"content": {"downloadCode": CODE}}, text=SimpleNamespace(content="pic"),
        )
        event = await _deliver(adapter, msg)
        assert event.message_type == MessageType.PHOTO
        assert event.media_urls and event.text == "pic"

    @pytest.mark.asyncio
    async def test_already_cached_path_skips_api(self, adapter, cache_dirs):
        from gateway.platforms.base import cache_image_from_bytes

        path = cache_image_from_bytes(JPEG, ".jpg")
        called = []

        async def handler(request, headers, options):
            called.append(request)

        _sdk(adapter, handler)
        event = await _deliver(adapter, _message(image_content=_Image(path), text=SimpleNamespace(content="kept")))
        assert called == []
        assert event.media_urls == [path]
        assert event.text == "kept"

    @pytest.mark.asyncio
    async def test_pre_resolved_url_is_cached_without_another_api_call(self, adapter, monkeypatch):
        called = []

        async def handler(request, headers, options):
            called.append(request)

        _sdk(adapter, handler)
        _install_http(monkeypatch, body=JPEG)
        event = await _deliver(adapter, _message(image_content=_Image(_signed_url()), text=SimpleNamespace(content="url")))
        assert called == []
        assert event.media_urls and event.media_urls[0].endswith(".jpg")
        assert event.text == "url"


class TestFailures:
    @pytest.mark.asyncio
    async def test_api_500_is_not_treated_as_a_permission_error(self, adapter, caplog):
        async def handler(request, headers, options):
            raise _ApiFailure(request.download_code, headers.x_acs_dingtalk_access_token)

        _sdk(adapter, handler)
        caplog.set_level(logging.DEBUG)
        msg = _message(image_content=_Image(CODE), text=SimpleNamespace(content="keep-me"))
        event = await _deliver(adapter, msg)
        assert event.media_urls == []
        assert event.text.startswith("keep-me")
        assert "HTTP 500" in event.text
        assert "unknownError" in event.text
        assert "does not show a missing permission" in event.text
        assert "HTTP 400" in event.text
        assert "could not be retrieved" not in event.text
        assert msg.image_content.download_code == ""
        _assert_clean(caplog.text)
        _assert_clean(event.text)

    @pytest.mark.asyncio
    async def test_empty_download_url_and_missing_code(self, adapter):
        calls = []

        async def handler(request, headers, options):
            calls.append(request.download_code)
            return SimpleNamespace(body=SimpleNamespace(download_url=""))

        _sdk(adapter, handler)
        event = await _deliver(adapter, _message(
            image_content=_Image(CODE), text=SimpleNamespace(content="still here")))
        assert calls == [CODE]
        assert event.media_urls == []
        assert "no downloadUrl" in event.text
        assert event.text.startswith("still here")

        adapter.handled.clear()
        msg = _message(message_id="m-empty", image_content=_Image("  "), text=SimpleNamespace(content="only text"))
        event = await _deliver(adapter, msg)
        assert calls == [CODE]
        assert event.media_urls == []
        assert event.text == "only text"

    @pytest.mark.asyncio
    async def test_missing_token_or_sdk_keeps_text_and_drops_code(self, adapter):
        async def handler(request, headers, options):
            raise AssertionError("API must not be called")

        _sdk(adapter, handler)
        adapter._get_access_token = AsyncMock(return_value=None)
        event = await _deliver(adapter, _message(image_content=_Image(CODE), text=SimpleNamespace(content="hello")))
        assert "access token unavailable" in event.text
        assert event.media_urls == []
        assert event.text.startswith("hello")

        adapter._robot_sdk = None
        adapter._get_access_token = AsyncMock(return_value=TOKEN)
        adapter.handled.clear()
        event = await _deliver(adapter, _message(
            message_id="m-sdk", image_content=_Image(CODE), text=SimpleNamespace(content="hello")))
        assert "robot SDK is not initialized" in event.text
        assert event.media_urls == []

    @pytest.mark.asyncio
    async def test_empty_robot_code(self, adapter, monkeypatch):
        adapter._robot_code = ""
        adapter._client_id = ""

        async def handler(request, headers, options):
            raise AssertionError("API must not be called")

        _sdk(adapter, handler)
        event = await _deliver(adapter, _message(
            robot_code="", image_content=_Image(CODE), text=SimpleNamespace(content="hello")))
        assert "robotCode is empty" in event.text
        assert event.media_urls == []

    @pytest.mark.asyncio
    async def test_http_timeout_and_status_and_non_image(self, adapter, monkeypatch, caplog):
        async def handler(request, headers, options):
            return SimpleNamespace(body=SimpleNamespace(download_url=_signed_url()))

        _sdk(adapter, handler)
        monkeypatch.setattr(asyncio, "sleep", AsyncMock())
        recorded = []
        _install_http(monkeypatch, error=httpx.TimeoutException("timed out"), recorded=recorded)
        caplog.set_level(logging.DEBUG)
        event = await _deliver(adapter, _message(image_content=_Image(CODE), text=SimpleNamespace(content="cap")))
        assert recorded and recorded[0]["timeout"] == 30.0
        assert "timed out" in event.text
        assert event.media_urls == []
        assert event.text.startswith("cap")
        _assert_clean(caplog.text)

        request = httpx.Request("GET", _signed_url())
        response = httpx.Response(502, request=request)
        error = httpx.HTTPStatusError("download failed", request=request, response=response)
        _install_http(monkeypatch, error=error)
        adapter.handled.clear()
        event = await _deliver(adapter, _message(
            message_id="m-http", image_content=_Image(CODE), text=SimpleNamespace(content="cap")))
        assert "download HTTP 502" in event.text
        assert event.media_urls == []
        _assert_clean(event.text)

        _install_http(monkeypatch, body=b"<!DOCTYPE html><html>nope</html>")
        adapter.handled.clear()
        event = await _deliver(adapter, _message(
            message_id="m-html", image_content=_Image(CODE), text=SimpleNamespace(content="cap")))
        assert "not an image" in event.text
        assert event.media_urls == []

    @pytest.mark.asyncio
    async def test_size_limit(self, adapter, monkeypatch):
        async def handler(request, headers, options):
            return SimpleNamespace(body=SimpleNamespace(download_url=_signed_url()))

        _sdk(adapter, handler)
        monkeypatch.setattr("gateway.platforms.base.get_inbound_media_max_bytes", lambda: 8)
        _install_http(monkeypatch, body=JPEG)
        event = await _deliver(adapter, _message(image_content=_Image(CODE), text=SimpleNamespace(content="cap")))
        assert "inbound size limit" in event.text
        assert event.media_urls == []
        assert CODE not in event.text

    @pytest.mark.asyncio
    async def test_unsafe_url_and_local_path_are_not_admitted(self, adapter, caplog):
        async def handler(request, headers, options):
            return SimpleNamespace(body=SimpleNamespace(
                download_url=f"http://127.0.0.1/meta?Signature={SIG}"))

        _sdk(adapter, handler)
        caplog.set_level(logging.DEBUG)
        event = await _deliver(adapter, _message(image_content=_Image(CODE), text=SimpleNamespace(content="cap")))
        assert event.media_urls == []
        assert "safety check" in event.text
        _assert_clean(caplog.text)

        async def file_url(request, headers, options):
            return SimpleNamespace(body=SimpleNamespace(download_url="file:///etc/passwd"))

        _sdk(adapter, file_url)
        adapter.handled.clear()
        event = await _deliver(adapter, _message(
            message_id="m-file", image_content=_Image(CODE), text=SimpleNamespace(content="cap")))
        assert event.media_urls == []
        assert "root:" not in event.text
        assert "/etc/passwd" not in event.text

        adapter.handled.clear()
        calls = []

        async def unexpected(request, headers, options):
            calls.append(request)

        _sdk(adapter, unexpected)
        event = await _deliver(adapter, _message(
            message_id="m-passwd", image_content=_Image("/etc/passwd"), text=SimpleNamespace(content="cap")))
        assert calls == []
        assert event.media_urls == []
        assert "root:" not in event.text

    @pytest.mark.asyncio
    async def test_cache_path_outside_cache_or_symlink_is_refused(self, adapter, cache_dirs, monkeypatch):
        import plugins.platforms.dingtalk.adapter as dt

        async def handler(request, headers, options):
            return SimpleNamespace(body=SimpleNamespace(download_url=_signed_url()))

        _sdk(adapter, handler)

        async def outside(url, *, kind, filename=""):
            return "/etc/passwd"

        monkeypatch.setattr(dt, "cache_resolved_url", outside)
        event = await _deliver(adapter, _message(image_content=_Image(CODE), text=SimpleNamespace(content="cap")))
        assert event.media_urls == []
        assert "/etc/passwd" not in event.text

        link = cache_dirs["IMAGE_CACHE_DIR"] / "link.jpg"
        target = cache_dirs["IMAGE_CACHE_DIR"] / "real.jpg"
        target.write_bytes(JPEG)
        link.symlink_to(target)

        async def via_link(url, *, kind, filename=""):
            return str(link)

        monkeypatch.setattr(dt, "cache_resolved_url", via_link)
        adapter.handled.clear()
        event = await _deliver(adapter, _message(
            message_id="m-link", image_content=_Image(CODE), text=SimpleNamespace(content="cap")))
        assert event.media_urls == []

    @pytest.mark.asyncio
    async def test_one_image_fails_and_the_other_is_kept(self, adapter, monkeypatch):
        async def handler(request, headers, options):
            if request.download_code == OTHER:
                raise _ApiFailure(request.download_code, TOKEN)
            return SimpleNamespace(body=SimpleNamespace(download_url=_signed_url()))

        _sdk(adapter, handler)
        _install_http(monkeypatch, body=JPEG)
        msg = _message(
            message_type="richText", image_content=None, text=None,
            rich_text_content=SimpleNamespace(rich_text_list=[
                {"text": "two"},
                {"type": "picture", "downloadCode": CODE},
                {"type": "picture", "downloadCode": OTHER},
            ]),
        )
        event = await _deliver(adapter, msg)
        assert event.text.startswith("two")
        assert "HTTP 500" in event.text
        assert len(event.media_urls) == 1
        assert event.media_urls[0].endswith(".jpg")
        assert OTHER not in event.text and CODE not in event.text


class TestSharedFiles:
    @pytest.mark.asyncio
    async def test_pdf_and_png_file_messages(self, adapter, monkeypatch, cache_dirs):
        png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16

        async def handler(request, headers, options):
            name = "pic.png" if request.download_code == OTHER else "notes.pdf"
            return SimpleNamespace(body=SimpleNamespace(download_url=_signed_url(name)))

        def factory_for(body):
            def _factory(**kwargs):
                return _stream_client(body=body)
            return _factory

        _sdk(adapter, handler)
        monkeypatch.setattr("tools.url_safety.is_safe_url", lambda url: True)
        monkeypatch.setattr("tools.url_safety.create_ssrf_safe_async_client", factory_for(PDF))
        pdf = await _deliver(adapter, _message(
            message_type="file", image_content=None, text=None,
            extensions={"content": {"downloadCode": CODE, "fileName": "notes.pdf"}},
        ))
        assert pdf.message_type == MessageType.DOCUMENT
        assert pdf.media_types == ["application/pdf"]
        assert pdf.text == "[文件] notes.pdf"
        pdf_path = Path(pdf.media_urls[0])
        assert pdf_path.name.endswith("notes.pdf")
        assert pdf_path.is_relative_to(cache_dirs["DOCUMENT_CACHE_DIR"])
        assert pdf_path.read_bytes() == PDF

        monkeypatch.setattr("tools.url_safety.create_ssrf_safe_async_client", factory_for(png_bytes))
        adapter.handled.clear()
        png = await _deliver(adapter, _message(
            message_id="m-png", message_type="file", image_content=None, text=None,
            extensions={"content": {"downloadCode": OTHER, "fileName": "../../etc/passwd.png"}},
        ))
        assert png.message_type == MessageType.PHOTO
        assert png.media_types == ["image/png"]
        png_path = Path(png.media_urls[0])
        assert png_path.suffix == ".png"
        assert png_path.is_relative_to(cache_dirs["IMAGE_CACHE_DIR"])
        assert "/etc/passwd" not in str(png_path)
        assert ".." not in png_path.name
        assert png_path.read_bytes() == png_bytes

    def test_audio_message_is_not_fetched_as_an_attachment(self):
        from plugins.platforms.dingtalk.inbound import collect_download_codes, extract_media

        msg = _message(
            message_type="audio", image_content=None,
            extensions={"content": {"downloadCode": CODE, "recognition": "spoken"}},
        )
        msg_type, urls, _mimes = extract_media(msg)
        assert msg_type == MessageType.VOICE
        assert urls == []
        assert collect_download_codes(msg) == []

    def test_voice_rich_text_stays_voice(self):
        from plugins.platforms.dingtalk.inbound import extract_media

        msg = _message(
            message_type="richText", image_content=None, rich_text_content=None,
            rich_text=[{"type": "voice", "downloadCode": "dl_voice_rt"}],
        )
        msg_type, urls, mimes = extract_media(msg)
        assert msg_type == MessageType.VOICE
        assert urls == ["dl_voice_rt"]
        assert mimes == ["audio"]


def test_failure_summary_has_no_secrets_and_does_not_invent_a_permission_cause():
    from plugins.platforms.dingtalk.media import describe_failure, failure_note

    summary = describe_failure(_ApiFailure(CODE, TOKEN), secrets=(CODE, TOKEN, SIG))
    note = failure_note("api", summary)
    assert "HTTP 500" in summary and "unknownError" in summary
    assert "permission" not in summary.lower()
    assert "does not show a missing permission" in note
    _assert_clean(summary)
    _assert_clean(note)

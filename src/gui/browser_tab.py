"""Video browser tab with embedded Chromium, navigation, and Convert overlay."""

import json
import logging
import re
import subprocess
import shutil
from urllib.parse import parse_qs, urlparse

from PySide6.QtCore import QLocale, QObject, QThread, QUrl, Signal, Qt
from PySide6.QtWebEngineCore import (
    QWebEnginePage,
    QWebEngineProfile,
    QWebEngineScript,
)
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from .cookie_manager import CookieManager
from .media_interceptor import MediaInterceptor

logger = logging.getLogger(__name__)


class _FormatProbeWorker(QObject):
    """Probe available download formats for a video via yt-dlp.

    Runs ``yt-dlp --dump-json`` in a background thread.  This only fetches
    metadata (no download) and does NOT trigger bot detection — the API
    calls are identical to what the platform's own player makes.

    **Limitation:** The probe does not pass the user's cookie file, so
    age-restricted or member-only videos may report different availability
    than the authenticated download.  This is acceptable because the probe
    is purely informational (the badge text) and does not affect the actual
    download path which uses ``--cookiefile`` when configured.
    """

    finished = Signal(str, str, str)  # video_id, info_text (HTML), yt_language

    def __init__(self, video_id: str, cookie_file: str = "",
                 po_token: str = "", parent: QObject | None = None):
        super().__init__(parent)
        self._video_id = video_id
        self._cookie_file = cookie_file
        self._po_token = po_token

    def run(self):
        import json

        yt_dlp = shutil.which("yt-dlp")
        if not yt_dlp:
            self.finished.emit(self._video_id, "", "")
            return

        url = f"https://www.youtube.com/watch?v={self._video_id}"
        cmd = [
            yt_dlp, "--dump-json", "--no-download",
            "--no-playlist", "--skip-download", url,
        ]
        # With the browser session's cookies and PO token the probe sees
        # the same formats as the authenticated download (without them
        # The platform reports only a limited fallback set, e.g. 360p H.264).
        if self._cookie_file:
            cmd[1:1] = ["--cookies", self._cookie_file]
        if self._po_token:
            cmd[1:1] = [
                "--extractor-args", f"youtube:po_token=web.gvs+{self._po_token}"
            ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=15,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if result.returncode != 0:
                self.finished.emit(self._video_id, "", "")
                return

            info = json.loads(result.stdout)
            formats = info.get("formats", [])

            # Find best video resolution
            best_height = 0
            best_vcodec = ""
            for f in formats:
                h = f.get("height") or 0
                vcodec = f.get("vcodec", "none")
                if vcodec != "none" and h > best_height:
                    best_height = h
                    best_vcodec = vcodec.split(".")[0]  # e.g. "avc1" → "avc1"

            # Find best audio
            best_abr = 0
            best_acodec = ""
            for f in formats:
                acodec = f.get("acodec", "none")
                abr = f.get("abr") or 0
                if acodec != "none" and abr > best_abr:
                    best_abr = abr
                    best_acodec = acodec.split(".")[0]

            # Duration
            duration = info.get("duration", 0)
            dur_str = f"{duration // 60}:{duration % 60:02d}" if duration else ""

            # LRCLIB lyrics check — yt-dlp only has artist/track for
            # the music service; regular videos need title parsing.
            artist = info.get("artist") or info.get("creator") or ""
            track = info.get("track") or ""
            if not artist or not track:
                # Parse "Artist - Title (extra)" from video title
                video_title = info.get("title") or ""
                if " - " in video_title:
                    parts_split = video_title.split(" - ", 1)
                    artist = artist or parts_split[0].strip()
                    track = track or parts_split[1].strip()
                    # Remove common suffixes like "(Official Video)", "(Live)"
                    track = re.sub(
                        r"\s*[\(\[](official|lyric|music|live|audio|hd|hq|video|visuali|4k|summerbreeze|wacken).*$",
                        "", track, flags=re.IGNORECASE,
                    ).strip()
                elif not artist:
                    artist = info.get("uploader") or info.get("channel") or ""
                    track = track or video_title
            lyrics_status = self._check_lrclib(artist, track)

            # Video platform language metadata
            yt_language = info.get("language") or ""

            codec_names = {
                "avc1": "H.264", "vp9": "VP9", "vp09": "VP9",
                "av01": "AV1", "opus": "Opus", "mp4a": "AAC",
            }

            parts = []
            if best_height:
                vname = codec_names.get(best_vcodec, best_vcodec)
                parts.append(f"\U0001f4f9 {best_height}p {vname}")
            if best_abr:
                aname = codec_names.get(best_acodec, best_acodec)
                parts.append(f"\U0001f50a {best_abr:.0f}k {aname}")
            if dur_str:
                parts.append(f"\u23f1 {dur_str}")
            if yt_language:
                parts.append(f"\U0001f310 {yt_language.upper()}")
            parts.append(lyrics_status)

            self.finished.emit(
                self._video_id, "\n".join(parts), yt_language
            )

        except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError):
            self.finished.emit(self._video_id, "", "")

    @staticmethod
    def _check_lrclib(artist: str, title: str) -> str:
        """Query LRCLIB for lyrics availability (silent, no console output)."""
        if not artist or not title:
            return "\U0001f4dd No metadata"
        try:
            from modules.lrclib_client import _search_by_fields, _search_by_query
            result = _search_by_fields(artist, title)
            if result is None:
                result = _search_by_query(f"{artist} {title}")
            if result is None:
                return "\U0001f4dd No lyrics"
            if result.synced_lyrics:
                return "\u2705 Synced lyrics"
            if result.plain_lyrics:
                return "\u26a0\ufe0f Plain lyrics only"
            return "\U0001f4dd No lyrics"
        except Exception:
            return "\U0001f4dd Lyrics N/A"


# Regex for accepted video URL patterns (used to validate user-pasted URLs).
_VIDEO_URL_RE = re.compile(
    r"^https?://"
    r"(?:(?:www\.|m\.|music\.)?youtube\.com"
    r"|youtu\.be)"
    r"/",
    re.IGNORECASE,
)

# Persistent overlay script — injected via QWebEngineScript (user-script API).
# Only shows the Convert button on video pages (/watch?v=...).
# Uses the ``yt-navigate-finish`` event for SPA re-injection and a
# periodic poll as fallback in case the platform removes the element.
#
# IMPORTANT: The click handler builds a *clean* URL containing only the
# video ID (``?v=...``).  Watch URLs often carry ``&list=...``,
# ``&index=...`` etc. which would cause yt-dlp to download entire
# playlists/mixes instead of the single selected video.
_CONVERT_OVERLAY_JS = r"""
(function() {
    'use strict';

    var BASE_CSS =
        'position:fixed;top:80px;left:16px;z-index:2147483647;' +
        'padding:10px 24px;border-radius:24px;font-size:15px;' +
        'font-weight:bold;user-select:none;line-height:1;' +
        'display:flex;align-items:center;justify-content:center;' +
        'font-family:system-ui,sans-serif;letter-spacing:0.5px;' +
        'pointer-events:auto;';

    var READY_CSS = BASE_CSS +
        'background:linear-gradient(135deg,#e91e63,#c2185b);color:#fff;' +
        'cursor:pointer;' +
        'box-shadow:0 0 16px 5px rgba(233,30,99,0.6),' +
        '0 0 35px 10px rgba(233,30,99,0.4),' +
        '0 0 60px 18px rgba(233,30,99,0.2);' +
        'animation:ultrasinger-glow 2s ease-in-out infinite alternate;' +
        'transition:transform 0.2s,box-shadow 0.2s;';

    var LOADING_CSS = BASE_CSS +
        'background:linear-gradient(135deg,#4a4a4a,#333);color:#bbb;' +
        'cursor:default;' +
        'box-shadow:0 0 8px 2px rgba(0,0,0,0.35);';

    function isVideoPage() {
        return window.location.pathname === '/watch' &&
               window.location.search.indexOf('v=') !== -1;
    }

    function currentVideoId() {
        var params = new URLSearchParams(window.location.search);
        return params.get('v') || '';
    }

    /* The button is clickable when EITHER the PO-token provider is running
       (yt-dlp can then download any video in full quality) OR the audio
       stream of the CURRENT video was captured by the media interceptor.
       The Python side calls window.__ultrasingerSetProviderReady(true) /
       window.__ultrasingerSetQueueReady('<videoId>') accordingly; until
       then a non-clickable "loading" hint is shown. */
    function applyBtnState() {
        var btn = document.getElementById('ultrasinger-convert-btn');
        if (!btn) return;
        var vid = currentVideoId();
        var ready = window.__ultrasingerProviderReady === true ||
                    (vid !== '' && window.__ultrasingerReadyVideoId === vid);
        var state = ready ? '1' : '0';
        if (btn.dataset.ready === state && btn.dataset.videoId === vid) return;
        btn.dataset.ready = state;
        btn.dataset.videoId = vid;
        if (ready) {
            btn.textContent = '\uD83C\uDFA4 Queue';
            btn.style.cssText = READY_CSS;
            btn.title = '';
        } else {
            btn.textContent = '\u23F3 loading...';
            btn.style.cssText = LOADING_CSS;
            btn.title = 'Waiting for the download source - the PO-token ' +
                        'provider is starting, or play the video to capture it';
        }
    }

    window.__ultrasingerSetProviderReady = function(ready) {
        window.__ultrasingerProviderReady = ready;
        applyBtnState();
    };

    window.__ultrasingerSetQueueReady = function(videoId) {
        window.__ultrasingerReadyVideoId = videoId;
        applyBtnState();
    };

    function updateBtn() {
        var existing = document.getElementById('ultrasinger-convert-btn');
        if (isVideoPage()) {
            if (existing) { applyBtnState(); return; }
            if (!document.body) return;

            var btn = document.createElement('div');
            btn.id = 'ultrasinger-convert-btn';

            /* Inject glow keyframes once */
            if (!document.getElementById('ultrasinger-glow-style')) {
                var style = document.createElement('style');
                style.id = 'ultrasinger-glow-style';
                style.textContent =
                    '@keyframes ultrasinger-glow {' +
                    '  0% { box-shadow: 0 0 14px 4px rgba(233,30,99,0.5),' +
                    '       0 0 30px 8px rgba(233,30,99,0.3),' +
                    '       0 0 50px 14px rgba(233,30,99,0.15); }' +
                    '  100% { box-shadow: 0 0 20px 7px rgba(233,30,99,0.7),' +
                    '       0 0 42px 14px rgba(233,30,99,0.45),' +
                    '       0 0 70px 22px rgba(233,30,99,0.25); }' +
                    '}';
                document.head.appendChild(style);
            }

            btn.addEventListener('mouseover', function() {
                if (this.dataset.ready !== '1') return;
                this.style.transform = 'scale(1.08)';
                this.style.animationPlayState = 'paused';
                this.style.boxShadow =
                    '0 0 25px 8px rgba(233,30,99,0.8),' +
                    '0 0 50px 16px rgba(233,30,99,0.5),' +
                    '0 0 80px 25px rgba(233,30,99,0.25)';
            });
            btn.addEventListener('mouseout', function() {
                if (this.dataset.ready !== '1') return;
                this.style.transform = 'scale(1)';
                this.style.animationPlayState = 'running';
                this.style.boxShadow = '';
            });
            btn.addEventListener('click', function() {
                // Ignore clicks while the audio stream is not captured yet
                if (this.dataset.ready !== '1') return;
                // Extract only the video ID — strip &list=, &index= etc.
                // to prevent yt-dlp from downloading an entire playlist.
                var params = new URLSearchParams(window.location.search);
                var videoId = params.get('v');
                var cleanUrl = 'https://www.youtube.com/watch?v=' + videoId;
                window.location.href = 'ultrasinger://convert?url=' +
                    encodeURIComponent(cleanUrl);
            });
            document.body.appendChild(btn);
            applyBtnState();

            /* Quality badge — below the Queue button so it doesn't
               overlap the platform's search suggestion dropdown. */
            if (!document.getElementById('ultrasinger-quality-badge')) {
                var badge = document.createElement('div');
                badge.id = 'ultrasinger-quality-badge';
                badge.style.cssText =
                    'position:fixed;top:140px;left:16px;z-index:2147483647;' +
                    'background:rgba(30,30,30,0.92);color:#aaa;' +
                    'padding:6px 12px;border-radius:10px;font-size:11px;' +
                    'font-family:system-ui,sans-serif;pointer-events:none;' +
                    'backdrop-filter:blur(4px);line-height:1.6;' +
                    'white-space:pre-line;max-width:180px;' +
                    'display:none;';
                document.body.appendChild(badge);
            }
            console.log('[UltraSinger] Convert button injected');
        } else {
            if (existing) {
                existing.remove();
                console.log('[UltraSinger] Convert button removed (not a video page)');
            }
            var badge = document.getElementById('ultrasinger-quality-badge');
            if (badge) badge.remove();
        }
    }

    // Run immediately
    updateBtn();

    // Re-check after SPA navigation
    window.addEventListener('yt-navigate-finish', function() {
        setTimeout(updateBtn, 500);
    });

    // Fallback poll every 3 s
    setInterval(updateBtn, 3000);
})();
"""


def _clean_video_url(url: str) -> str:
    """Strip playlist/mix parameters from a video watch URL.

    Watch URLs often contain ``&list=``, ``&index=``,
    ``&start_radio=`` etc. which cause yt-dlp to download the entire
    playlist instead of a single video.  This function keeps only the
    video ID parameter.
    """
    parsed = urlparse(url)
    params = parse_qs(parsed.query)
    video_id = params.get("v", [""])[0]
    if video_id:
        return f"https://www.youtube.com/watch?v={video_id}"
    # Not a standard watch URL — return as-is (e.g. youtu.be short links)
    return url


class UltraSingerWebPage(QWebEnginePage):
    """Custom web page that intercepts the ultrasinger:// URL scheme."""

    convert_requested = Signal(str)

    def javaScriptConsoleMessage(self, level, message, line, source):
        """Suppress noisy JS console messages from embedded websites."""
        # Only forward errors from our own overlay script
        if source and "userscript" in source.lower():
            logger.debug("JS [%s:%d]: %s", source, line, message)

    def acceptNavigationRequest(self, url: QUrl, nav_type, is_main_frame):
        if url.scheme() == "ultrasinger":
            query = parse_qs(urlparse(url.toString()).query)
            video_url = query.get("url", [""])[0]
            if video_url:
                # Double-safety: clean the URL in Python too
                self.convert_requested.emit(_clean_video_url(video_url))
            return False
        return super().acceptNavigationRequest(url, nav_type, is_main_frame)


class BrowserTab(QWidget):
    """Video browser with navigation bar, cookie management, and Convert overlay."""

    convert_requested = Signal(str, str)  # url, page_title

    def __init__(self, parent=None):
        super().__init__(parent)
        # Persistent browser profile — use a named profile so Qt/Chromium
        # manages its own storage location automatically.  Do NOT override
        # setPersistentStoragePath: Qt's Chromium backend may initialise the
        # cookie store in the constructor, and overriding the path afterwards
        # can silently break cookie persistence across restarts.
        self._profile = QWebEngineProfile("ultrasinger", self)
        self._profile.setPersistentCookiesPolicy(
            QWebEngineProfile.PersistentCookiesPolicy.ForcePersistentCookies
        )

        # NOTE: we intentionally keep the DEFAULT QtWebEngine user agent.
        # Spoofing a plain Chrome UA (to coax the platform's high-quality player)
        # breaks Google sign-in ("This browser or app may not be secure",
        # because a Chrome UA without Chrome client hints is rejected).
        # Full-quality downloads no longer depend on the embedded player at
        # all — they go through yt-dlp + the bgutil PO-token provider — so
        # the browser only needs to log in and pick videos, which the
        # default UA does reliably.

        # Real browsers always send an Accept-Language header; the Qt
        # default is empty, which is another embedded-browser fingerprint.
        if not self._profile.httpAcceptLanguage():
            locale = QLocale().name().replace("_", "-")  # e.g. de-DE
            lang = locale.split("-")[0]
            if lang and lang != "en":
                accept = f"{locale},{lang};q=0.9,en-US;q=0.8,en;q=0.7"
            else:
                accept = "en-US,en;q=0.9"
            self._profile.setHttpAcceptLanguage(accept)
            logger.info("Browser Accept-Language set: %s", accept)

        # Cookie manager
        self.cookie_manager = CookieManager(self._profile, self)
        # Set by MainWindow to the configured cookie-file path; when set,
        # the format probe exports and uses the browser cookies
        self.probe_cookie_file: str = ""
        # True once the PO-token provider is running (enables the Queue
        # button globally; re-asserted to the page on every navigation)
        self._provider_ready: bool = False

        # Media interceptor — passively captures audio stream URLs
        self.media_interceptor = MediaInterceptor(self)
        self._profile.setUrlRequestInterceptor(self.media_interceptor)
        self._probe_yt_language = ""  # Video platform language from yt-dlp metadata

        # Web page + view
        self._page = UltraSingerWebPage(self._profile, self)
        self._page.convert_requested.connect(self._on_convert_with_title)

        self._view = QWebEngineView(self)
        self._view.setPage(self._page)

        # Inject convert overlay via user-script API (Tampermonkey-equivalent).
        self._setup_overlay_script()

        # Build UI
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Toolbar
        toolbar = QWidget()
        toolbar.setObjectName("browserToolbar")
        toolbar.setFixedHeight(44)
        tb_layout = QHBoxLayout(toolbar)
        tb_layout.setContentsMargins(8, 4, 8, 4)
        tb_layout.setSpacing(6)

        style = QApplication.style()

        self._back_btn = QPushButton()
        self._back_btn.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_ArrowBack))
        self._back_btn.setAccessibleName("Back")
        self._back_btn.clicked.connect(self._view.back)
        tb_layout.addWidget(self._back_btn)

        self._fwd_btn = QPushButton()
        self._fwd_btn.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_ArrowForward))
        self._fwd_btn.setAccessibleName("Forward")
        self._fwd_btn.clicked.connect(self._view.forward)
        tb_layout.addWidget(self._fwd_btn)

        self._refresh_btn = QPushButton()
        self._refresh_btn.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_BrowserReload))
        self._refresh_btn.setAccessibleName("Refresh")
        self._refresh_btn.clicked.connect(self._view.reload)
        tb_layout.addWidget(self._refresh_btn)

        self._url_bar = QLineEdit()
        self._url_bar.setObjectName("browserUrlBar")
        self._url_bar.setPlaceholderText(
            "Paste a video URL or navigate below..."
        )
        self._url_bar.returnPressed.connect(self._on_url_bar_submit)
        tb_layout.addWidget(self._url_bar, 1)

        # Cookie status indicator
        self._cookie_dot = QLabel("\u25CF")
        self._cookie_dot.setObjectName("statusDot")
        self._cookie_dot.setStyleSheet("color: #605848; font-size: 14px;")
        self._cookie_dot.setToolTip("Not logged in")
        tb_layout.addWidget(self._cookie_dot)

        layout.addWidget(toolbar)
        layout.addWidget(self._view, 1)

        # Track URL changes — also assigns intercepted streams to video IDs
        self._page.urlChanged.connect(self._on_url_changed)
        self.media_interceptor.audio_captured.connect(
            self._on_audio_captured
        )
        self.media_interceptor.po_token_captured.connect(
            self._on_po_token_captured
        )

        # Track cookie status
        self.cookie_manager.cookies_changed.connect(self._update_cookie_status)
        self._update_cookie_status()

        # Load default video platform
        self._view.setUrl(QUrl("https://www.youtube.com"))

    # ── URL bar ──────────────────────────────────────────────────────────

    def _on_url_changed(self, url: QUrl):
        """Sync URL bar with the browser's current location."""
        self._url_bar.setText(url.toString())

        # Track current video ID for interceptor assignment
        params = parse_qs(urlparse(url.toString()).query)
        self._current_video_id = params.get("v", [""])[0]

        # Re-assert provider readiness on the fresh page (a full page load
        # resets window globals, so the overlay must be told again).
        if self._provider_ready and self._page:
            self._page.runJavaScript(
                "window.__ultrasingerSetProviderReady && "
                "window.__ultrasingerSetProviderReady(true);"
            )

        # Probe available formats when navigating to a video page
        if self._current_video_id:
            self._probe_formats(self._current_video_id)
            # Unlock the Queue button right away when a still-valid stream
            # was captured for this video, or when the session PO token is
            # known (then yt-dlp can download any video in full quality).
            if (
                self.media_interceptor.get_stream(self._current_video_id)
                or self.media_interceptor.get_po_token()
            ):
                self._notify_queue_ready(self._current_video_id)

    def _on_audio_captured(self, stream):
        """When a new audio stream is captured, assign it to the current video."""
        if hasattr(self, "_current_video_id") and self._current_video_id:
            self.media_interceptor.assign_to_video(
                self._current_video_id, stream
            )
            # Unlock the overlay Queue button (it shows a non-clickable
            # "loading" hint until the stream is captured)
            self._notify_queue_ready(self._current_video_id)

    def _on_po_token_captured(self, _pot: str):
        """Session PO token captured — the Queue button can be unlocked."""
        if getattr(self, "_current_video_id", ""):
            self._notify_queue_ready(self._current_video_id)

    def set_provider_ready(self, ready: bool):
        """Enable the Queue button globally once the PO-token provider runs.

        With the provider up, yt-dlp downloads any video in full quality, so
        the button no longer needs a per-video captured stream.  Also
        re-probes the current video so the quality badge reflects the full
        format list instead of the anonymous 360p fallback.
        """
        self._provider_ready = ready
        if self._page:
            self._page.runJavaScript(
                "window.__ultrasingerSetProviderReady && "
                f"window.__ultrasingerSetProviderReady({str(bool(ready)).lower()});"
            )
        if ready and getattr(self, "_current_video_id", ""):
            self._probe_formats(self._current_video_id)

    def _notify_queue_ready(self, video_id: str):
        """Tell the overlay script that the Queue button may be enabled."""
        if self._page:
            self._page.runJavaScript(
                "window.__ultrasingerSetQueueReady && "
                f"window.__ultrasingerSetQueueReady({json.dumps(video_id)});"
            )

    # ── Format probe ──────────────────────────────────────────────────────

    def _probe_formats(self, video_id: str):
        """Query available download formats in the background."""
        # Let any previous probe finish on its own; stale results are
        # discarded by the video_id check in _on_format_probed.
        self._probe_video_id = video_id
        self._probe_yt_language = ""  # reset until probe completes

        # Hide badge while loading
        if self._page:
            self._page.runJavaScript(
                "(function() {"
                "  var b = document.getElementById('ultrasinger-quality-badge');"
                "  if (b) { b.style.display = 'none'; b.textContent = ''; }"
                "})();"
            )

        # Export fresh browser cookies for the probe so it sees the same
        # formats as the authenticated download (badge accuracy).
        cookie_file = ""
        if self.probe_cookie_file and self.cookie_manager.has_video_cookies:
            try:
                self.cookie_manager.export_netscape(self.probe_cookie_file)
                cookie_file = self.probe_cookie_file
            except OSError as e:
                logger.warning("Cookie export for format probe failed: %s", e)

        thread = QThread(self)
        worker = _FormatProbeWorker(
            video_id, cookie_file, self.media_interceptor.get_po_token()
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_format_probed)
        worker.finished.connect(thread.quit)
        # prevent Qt from garbage-collecting thread/worker prematurely
        thread.finished.connect(lambda: self._cleanup_probe(thread, worker))
        thread.start()

    def _cleanup_probe(self, thread, worker):
        """Clean up finished probe thread and worker."""
        thread.deleteLater()
        worker.deleteLater()

    def _on_format_probed(self, video_id: str, info_text: str, yt_language: str):
        """Show the format info badge in the browser overlay."""
        if not info_text or not self._page:
            return
        # Only update if still on the same video
        if self._probe_video_id != video_id:
            return
        # Store the video platform language for use as pipeline hint
        self._probe_yt_language = yt_language or ""

        # Escape for JS string literal (backslash, quotes, control chars)
        safe_text = (
            info_text
            .replace("\\", "\\\\")
            .replace("'", "\\'")
            .replace("\n", "\\n")
            .replace("\r", "\\r")
        )
        self._page.runJavaScript(
            "(function() {"
            "  var b = document.getElementById('ultrasinger-quality-badge');"
            f"  if (b) {{ b.textContent = '{safe_text}'; b.style.display = 'block'; }}"
            "})();"
        )

    def _on_url_bar_submit(self):
        """Navigate to a user-pasted URL after validating it."""
        text = self._url_bar.text().strip()
        if not text:
            return

        # Accept supported video platform URLs
        if _VIDEO_URL_RE.match(text):
            self._view.setUrl(QUrl(text))
        else:
            self._url_bar.setText(self._page.url().toString())
            self._url_bar.setToolTip(
                "Only supported video URLs are allowed "
                "(youtube.com, youtu.be, music.youtube.com)"
            )

    # ── Overlay ──────────────────────────────────────────────────────────

    def _setup_overlay_script(self):
        """Register the Convert-button overlay as a persistent user script.

        Uses ``QWebEngineScript`` (Chromium's user-script API) so the
        overlay is automatically injected on every full page load.  The
        JS itself also listens for the ``yt-navigate-finish`` custom
        event and polls periodically to survive SPA navigation.
        """
        script = QWebEngineScript()
        script.setName("ultrasinger-overlay")
        script.setSourceCode(_CONVERT_OVERLAY_JS)
        script.setInjectionPoint(QWebEngineScript.InjectionPoint.DocumentReady)
        script.setWorldId(QWebEngineScript.ScriptWorldId.MainWorld)
        script.setRunsOnSubFrames(False)
        self._page.scripts().insert(script)
        logger.debug("Overlay user-script registered")

    # ── Cookie status ────────────────────────────────────────────────────

    def _update_cookie_status(self):
        if self.cookie_manager.has_video_cookies:
            self._cookie_dot.setStyleSheet("color: #4caf50; font-size: 14px;")
            self._cookie_dot.setToolTip(
                f"Logged in ({self.cookie_manager.video_cookie_count} cookies)"
            )
        else:
            self._cookie_dot.setStyleSheet("color: #605848; font-size: 14px;")
            self._cookie_dot.setToolTip("Not logged in")

    @property
    def yt_language(self) -> str:
        """Return the video platform language metadata for the current video, if available."""
        return getattr(self, "_probe_yt_language", "")

    def _on_convert_with_title(self, url: str):
        """Extract the page title asynchronously, then emit convert_requested."""
        def _callback(title):
            # Clean up common suffixes from video platform titles
            if title and " - " in title:
                title = title.rsplit(" - ", 1)[0].strip()
            # Strip the platform's notification count prefix like "(1) " or "(23) "
            if title:
                title = re.sub(r"^\(\d+\)\s*", "", title)
            if not title:
                title = url
            self.convert_requested.emit(url, title)

        self._page.runJavaScript("document.title", _callback)

    def shutdown(self):
        """Shut down the web engine cleanly so Chromium can flush cookies.

        Must be called before the widget is destroyed.  Navigating to
        ``about:blank`` triggers Chromium's internal shutdown sequence
        which flushes cookies and persistent storage to disk and
        terminates the renderer subprocess.
        """
        self._view.setPage(None)
        self._page.deleteLater()
        self._page = None
        self._view.deleteLater()
        self._view = None

    def current_url(self) -> str:
        return self._page.url().toString() if self._page else ""

    def navigate(self, url: str):
        if self._view:
            self._view.setUrl(QUrl(url))

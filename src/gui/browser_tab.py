"""YouTube browser tab with embedded Chromium, navigation, and Convert overlay."""

import logging
import re
from urllib.parse import parse_qs, urlparse

from PySide6.QtCore import QUrl, Signal, Qt
from PySide6.QtWebEngineCore import (
    QWebEnginePage,
    QWebEngineProfile,
    QWebEngineScript,
)
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .config import get_browser_profile_path
from .cookie_manager import CookieManager

logger = logging.getLogger(__name__)

# Regex for accepted YouTube URL patterns (used to validate user-pasted URLs).
_YOUTUBE_URL_RE = re.compile(
    r"^https?://"
    r"(?:(?:www\.|m\.|music\.)?youtube\.com"
    r"|youtu\.be)"
    r"/",
    re.IGNORECASE,
)

# Persistent overlay script — injected via QWebEngineScript (user-script API).
# Only shows the Convert button on YouTube video pages (/watch?v=...).
# Uses YouTube's ``yt-navigate-finish`` event for SPA re-injection and a
# periodic poll as fallback in case YouTube removes the element.
#
# IMPORTANT: The click handler builds a *clean* URL containing only the
# video ID (``?v=...``).  YouTube URLs often carry ``&list=...``,
# ``&index=...`` etc. which would cause yt-dlp to download entire
# playlists/mixes instead of the single selected video.
_CONVERT_OVERLAY_JS = r"""
(function() {
    'use strict';

    function isVideoPage() {
        return window.location.pathname === '/watch' &&
               window.location.search.indexOf('v=') !== -1;
    }

    function updateBtn() {
        var existing = document.getElementById('ultrasinger-convert-btn');
        if (isVideoPage()) {
            if (existing) return;
            if (!document.body) return;

            var btn = document.createElement('div');
            btn.id = 'ultrasinger-convert-btn';
            btn.textContent = '\uD83C\uDFA4 Convert';
            btn.style.cssText =
                'position:fixed;bottom:24px;right:24px;z-index:2147483647;' +
                'background:linear-gradient(135deg,#e91e63,#c2185b);color:#fff;' +
                'padding:14px 28px;border-radius:28px;cursor:pointer;font-size:16px;' +
                'font-weight:bold;box-shadow:0 4px 16px rgba(233,30,99,0.4);' +
                'transition:transform 0.2s,box-shadow 0.2s;user-select:none;' +
                'font-family:system-ui,sans-serif;letter-spacing:0.5px;' +
                'pointer-events:auto;';
            btn.addEventListener('mouseover', function() {
                this.style.transform = 'scale(1.08)';
                this.style.boxShadow = '0 6px 24px rgba(233,30,99,0.6)';
            });
            btn.addEventListener('mouseout', function() {
                this.style.transform = 'scale(1)';
                this.style.boxShadow = '0 4px 16px rgba(233,30,99,0.4)';
            });
            btn.addEventListener('click', function() {
                // Extract only the video ID — strip &list=, &index= etc.
                // to prevent yt-dlp from downloading an entire playlist.
                var params = new URLSearchParams(window.location.search);
                var videoId = params.get('v');
                var cleanUrl = 'https://www.youtube.com/watch?v=' + videoId;
                window.location.href = 'ultrasinger://convert?url=' +
                    encodeURIComponent(cleanUrl);
            });
            document.body.appendChild(btn);
            console.log('[UltraSinger] Convert button injected');
        } else {
            if (existing) {
                existing.remove();
                console.log('[UltraSinger] Convert button removed (not a video page)');
            }
        }
    }

    // Run immediately
    updateBtn();

    // Re-check after YouTube SPA navigation
    window.addEventListener('yt-navigate-finish', function() {
        setTimeout(updateBtn, 500);
    });

    // Fallback poll every 3 s
    setInterval(updateBtn, 3000);
})();
"""


def _clean_youtube_url(url: str) -> str:
    """Strip playlist/mix parameters from a YouTube URL.

    YouTube watch URLs often contain ``&list=``, ``&index=``,
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

    def acceptNavigationRequest(self, url: QUrl, nav_type, is_main_frame):
        if url.scheme() == "ultrasinger":
            query = parse_qs(urlparse(url.toString()).query)
            youtube_url = query.get("url", [""])[0]
            if youtube_url:
                # Double-safety: clean the URL in Python too
                self.convert_requested.emit(_clean_youtube_url(youtube_url))
            return False
        return super().acceptNavigationRequest(url, nav_type, is_main_frame)


class BrowserTab(QWidget):
    """YouTube browser with navigation bar, cookie management, and Convert overlay."""

    convert_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        # Persistent browser profile
        profile_path = get_browser_profile_path()
        self._profile = QWebEngineProfile("ultrasinger", self)
        self._profile.setPersistentStoragePath(profile_path)
        self._profile.setCachePath(profile_path + "/cache")

        # Cookie manager
        self.cookie_manager = CookieManager(self._profile, self)

        # Web page + view
        self._page = UltraSingerWebPage(self._profile, self)
        self._page.convert_requested.connect(self.convert_requested.emit)

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

        self._back_btn = QPushButton("\u25C0")
        self._back_btn.setAccessibleName("Back")
        self._back_btn.clicked.connect(self._view.back)
        tb_layout.addWidget(self._back_btn)

        self._fwd_btn = QPushButton("\u25B6")
        self._fwd_btn.setAccessibleName("Forward")
        self._fwd_btn.clicked.connect(self._view.forward)
        tb_layout.addWidget(self._fwd_btn)

        self._refresh_btn = QPushButton("\u21BB")
        self._refresh_btn.setAccessibleName("Refresh")
        self._refresh_btn.clicked.connect(self._view.reload)
        tb_layout.addWidget(self._refresh_btn)

        self._url_bar = QLineEdit()
        self._url_bar.setObjectName("browserUrlBar")
        self._url_bar.setPlaceholderText(
            "Paste a YouTube URL or navigate below..."
        )
        self._url_bar.returnPressed.connect(self._on_url_bar_submit)
        tb_layout.addWidget(self._url_bar, 1)

        # Cookie status indicator
        self._cookie_dot = QLabel("\u25CF")
        self._cookie_dot.setObjectName("statusDot")
        self._cookie_dot.setStyleSheet("color: #616161; font-size: 14px;")
        self._cookie_dot.setToolTip("Not logged in")
        tb_layout.addWidget(self._cookie_dot)

        layout.addWidget(toolbar)
        layout.addWidget(self._view, 1)

        # Track URL changes
        self._page.urlChanged.connect(self._on_url_changed)

        # Track cookie status
        self.cookie_manager.cookies_changed.connect(self._update_cookie_status)
        self._update_cookie_status()

        # Load YouTube
        self._view.setUrl(QUrl("https://www.youtube.com"))

    # ── URL bar ──────────────────────────────────────────────────────────

    def _on_url_changed(self, url: QUrl):
        """Sync URL bar with the browser's current location."""
        self._url_bar.setText(url.toString())

    def _on_url_bar_submit(self):
        """Navigate to a user-pasted URL after validating it."""
        text = self._url_bar.text().strip()
        if not text:
            return

        # Accept youtu.be short links, youtube.com, music.youtube.com, m.youtube.com
        if _YOUTUBE_URL_RE.match(text):
            self._view.setUrl(QUrl(text))
        else:
            self._url_bar.setText(self._page.url().toString())
            self._url_bar.setToolTip(
                "Only YouTube URLs are allowed "
                "(youtube.com, youtu.be, music.youtube.com)"
            )

    # ── Overlay ──────────────────────────────────────────────────────────

    def _setup_overlay_script(self):
        """Register the Convert-button overlay as a persistent user script.

        Uses ``QWebEngineScript`` (Chromium's user-script API) so the
        overlay is automatically injected on every full page load.  The
        JS itself also listens for YouTube's ``yt-navigate-finish`` custom
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
        if self.cookie_manager.has_youtube_cookies:
            self._cookie_dot.setStyleSheet("color: #4caf50; font-size: 14px;")
            self._cookie_dot.setToolTip(
                f"Logged in ({self.cookie_manager.youtube_cookie_count} cookies)"
            )
        else:
            self._cookie_dot.setStyleSheet("color: #616161; font-size: 14px;")
            self._cookie_dot.setToolTip("Not logged in")

    def current_url(self) -> str:
        return self._page.url().toString()

    def navigate(self, url: str):
        self._view.setUrl(QUrl(url))

"""Video browser tab with embedded Chromium, navigation, and Convert overlay."""

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

logger = logging.getLogger(__name__)

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
            btn.textContent = '\uD83C\uDFA4 Queue';
            btn.style.cssText =
                'position:fixed;top:80px;left:16px;z-index:2147483647;' +
                'background:linear-gradient(135deg,#e91e63,#c2185b);color:#fff;' +
                'padding:10px 24px;border-radius:24px;cursor:pointer;font-size:15px;' +
                'font-weight:bold;user-select:none;line-height:1;' +
                'display:flex;align-items:center;justify-content:center;' +
                'font-family:system-ui,sans-serif;letter-spacing:0.5px;' +
                'pointer-events:auto;' +
                'box-shadow:0 0 16px 5px rgba(233,30,99,0.6),' +
                '0 0 35px 10px rgba(233,30,99,0.4),' +
                '0 0 60px 18px rgba(233,30,99,0.2);' +
                'animation:ultrasinger-glow 2s ease-in-out infinite alternate;' +
                'transition:transform 0.2s,box-shadow 0.2s;';

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
                this.style.transform = 'scale(1.08)';
                this.style.animationPlayState = 'paused';
                this.style.boxShadow =
                    '0 0 25px 8px rgba(233,30,99,0.8),' +
                    '0 0 50px 16px rgba(233,30,99,0.5),' +
                    '0 0 80px 25px rgba(233,30,99,0.25)';
            });
            btn.addEventListener('mouseout', function() {
                this.style.transform = 'scale(1)';
                this.style.animationPlayState = 'running';
                this.style.boxShadow = '';
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

        # Cookie manager
        self.cookie_manager = CookieManager(self._profile, self)

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

        # Track URL changes
        self._page.urlChanged.connect(self._on_url_changed)

        # Track cookie status
        self.cookie_manager.cookies_changed.connect(self._update_cookie_status)
        self._update_cookie_status()

        # Load default video platform
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

    def _on_convert_with_title(self, url: str):
        """Extract the page title asynchronously, then emit convert_requested."""
        def _callback(title):
            # Clean up common suffixes from video platform titles
            if title and " - " in title:
                title = title.rsplit(" - ", 1)[0].strip()
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

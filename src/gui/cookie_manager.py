"""Cookie extraction from QWebEngine and Netscape-format export for yt-dlp."""

import logging
from pathlib import Path

from PySide6.QtCore import QObject, Signal
from PySide6.QtNetwork import QNetworkCookie
from PySide6.QtWebEngineCore import QWebEngineCookieStore, QWebEngineProfile

from .cookie_file import write_cookie_file, write_private_copy
from .platform_signin import (
    STORED_ACCOUNT_COOKIE_HOSTS,
    STORED_ACCOUNT_COOKIES,
    is_account_cookie,
    is_signed_in,
)

logger = logging.getLogger(__name__)


def _name(cookie: QNetworkCookie) -> str:
    return bytes(cookie.name()).decode("utf-8", errors="replace")


def _is_account(cookie: QNetworkCookie) -> bool:
    return is_account_cookie(_name(cookie), cookie.domain())


class CookieManager(QObject):
    """Manages browser cookies: collection, export to Netscape format, clearing.

    While signing in to the video platform is not allowed, no account cookie
    stays in the browser: those of an earlier session are deleted at start,
    and new ones as soon as they are set.
    """

    cookies_changed = Signal()
    # The browser deleted an account cookie while signing in is not allowed
    account_cookies_removed = Signal()

    def __init__(self, profile: QWebEngineProfile, parent=None, signin_allowed: bool = False):
        super().__init__(parent)
        self._cookies: list[QNetworkCookie] = []
        self._signin_allowed = signin_allowed
        self._profile = profile
        self._store: QWebEngineCookieStore = profile.cookieStore()
        self._store.cookieAdded.connect(self._on_cookie_added)
        self._store.cookieRemoved.connect(self._on_cookie_removed)
        self._store.loadAllCookies()
        if not signin_allowed:
            # Waits in the store until it has started, before the first page loads
            self.delete_account_cookies()

    @property
    def signin_allowed(self) -> bool:
        return self._signin_allowed

    def set_signin_allowed(self, allowed: bool):
        """Allow account cookies, or delete those in the browser and refuse new ones."""
        self._signin_allowed = allowed
        if not allowed:
            self.delete_account_cookies()

    def delete_account_cookies(self):
        """Delete the account cookies from the browser; anonymous ones stay.

        Those the browser has reported, and those of an earlier session by
        name (the browser does not report its stored cookies; see
        ``STORED_ACCOUNT_COOKIES``). A name not stored is no error.
        """
        tracked = [c for c in self._cookies if _is_account(c)]
        for cookie in tracked:
            self._store.deleteCookie(cookie)
        for host in STORED_ACCOUNT_COOKIE_HOSTS:
            for name in STORED_ACCOUNT_COOKIES:
                cookie = QNetworkCookie(name.encode("ascii"), b"")
                cookie.setDomain(host)
                cookie.setPath("/")
                cookie.setSecure(True)  # an https address, for which also secure cookies count
                self._store.deleteCookie(cookie)
        logger.info("Deleting the account cookies in the browser (signing in is off)")
        if tracked:
            self._cookies = [c for c in self._cookies if not _is_account(c)]
            self.cookies_changed.emit()

    def _on_cookie_added(self, cookie: QNetworkCookie):
        if not self._signin_allowed and _is_account(cookie):
            self._store.deleteCookie(cookie)
            logger.debug("Deleted the account cookie %s of %s", _name(cookie), cookie.domain())
            return
        # Avoid duplicates
        for i, existing in enumerate(self._cookies):
            if (
                existing.name() == cookie.name()
                and existing.domain() == cookie.domain()
                and existing.path() == cookie.path()
            ):
                self._cookies[i] = cookie
                self.cookies_changed.emit()
                return
        self._cookies.append(cookie)
        self.cookies_changed.emit()

    def _on_cookie_removed(self, cookie: QNetworkCookie):
        self._cookies = [
            c
            for c in self._cookies
            if not (
                c.name() == cookie.name()
                and c.domain() == cookie.domain()
                and c.path() == cookie.path()
            )
        ]
        self.cookies_changed.emit()
        if not self._signin_allowed and _is_account(cookie):
            self.account_cookies_removed.emit()

    @property
    def video_cookie_count(self) -> int:
        """Number of cookies for the video platform domain."""
        return sum(
            1
            for c in self._cookies
            if ".youtube.com" in c.domain() or "youtube.com" in c.domain()
        )

    @property
    def has_video_cookies(self) -> bool:
        return self.video_cookie_count > 0

    @property
    def is_signed_in(self) -> bool:
        """Whether the browser holds a signed-in session of the video platform."""
        return is_signed_in((_name(c), c.domain()) for c in self._cookies)

    def export_netscape(self, filepath: str | Path) -> Path:
        """Export all cookies in Netscape format compatible with yt-dlp.

        The file is replaced in one step and readable only by the owner.
        Returns the path to the written file.
        """
        filepath = write_cookie_file(filepath, self._netscape_text())
        logger.info("Exported %d cookies to %s", len(self._cookies), filepath)
        return filepath

    def export_private_copy(self, directory: str | Path) -> Path:
        """Export all cookies to a new private file in ``directory`` for one
        yt-dlp run (see ``cookie_file``); the caller removes it."""
        return write_private_copy(directory, self._netscape_text())

    def _netscape_text(self) -> str:
        """All cookies in Netscape format."""
        lines = ["# Netscape HTTP Cookie File", "# Generated by UltraSinger GUI", ""]
        for cookie in self._cookies:
            domain = cookie.domain()
            if not domain:
                continue
            include_subdomains = "TRUE" if domain.startswith(".") else "FALSE"
            path = cookie.path() or "/"
            secure = "TRUE" if cookie.isSecure() else "FALSE"
            expiry_dt = cookie.expirationDate()
            expiry = (
                str(int(expiry_dt.toSecsSinceEpoch()))
                if expiry_dt.isValid()
                else "0"
            )
            name = _name(cookie)
            value = bytes(cookie.value()).decode("utf-8", errors="replace")
            lines.append(
                f"{domain}\t{include_subdomains}\t{path}\t{secure}\t{expiry}\t{name}\t{value}"
            )
        return "\n".join(lines)

    def clear_all(self):
        """Delete all cookies from the browser profile."""
        self._store.deleteAllCookies()
        self._cookies.clear()
        self.cookies_changed.emit()
        logger.info("All browser cookies cleared")

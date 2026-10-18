"""Signing in to the video platform is off unless the user allows it.

Downloads with an account's cookies can get that account restricted or locked,
and downloads work without them. While signing in is off, the embedded browser
does not open the sign-in pages, the account cookies are deleted from it, and
no cookies go to yt-dlp (conversions and the format probe); an exported cookie
file is deleted. Allowed (after a warning), everything works as before.
"""

import http.cookiejar
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest

pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)
pytest.importorskip("PySide6.QtWebEngineCore", exc_type=ImportError)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QLocale, QUrl, Signal
from PySide6.QtNetwork import QNetworkCookie
from PySide6.QtWebEngineCore import QWebEnginePage, QWebEngineUrlRequestInfo
from PySide6.QtWidgets import QApplication, QLabel, QMessageBox, QWidget

from src.gui import cookie_file
from src.gui import browser_tab as browser_tab_module
from src.gui import main_window as main_window_module
from src.gui import preferences_tab as preferences_module
from src.gui.browser_tab import BrowserTab, UltraSingerWebPage
from src.gui.config import _DEFAULTS, load_config
from src.gui.cookie_manager import CookieManager
from src.gui.media_interceptor import MediaInterceptor
from src.gui.platform_signin import (
    STORED_ACCOUNT_COOKIE_HOSTS,
    STORED_ACCOUNT_COOKIES,
    account_cookie_hosts,
    is_account_cookie,
    is_signed_in,
    is_signin_host,
    is_signin_url,
)
from src.gui.preferences_tab import PreferencesTab
from src.gui.queue_file import read_queue_file
from src.gui.queue_manager import QueueManager
from src.gui.ultrasinger_runner import UltraSingerRunner

_app = QApplication.instance() or QApplication([])

_VIDEO = "https://video.invalid/watch?v=abc"
# Where the platform's "Sign in" button leads
_SIGNIN_BUTTON = ("https://accounts.google.com/ServiceLogin?service=youtube&uilel=3&passive=true"
                  "&continue=https%3A%2F%2Fwww.youtube.com%2Fsignin%3Faction_handle_signin%3Dtrue")
_NAV = QWebEnginePage.NavigationType
_RT = QWebEngineUrlRequestInfo.ResourceType


@pytest.fixture
def settings_dir(tmp_path, monkeypatch):
    """config.json in a temporary directory, and no keyring (nothing reaches the real one)."""
    from src.gui import config as config_module
    from src.gui import secrets as secrets_module

    directory = tmp_path / "settings"
    directory.mkdir()
    monkeypatch.setattr(config_module, "_CONFIG_DIR", directory)
    monkeypatch.setattr(config_module, "_CONFIG_FILE", directory / "config.json")
    monkeypatch.setattr(secrets_module, "_keyring_available", False)
    return directory


def _cookie_file(tmp_path) -> Path:
    path = tmp_path / "cookies.txt"
    path.write_text("# Netscape HTTP Cookie File\n.youtube.com\tTRUE\t/\tTRUE\t0\tSID\tsecret\n",
                    encoding="utf-8")
    return path


# ── The sign-in pages ─────────────────────────────────────────────────────


@pytest.mark.parametrize("host", [
    "accounts.google.com", "ACCOUNTS.GOOGLE.COM.", "accounts.google.de", "accounts.google.co.uk",
    "accounts.google.com.au", "accounts.youtube.com",
])
def test_signin_hosts(host):
    assert is_signin_host(host)


@pytest.mark.parametrize("host", [
    "www.youtube.com", "youtube.com", "m.youtube.com", "music.youtube.com", "consent.youtube.com",
    "www.google.com", "i.ytimg.com", "rr1---sn-abc.googlevideo.com", "accounts.google.com.example.org",
    "myaccounts.google.com", "",
])
def test_the_rest_of_the_site_is_no_signin_host(host):
    assert not is_signin_host(host)


def test_signin_urls():
    assert is_signin_url(_SIGNIN_BUTTON)
    assert is_signin_url("https://accounts.youtube.com/accounts/SetSID?ssdc=1")
    assert is_signin_url("https://accounts.google.com:8443/")
    for url in ("https://www.youtube.com/", "https://www.youtube.com/watch?v=abc",
                "https://www.youtube.com/results?search_query=accounts.google.com",
                "https://consent.youtube.com/m?continue=x", "http://[accounts.google.com/"):
        assert not is_signin_url(url), url


# ── The account cookies ───────────────────────────────────────────────────


@pytest.mark.parametrize("name, domain", [
    ("SID", ".youtube.com"), ("HSID", ".youtube.com"), ("SSID", ".youtube.com"),
    ("APISID", ".youtube.com"), ("SAPISID", ".youtube.com"), ("LOGIN_INFO", ".youtube.com"),
    ("SIDCC", ".youtube.com"), ("__Secure-1PSID", ".youtube.com"), ("__Secure-3PSID", ".youtube.com"),
    ("__Secure-1PAPISID", ".youtube.com"), ("__Secure-3PAPISID", ".youtube.com"),
    ("__Secure-1PSIDTS", ".google.com"), ("__Secure-3PSIDCC", ".google.com"),
    ("SID", ".google.com"), ("SAPISID", ".google.de"), ("SID", "www.youtube.com"),
    ("__Host-1PLSID", "accounts.google.com"), ("ACCOUNT_CHOOSER", "accounts.google.com"),
    ("__Secure-OSID", "myaccount.google.com"),
])
def test_account_cookies(name, domain):
    assert is_account_cookie(name, domain)


@pytest.mark.parametrize("name, domain", [
    ("VISITOR_INFO1_LIVE", ".youtube.com"), ("VISITOR_PRIVACY_METADATA", ".youtube.com"),
    ("YSC", ".youtube.com"), ("PREF", ".youtube.com"), ("SOCS", ".youtube.com"),
    ("CONSENT", ".youtube.com"), ("GPS", ".youtube.com"), ("__Secure-ROLLOUT_TOKEN", ".youtube.com"),
    ("NID", ".google.com"), ("AEC", ".google.com"), ("SOCS", ".google.com"),
])
def test_anonymous_cookies_stay(name, domain):
    assert not is_account_cookie(name, domain)


def test_cookies_of_other_sites_are_left_alone():
    for domain in (".example.org", "notyoutube.com", ".youtube.com.example.org"):
        assert not is_account_cookie("SID", domain), domain


def _yt_dlp_signed_in(names):
    """yt-dlp's own notion of an authenticated session (offline)."""
    yt_dlp = pytest.importorskip("yt_dlp")
    from yt_dlp.extractor.youtube import YoutubeIE

    ydl = yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True})
    for name in names:
        ydl.cookiejar.set_cookie(http.cookiejar.Cookie(
            0, name, "value", None, False, ".youtube.com", True, True, "/", True, True, None,
            False, None, None, {}))
    extractor = YoutubeIE(ydl)
    if not hasattr(type(extractor), "_has_auth_cookies"):
        pytest.skip("yt-dlp no longer has _has_auth_cookies")
    return extractor._has_auth_cookies


@pytest.mark.parametrize("names", [
    ("LOGIN_INFO", "SAPISID"), ("LOGIN_INFO", "__Secure-3PAPISID"), ("LOGIN_INFO", "__Secure-1PAPISID"),
    ("LOGIN_INFO",), ("SAPISID", "SID", "HSID"), ("VISITOR_INFO1_LIVE", "YSC", "PREF", "SOCS"), (),
])
def test_signed_in_as_yt_dlp_sees_it(names):
    ours = is_signed_in((name, ".youtube.com") for name in names)
    assert ours == _yt_dlp_signed_in(names)
    if ours:
        # Without the cookies that make it signed in, yt-dlp sees no account
        anonymous = [name for name in names if not is_account_cookie(name, ".youtube.com")]
        assert not _yt_dlp_signed_in(anonymous)


def test_the_names_deleted_at_start_are_account_cookies():
    for name in STORED_ACCOUNT_COOKIES:
        assert any(is_account_cookie(name, host) for host in STORED_ACCOUNT_COOKIE_HOSTS), name
    for name in ("SID", "HSID", "SSID", "APISID", "SAPISID", "LOGIN_INFO",
                 "__Secure-1PAPISID", "__Secure-3PAPISID"):
        assert name in STORED_ACCOUNT_COOKIES
    assert "www.youtube.com" in STORED_ACCOUNT_COOKIE_HOSTS
    assert "accounts.google.com" in STORED_ACCOUNT_COOKIE_HOSTS


@pytest.mark.parametrize("territory, host", [
    ("DE", "accounts.google.de"), ("de", "accounts.google.de"), ("AT", "accounts.google.at"),
    ("GB", "accounts.google.co.uk"), ("JP", "accounts.google.co.jp"), ("BR", "accounts.google.com.br"),
])
def test_the_country_domain_of_the_locale_is_covered(territory, host):
    """The provider's country domain of the locale; for one with a stored account."""
    hosts = account_cookie_hosts(territory)
    assert host in hosts
    assert hosts[:len(STORED_ACCOUNT_COOKIE_HOSTS)] == STORED_ACCOUNT_COOKIE_HOSTS
    assert all(is_signin_host(h) for h in hosts if h.startswith("accounts."))
    assert is_account_cookie("SID", "." + host.removeprefix("accounts."))


@pytest.mark.parametrize("territory", ["", "C", "DEU", "1A", " "])
def test_no_country_domain_without_a_country(territory):
    assert account_cookie_hosts(territory) == STORED_ACCOUNT_COOKIE_HOSTS


def test_the_locale_gives_the_country():
    from src.gui.cookie_manager import _locale_territory

    previous = QLocale()
    try:
        for name, code in (("de_DE", "DE"), ("en_GB", "GB"), ("pt_BR", "BR"), ("C", "")):
            QLocale.setDefault(QLocale(name))
            assert _locale_territory() == code, name
    finally:
        QLocale.setDefault(previous)


# ── The cookie manager ────────────────────────────────────────────────────


def _qcookie(name, domain):
    cookie = QNetworkCookie(name.encode(), b"value")
    cookie.setDomain(domain)
    cookie.setPath("/")
    return cookie


def _manager(signin_allowed, territory=""):
    """A cookie manager on a stand-in cookie store (no browser profile), in a
    locale of the country ``territory``."""
    profile = MagicMock()
    with patch("src.gui.cookie_manager._locale_territory", return_value=territory):
        manager = CookieManager(profile, None, signin_allowed=signin_allowed)
    return manager, profile.cookieStore.return_value


def _deleted(store):
    return [(bytes(call.args[0].name()).decode(), call.args[0].domain())
            for call in store.deleteCookie.call_args_list]


def _tracked(manager):
    return sorted((bytes(c.name()).decode(), c.domain()) for c in manager._cookies)


def test_start_while_off_deletes_the_stored_account_cookies():
    """The browser does not report its stored cookies, so they are deleted by name;
    an https address, so that secure cookies are deleted too."""
    _unused, store = _manager(signin_allowed=False)
    expected = {(name, host) for host in STORED_ACCOUNT_COOKIE_HOSTS for name in STORED_ACCOUNT_COOKIES}
    assert set(_deleted(store)) == expected
    for call in store.deleteCookie.call_args_list:
        assert call.args[0].isSecure() and call.args[0].path() == "/"
    assert not any(name in ("VISITOR_INFO1_LIVE", "YSC", "PREF", "SOCS", "NID")
                   for name, _host in _deleted(store))


def test_start_while_off_also_deletes_them_on_the_country_domain_of_the_locale():
    _unused, store = _manager(signin_allowed=False, territory="DE")
    deleted = set(_deleted(store))
    for name in STORED_ACCOUNT_COOKIES:
        assert (name, "accounts.google.de") in deleted, name
        assert (name, "accounts.google.com") in deleted, name


def test_start_while_allowed_deletes_nothing():
    _unused, store = _manager(signin_allowed=True, territory="DE")
    store.deleteCookie.assert_not_called()


def test_account_cookies_set_while_off_are_deleted():
    manager, store = _manager(signin_allowed=False)
    store.deleteCookie.reset_mock()
    changes = []
    manager.cookies_changed.connect(lambda: changes.append(1))
    for name, domain in (("SID", ".youtube.com"), ("LOGIN_INFO", ".youtube.com"),
                         ("VISITOR_INFO1_LIVE", ".youtube.com"), ("PREF", ".youtube.com")):
        manager._on_cookie_added(_qcookie(name, domain))
    assert _deleted(store) == [("SID", ".youtube.com"), ("LOGIN_INFO", ".youtube.com")]
    assert _tracked(manager) == [("PREF", ".youtube.com"), ("VISITOR_INFO1_LIVE", ".youtube.com")]
    assert not manager.is_signed_in


def test_switching_off_deletes_the_account_cookies_and_keeps_the_anonymous_ones():
    manager, store = _manager(signin_allowed=True)
    for name in ("SID", "LOGIN_INFO", "SAPISID", "VISITOR_INFO1_LIVE", "YSC"):
        manager._on_cookie_added(_qcookie(name, ".youtube.com"))
    assert manager.is_signed_in
    manager.set_signin_allowed(False)
    deleted = _deleted(store)
    for name in ("SID", "LOGIN_INFO", "SAPISID"):
        assert (name, ".youtube.com") in deleted
    assert ("VISITOR_INFO1_LIVE", ".youtube.com") not in deleted
    assert _tracked(manager) == [("VISITOR_INFO1_LIVE", ".youtube.com"), ("YSC", ".youtube.com")]
    assert not manager.is_signed_in and not manager.signin_allowed


def test_a_deleted_account_cookie_is_reported_while_off():
    """The browser tab then loads the page again, signed out."""
    manager, _store = _manager(signin_allowed=False)
    reported = []
    manager.account_cookies_removed.connect(lambda: reported.append(1))
    manager._on_cookie_removed(_qcookie("YSC", ".youtube.com"))
    assert reported == []
    manager._on_cookie_removed(_qcookie("SID", ".youtube.com"))
    assert reported == [1]
    manager.set_signin_allowed(True)
    manager._on_cookie_removed(_qcookie("SIDCC", ".youtube.com"))  # e.g. a rotated cookie
    assert reported == [1]


# ── The browser does not open the sign-in pages ───────────────────────────


def _page(signin_allowed):
    """The page without a browser engine; the base class accepts what it is asked."""
    page = UltraSingerWebPage.__new__(UltraSingerWebPage)
    page.signin_allowed = signin_allowed
    return page


@pytest.mark.parametrize("nav_type", [_NAV.NavigationTypeLinkClicked, _NAV.NavigationTypeTyped,
                                      _NAV.NavigationTypeRedirect, _NAV.NavigationTypeOther])
def test_page_does_not_open_signin_pages_while_off(nav_type):
    page = _page(signin_allowed=False)
    with patch.object(QWebEnginePage, "acceptNavigationRequest", return_value=True) as base, \
            patch.object(UltraSingerWebPage, "signin_blocked", MagicMock()) as blocked:
        assert UltraSingerWebPage.acceptNavigationRequest(page, QUrl(_SIGNIN_BUTTON), nav_type, True) is False
        # In a frame: not opened either, without the notice
        assert UltraSingerWebPage.acceptNavigationRequest(
            page, QUrl("https://accounts.youtube.com/RotateCookiesPage"), nav_type, False) is False
        # The rest of the site works signed out
        for url in ("https://www.youtube.com/watch?v=abc", "https://www.youtube.com/results?search_query=x",
                    "https://consent.youtube.com/m?continue=x"):
            assert UltraSingerWebPage.acceptNavigationRequest(page, QUrl(url), nav_type, True) is True
    assert blocked.emit.call_args_list == [call("accounts.google.com")]
    assert base.call_count == 3


def test_page_opens_signin_pages_when_allowed():
    page = _page(signin_allowed=True)
    with patch.object(QWebEnginePage, "acceptNavigationRequest", return_value=True) as base, \
            patch.object(UltraSingerWebPage, "signin_blocked", MagicMock()) as blocked:
        assert UltraSingerWebPage.acceptNavigationRequest(
            page, QUrl(_SIGNIN_BUTTON), _NAV.NavigationTypeLinkClicked, True) is True
    base.assert_called_once()
    blocked.emit.assert_not_called()


def _request(url, resource_type):
    info = MagicMock()
    info.requestUrl.return_value = QUrl(url)
    info.resourceType.return_value = resource_type
    return info


@pytest.mark.parametrize("resource_type, notice", [(_RT.ResourceTypeMainFrame, True),
                                                   (_RT.ResourceTypeSubFrame, False)])
def test_interceptor_blocks_signin_pages_while_off(resource_type, notice):
    """The second line, behind the page's own check."""
    interceptor = MediaInterceptor(signin_allowed=False)
    blocked = []
    interceptor.signin_blocked.connect(blocked.append)
    info = _request(_SIGNIN_BUTTON, resource_type)
    interceptor.interceptRequest(info)
    info.block.assert_called_once_with(True)
    assert blocked == (["accounts.google.com"] if notice else [])


def test_interceptor_lets_the_rest_through():
    interceptor = MediaInterceptor(signin_allowed=False)
    for url, resource_type in (("https://www.youtube.com/watch?v=abc", _RT.ResourceTypeMainFrame),
                               ("https://accounts.google.com/x.png", _RT.ResourceTypeImage)):
        info = _request(url, resource_type)
        interceptor.interceptRequest(info)
        info.block.assert_not_called()
    interceptor.signin_allowed = True
    info = _request(_SIGNIN_BUTTON, _RT.ResourceTypeMainFrame)
    interceptor.interceptRequest(info)
    info.block.assert_not_called()


# ── The browser tab ───────────────────────────────────────────────────────


def _browser(url="https://www.youtube.com/watch?v=abc", allowed=True, armed=False):
    """The attributes of BrowserTab that the sign-in methods use."""
    return SimpleNamespace(
        _signin_allowed=allowed,
        _page=SimpleNamespace(signin_allowed=allowed, url=lambda: QUrl(url)),
        _view=MagicMock(),
        media_interceptor=SimpleNamespace(signin_allowed=allowed),
        cookie_manager=MagicMock(is_signed_in=False, video_cookie_count=0),
        _signout_reload_armed=armed,
        _signout_reload_timer=MagicMock(),
        _signin_notice=MagicMock(),
        _signin_notice_timer=MagicMock(),
        _cookie_dot=QLabel(),
        _update_cookie_status=MagicMock(),
        settings_requested=MagicMock(),
    )


def test_switching_off_applies_to_the_browser_at_once():
    tab = _browser(allowed=True)
    BrowserTab.set_signin_allowed(tab, False)
    assert tab._signin_allowed is False
    assert tab._page.signin_allowed is False and tab.media_interceptor.signin_allowed is False
    tab.cookie_manager.set_signin_allowed.assert_called_once_with(False)
    # The page is loaded again once the browser has deleted the account cookies
    assert tab._signout_reload_armed
    tab._signout_reload_timer.start.assert_not_called()
    tab._update_cookie_status.assert_called_once()


def test_switching_off_leaves_an_open_signin_page():
    tab = _browser(url=_SIGNIN_BUTTON, allowed=True)
    BrowserTab.set_signin_allowed(tab, False)
    tab._signout_reload_timer.start.assert_called_once()
    BrowserTab._reload_signed_out(tab)
    tab._view.setUrl.assert_called_once_with(QUrl(browser_tab_module._HOME_URL))


def test_the_page_is_loaded_again_once_after_the_deletion():
    """Whatever might set such cookies again, the page does not reload over and over."""
    tab = _browser(allowed=False, armed=True)
    BrowserTab._on_account_cookies_removed(tab)
    tab._signout_reload_timer.start.assert_called_once()
    BrowserTab._reload_signed_out(tab)
    tab._view.setUrl.assert_called_once_with(QUrl("https://www.youtube.com/watch?v=abc"))
    BrowserTab._on_account_cookies_removed(tab)
    tab._signout_reload_timer.start.assert_called_once()


def test_switching_on_hides_the_notice():
    tab = _browser(allowed=False, armed=True)
    BrowserTab.set_signin_allowed(tab, True)
    tab.cookie_manager.set_signin_allowed.assert_called_once_with(True)
    assert tab._page.signin_allowed and tab.media_interceptor.signin_allowed
    assert not tab._signout_reload_armed
    tab._signout_reload_timer.stop.assert_called_once()
    tab._signin_notice.hide.assert_called_once()


def test_a_blocked_signin_page_shows_a_notice_with_the_way_to_the_setting():
    tab = _browser(allowed=False)
    BrowserTab._on_signin_blocked(tab, "accounts.google.com")
    tab._signin_notice.show.assert_called_once()
    tab._signin_notice_timer.start.assert_called_once()
    BrowserTab._on_signin_notice_link(tab, "settings")
    tab.settings_requested.emit.assert_called_once()
    tab._signin_notice.hide.assert_called_once()
    assert "Settings" in browser_tab_module._SIGNIN_NOTICE
    allowed = _browser(allowed=True)
    BrowserTab._on_signin_blocked(allowed, "accounts.google.com")  # a queued signal from before
    allowed._signin_notice.show.assert_not_called()


def test_the_status_dot():
    tab = _browser(allowed=False)
    BrowserTab._update_cookie_status(tab)
    assert "signing in is off" in tab._cookie_dot.toolTip()
    tab = _browser(allowed=True)
    tab.cookie_manager.is_signed_in = True
    tab.cookie_manager.video_cookie_count = 12
    BrowserTab._update_cookie_status(tab)
    assert tab._cookie_dot.toolTip() == "Signed in (12 cookies)"


class _ProbeHost:
    """The attributes of BrowserTab that _probe_formats uses."""

    def __init__(self, tmp_path, signin_allowed):
        self._page = None
        self.signin_allowed = signin_allowed
        self.probe_config = {}  # the saved settings: the default download limits
        self.probe_cookie_file = str(tmp_path / "cookies.txt")
        self.cookie_manager = MagicMock(has_video_cookies=True)
        self.cookie_manager.export_private_copy.side_effect = (
            lambda directory: cookie_file.write_private_copy(directory, "cookies"))
        self.media_interceptor = MagicMock(get_po_token=MagicMock(return_value=""))
        self._probes = []

    def _on_format_probed(self, *args):
        pass

    def _on_badge_ready(self, *args):
        pass


@pytest.mark.parametrize("signin_allowed", [False, True])
def test_the_format_probe_gets_cookies_only_while_signing_in_is_allowed(tmp_path, signin_allowed):
    host = _ProbeHost(tmp_path, signin_allowed)
    commands = []

    def fake_popen(cmd, **kwargs):
        commands.append(list(cmd))
        return MagicMock(returncode=1, communicate=MagicMock(return_value=("", "")))

    with patch("src.gui.browser_tab.shutil.which", return_value="yt-dlp"), \
            patch("src.gui.browser_tab.subprocess.Popen", side_effect=fake_popen):
        BrowserTab._probe_formats(host, "vid")
        for thread, _worker in host._probes:
            thread.join(5)
    assert ("--cookies" in commands[0]) is signin_allowed
    assert "--ignore-config" in commands[0]
    assert host.cookie_manager.export_private_copy.called is signin_allowed
    assert list(tmp_path.iterdir()) == []  # a copy is removed after the probe


def test_the_probe_reads_no_yt_dlp_configuration_of_the_user_but_its_option_file(tmp_path, monkeypatch):
    """A configuration file of the user could add cookies (e.g. --cookies-from-browser)
    although signing in is off; the private option file with the PO token is still read.
    yt-dlp's own option parser reads the probe's command line."""
    yt_dlp_options = pytest.importorskip("yt_dlp.options")
    from src.gui import config as config_module

    monkeypatch.setattr(config_module, "_CONFIG_DIR", tmp_path / "settings")
    for name in ("HOME", "USERPROFILE", "APPDATA", "XDG_CONFIG_HOME"):  # no real one is read
        monkeypatch.setenv(name, str(tmp_path / "nobody"))
    home = tmp_path / "home"
    home.mkdir()
    (home / "yt-dlp.conf").write_text("--cookies users-cookies.txt\n", encoding="utf-8")
    host = _ProbeHost(tmp_path, signin_allowed=False)
    host.media_interceptor.get_po_token.return_value = "pot-value"
    parsed = {}

    def fake_popen(cmd, **kwargs):
        args = cmd[1:] + ["-P", f"home:{home}"]  # the user's configuration as the home one
        parsed["probe"] = yt_dlp_options.parseOpts(args, ignore_config_files=False)[1]
        without = [arg for arg in args if arg != "--ignore-config"]
        parsed["without the option"] = yt_dlp_options.parseOpts(without, ignore_config_files=False)[1]
        return MagicMock(returncode=1, communicate=MagicMock(return_value=("", "")))

    with patch("src.gui.browser_tab.shutil.which", return_value="yt-dlp"), \
            patch("src.gui.browser_tab.subprocess.Popen", side_effect=fake_popen):
        BrowserTab._probe_formats(host, "vid")
        for thread, _worker in host._probes:
            thread.join(5)
    assert parsed["probe"].cookiefile is None
    assert parsed["probe"].extractor_args == {"youtube": {"po_token": ["web.gvs+pot-value"]}}
    assert parsed["without the option"].cookiefile == "users-cookies.txt"


# ── No cookies go to the conversions ──────────────────────────────────────


def _args(cookie, **settings):
    return UltraSingerRunner().build_args({**_DEFAULTS, "cookie_file": str(cookie), **settings}, _VIDEO)


def test_the_runner_passes_the_cookie_file_only_while_signing_in_is_allowed(tmp_path):
    cookie = _cookie_file(tmp_path)
    assert "--cookiefile" not in _args(cookie, allow_platform_signin=False)
    args = _args(cookie, allow_platform_signin=True)
    assert args[args.index("--cookiefile") + 1] == str(cookie)
    # Without the setting, or with a value that is no switch: no cookies
    assert "--cookiefile" not in _args(cookie, allow_platform_signin="true")
    config = {key: value for key, value in _DEFAULTS.items() if key != "allow_platform_signin"}
    assert "--cookiefile" not in UltraSingerRunner().build_args(
        {**config, "cookie_file": str(cookie)}, _VIDEO)
    assert _DEFAULTS["allow_platform_signin"] is False


def _queue(tmp_path, cookie, allowed):
    manager = QueueManager()
    manager._runner.start = MagicMock()
    lines = []
    manager.line_output.connect(lines.append)
    manager.set_global_config({**_DEFAULTS, "output_folder": str(tmp_path), "cookie_file": str(cookie),
                               "allow_platform_signin": allowed})
    return manager, lines


def _queue_run(tmp_path, cookie, allowed):
    manager, lines = _queue(tmp_path, cookie, allowed)
    manager.add_item(_VIDEO, "url", "Song Title")
    manager.start_all()
    return manager._runner.start.call_args.args[0], lines


def test_the_queue_passes_no_cookies_while_off(tmp_path):
    cookie = _cookie_file(tmp_path)
    args, lines = _queue_run(tmp_path, cookie, allowed=False)
    assert "--cookiefile" not in args
    assert not any("Cookie file" in line for line in lines)


def test_the_queue_passes_and_names_the_cookie_file_when_allowed(tmp_path):
    cookie = _cookie_file(tmp_path)
    args, lines = _queue_run(tmp_path, cookie, allowed=True)
    assert args[args.index("--cookiefile") + 1] == str(cookie)
    assert f"[Queue] Cookie file: {cookie}" in lines


def test_switching_off_reaches_the_next_song_of_a_running_queue(tmp_path):
    manager, _lines = _queue(tmp_path, _cookie_file(tmp_path), allowed=True)
    manager.add_item(_VIDEO, "url", "First Song")
    manager.add_item("https://video.invalid/watch?v=def", "url", "Second Song")
    manager.start_all()
    assert "--cookiefile" in manager._runner.start.call_args.args[0]
    manager.set_signin_allowed(False)  # while the first song converts
    manager._on_item_finished(0)  # the queue goes on with the second song
    assert manager.is_running and manager._runner.start.call_count == 2
    second = manager._runner.start.call_args.args[0]
    assert second[second.index("-i") + 1] == "https://video.invalid/watch?v=def"
    assert "--cookiefile" not in second


@pytest.mark.parametrize("downloaded", [True, False])
def test_switching_off_during_the_browser_pre_download_reaches_that_song(tmp_path, downloaded):
    """The song's settings were taken before its pre-download; the conversion starts after it.
    The cookie file may still exist (its deletion can fail, e.g. while another program reads it)."""
    manager, _lines = _queue(tmp_path, _cookie_file(tmp_path), allowed=True)
    manager.add_item(_VIDEO, "url", "Song Title")
    item = manager._items[0]
    audio = tmp_path / "audio.m4a"
    if downloaded:
        audio.write_bytes(b"audio")
    manager._predownload = (item, {**manager._global_config}, str(audio))
    manager.set_signin_allowed(False)  # while the browser downloads the audio
    manager._on_predownload_finished(downloaded, str(audio), "" if downloaded else "failed")
    args = manager._runner.start.call_args.args[0]
    assert args[args.index("-i") + 1] == (str(audio) if downloaded else _VIDEO)
    assert "--cookiefile" not in args


def test_a_queue_file_cannot_allow_signing_in_for_a_song(tmp_path):
    path = tmp_path / "queue.json"
    path.write_text(json.dumps({"version": 1, "items": [{
        "input_type": "url", "input_source": _VIDEO, "title": "Song Title",
        "settings_overrides": {"allow_platform_signin": True, "cookie_file": "C:/cookies.txt"},
    }]}), encoding="utf-8")
    for imported in (False, True):
        items, _problems = read_queue_file(path, imported=imported)
        assert items[0].settings_overrides == {}


# ── The setting ───────────────────────────────────────────────────────────


def _answer_warning(monkeypatch, confirm):
    """Stand in for the warning dialog: record it and press one of its buttons."""
    shown = []

    def fake_exec(box):
        shown.append(SimpleNamespace(
            title=box.windowTitle(), text=box.text(), details=box.informativeText(),
            buttons={box.buttonRole(button): button.text() for button in box.buttons()},
            default=box.defaultButton().text(), icon=box.icon()))
        wanted = (QMessageBox.ButtonRole.AcceptRole if confirm else QMessageBox.ButtonRole.RejectRole)
        next(button for button in box.buttons() if box.buttonRole(button) == wanted).click()
        return 0

    monkeypatch.setattr(QMessageBox, "exec", fake_exec)
    return shown


def _saved(settings_dir):
    return json.loads((settings_dir / "config.json").read_text(encoding="utf-8"))


def test_signing_in_is_off_by_default():
    tab = PreferencesTab(dict(_DEFAULTS))
    assert not tab._signin_switch.isChecked() and not tab.signin_allowed
    assert tab.collect_all()["allow_platform_signin"] is False
    assert "throwaway account" in tab._signin_switch.toolTip()
    assert "mainly age-restricted or members-only" in tab._signin_switch.toolTip()
    assert not tab._export_btn.isEnabled()


def test_switching_on_asks_first_and_cancel_keeps_it_off(settings_dir, monkeypatch):
    shown = _answer_warning(monkeypatch, confirm=False)
    tab = PreferencesTab(dict(_DEFAULTS), MagicMock(is_signed_in=False, video_cookie_count=0))
    changes = []
    tab.signin_allowed_changed.connect(changes.append)
    tab._signin_switch.click()
    assert len(shown) == 1
    warning = shown[0]
    assert warning.icon == QMessageBox.Icon.Warning
    assert warning.buttons[QMessageBox.ButtonRole.AcceptRole] == "I understand the risk, allow signing in"
    assert QMessageBox.ButtonRole.RejectRole in warning.buttons
    assert warning.default != warning.buttons[QMessageBox.ButtonRole.AcceptRole]  # Enter cancels
    for words in ("restrict or lock accounts", "automated downloads", "throwaway account",
                  "never your main account", "Downloads work without signing in",
                  "Mainly age-restricted or members-only"):
        assert words in warning.details, words
    assert not tab._signin_switch.isChecked() and not tab.signin_allowed
    assert changes == []
    assert not (settings_dir / "config.json").exists()
    assert not tab._export_btn.isEnabled()


def test_confirming_allows_signing_in_and_saves_it(settings_dir, monkeypatch):
    _answer_warning(monkeypatch, confirm=True)
    config = dict(_DEFAULTS)
    tab = PreferencesTab(config, MagicMock(is_signed_in=False, video_cookie_count=0))
    changes = []
    tab.signin_allowed_changed.connect(changes.append)
    tab._signin_switch.click()
    assert tab._signin_switch.isChecked() and tab.signin_allowed
    assert changes == [True]
    assert config["allow_platform_signin"] is True
    assert _saved(settings_dir)["allow_platform_signin"] is True
    assert load_config()["allow_platform_signin"] is True
    assert tab._export_btn.isEnabled()
    # A new settings page (next start) shows it on
    assert PreferencesTab(load_config())._signin_switch.isChecked()


def test_switching_off_needs_no_confirmation_and_is_saved(settings_dir, monkeypatch):
    shown = _answer_warning(monkeypatch, confirm=True)
    tab = PreferencesTab({**_DEFAULTS, "allow_platform_signin": True},
                         MagicMock(is_signed_in=True, video_cookie_count=3))
    changes = []
    tab.signin_allowed_changed.connect(changes.append)
    tab._signin_switch.click()
    assert shown == []
    assert not tab._signin_switch.isChecked() and changes == [False]
    assert _saved(settings_dir)["allow_platform_signin"] is False
    assert tab._cookie_status.text() == "Signing in is off: downloads get no cookies."


def test_the_texts_name_no_brand():
    texts = [preferences_module._SIGNIN_TOOLTIP, preferences_module.SIGNIN_WARNING_TEXT,
             preferences_module.SIGNIN_WARNING_DETAILS, preferences_module.SIGNIN_CONFIRM_BUTTON,
             browser_tab_module._SIGNIN_NOTICE, main_window_module.MainWindow._BOT_CHECK,
             main_window_module.MainWindow._BOT_CHECK_ADVICE, main_window_module.MainWindow._ACCOUNT_RISK]
    for text in texts:
        assert "youtube" not in text.lower() and "google" not in text.lower(), text


# ── The main window ───────────────────────────────────────────────────────


class _FakeBrowserTab(QWidget):
    convert_requested = Signal(str, str)
    settings_requested = Signal()

    def __init__(self, signin_allowed=False):
        super().__init__()
        self.created_allowed = signin_allowed
        self.cookie_manager = MagicMock(has_video_cookies=True, is_signed_in=False, video_cookie_count=0)
        self.media_interceptor = MagicMock()
        self.probe_cookie_file = ""
        self.set_signin_allowed = MagicMock()
        self.shutdown = MagicMock()
        self._view = MagicMock()


@pytest.fixture
def open_window(settings_dir, monkeypatch):
    """Opens a main window on the temporary settings directory."""
    from src.gui.widgets import llm_provider_list

    monkeypatch.setattr(llm_provider_list, "_shutting_down", False)

    def open_(cookie, allowed):
        config = {**_DEFAULTS, "cookie_file": str(cookie), "allow_platform_signin": allowed}
        with patch.object(main_window_module, "BrowserTab", _FakeBrowserTab), \
                patch.object(main_window_module.MainWindow, "_start_potoken_provider", lambda self: None), \
                patch.object(main_window_module.MainWindow, "_start_ytdlp_update_check",
                             lambda self: None):
            window = main_window_module.MainWindow(config)
        window._queue_save_timer.stop()
        window._queue_saving = False
        return window

    return open_


def _console(window):
    return window._queue_tab._log.toPlainText()


def test_start_while_off_deletes_the_exported_cookie_file(open_window, tmp_path):
    cookie = _cookie_file(tmp_path)
    window = open_window(cookie, allowed=False)
    assert not cookie.exists()
    assert window._browser_tab.created_allowed is False


def test_start_while_allowed_keeps_the_cookie_file(open_window, tmp_path):
    cookie = _cookie_file(tmp_path)
    window = open_window(cookie, allowed=True)
    assert cookie.exists()
    assert window._browser_tab.created_allowed is True


@pytest.mark.parametrize("allowed", [False, True])
def test_cookies_are_exported_only_while_signing_in_is_allowed(open_window, tmp_path, allowed):
    window = open_window(tmp_path / "cookies.txt", allowed)
    window._auto_export_cookies()
    assert window._browser_tab.cookie_manager.export_netscape.called is allowed


def test_switching_off_in_the_settings_applies_at_once(open_window, tmp_path):
    cookie = _cookie_file(tmp_path)
    window = open_window(cookie, allowed=True)
    window._queue_mgr.set_global_config(window._config)
    window._settings_tab._signin_switch.click()  # off: no question
    window._browser_tab.set_signin_allowed.assert_called_once_with(False)
    assert window._config["allow_platform_signin"] is False
    assert window._queue_mgr._global_config["allow_platform_signin"] is False
    assert not cookie.exists()
    assert "Signing in to the video platform is off" in _console(window)


def test_switching_on_in_the_settings(open_window, tmp_path, monkeypatch):
    _answer_warning(monkeypatch, confirm=True)
    window = open_window(tmp_path / "cookies.txt", allowed=False)
    window._settings_tab._signin_switch.click()
    window._browser_tab.set_signin_allowed.assert_called_once_with(True)
    assert window._config["allow_platform_signin"] is True
    assert "throwaway account" in _console(window)


_BOT_LINE = "ERROR: abc: Sign in to confirm you're not a bot. This helps protect our community."


def test_bot_check_while_off_advises_waiting_instead_of_signing_in(open_window, tmp_path):
    cookie = _cookie_file(tmp_path)
    window = open_window(cookie, allowed=False)
    cookie.write_text("a file the user put there", encoding="utf-8")
    window._sidebar.set_active = MagicMock()
    with patch.object(QMessageBox, "information") as information:
        window._check_for_bot_detection(_BOT_LINE)
        window._check_for_bot_detection(_BOT_LINE)  # once per queue run
    information.assert_called_once()
    message = information.call_args.args[2]
    for words in ("not a bot", "usually happens after many downloads from one connection", "Wait a while",
                  "another network", "Signing in is not needed for downloads"):
        assert words in message, words
    assert "throwaway" not in message
    window._browser_tab.cookie_manager.clear_all.assert_not_called()
    window._sidebar.set_active.assert_not_called()
    assert cookie.exists()
    assert "[Queue] WARNING: The video platform asked to confirm that this is not a bot" in _console(window)


def test_bot_check_while_allowed_resets_the_cookies_and_warns_about_the_account(open_window, tmp_path):
    cookie = _cookie_file(tmp_path)
    window = open_window(cookie, allowed=True)
    window._sidebar.set_active = MagicMock()
    with patch.object(QMessageBox, "information") as information:
        window._check_for_bot_detection(_BOT_LINE)
    message = information.call_args.args[2]
    for words in ("not a bot", "usually happens after many downloads from one connection", "were deleted",
                  "restricted or locked", "throwaway account", "never your main account",
                  "Signing in is not needed for downloads"):
        assert words in message, words
    window._browser_tab.cookie_manager.clear_all.assert_called_once()
    assert not cookie.exists()
    window._sidebar.set_active.assert_called_once_with(window._TAB_VIDEO)


def test_the_notice_link_shows_the_setting(open_window, tmp_path):
    window = open_window(tmp_path / "cookies.txt", allowed=False)
    window._settings_tab.show_signin_switch = MagicMock()
    window._browser_tab.settings_requested.emit()
    assert window._stack.currentIndex() == window._TAB_SETTINGS
    window._settings_tab.show_signin_switch.assert_called_once()

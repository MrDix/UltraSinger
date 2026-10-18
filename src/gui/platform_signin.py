"""Signing in to the video platform: its sign-in pages and account cookies.

Downloads with an account's cookies can get that account restricted or locked
by the platform, and downloads work without them (only age-restricted or
members-only videos need an account). So signing in is off unless the user
allows it in the settings ("Allow signing in to the video platform"). While it
is off:

- the embedded browser does not open the sign-in pages (``is_signin_host``);
- the account cookies are deleted from the browser (``is_account_cookie``);
  anonymous cookies, such as the visitor and consent cookies, stay;
- no cookies go to yt-dlp: the conversions and the format probe get no cookie
  file, and an exported one is deleted.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from urllib.parse import urlsplit

# The hosts of the sign-in pages. The platform's "Sign in" button opens the
# account provider's sign-in page (/ServiceLogin on the first host below, or
# on one of its country domains); after a sign-in, the platform's own
# accounts host sets the account cookies on the platform's domain.
_SIGNIN_HOST = re.compile(r"accounts\.(?:google\.(?:com?\.)?[a-z]{2,3}|youtube\.com)")

# The domains whose account cookies are deleted: the account provider's (with
# its country domains) and the platform's, with their subdomains
_ACCOUNT_DOMAIN = re.compile(r"(?:.+\.)?(?:google\.(?:com?\.)?[a-z]{2,3}|youtube\.com)")

# The platform's domain, where yt-dlp reads the account cookies
_PLATFORM_DOMAIN = re.compile(r"(?:.+\.)?youtube\.com")

# The session and security cookies of a signed-in account
_ACCOUNT_COOKIES = frozenset({
    "SID", "HSID", "SSID", "APISID", "SAPISID", "SIDCC", "LSID", "OSID", "LOGIN_INFO",
})
# Their secure variants: __Secure-1PSID, __Secure-3PSIDTS, __Secure-1PSIDCC,
# __Secure-3PAPISID, __Host-1PLSID, __Secure-OSID and the like
_ACCOUNT_COOKIE_PREFIX = re.compile(r"__(?:Secure|Host)-(?:[13]P(?:SID|APISID|LSID)|OSID)")

# yt-dlp counts a session as signed in when LOGIN_INFO and one of these are set
_SIGNED_IN_KEYS = ("SAPISID", "__Secure-1PAPISID", "__Secure-3PAPISID")

# The browser does not report the cookies it has stored (Qt's loadAllCookies
# reports none), so the account cookies of an earlier session are deleted by
# name: each of these names on each of these hosts (a deletion for a host also
# deletes the cookies of its parent domain, e.g. of .youtube.com for
# www.youtube.com). Cookies set while the GUI runs are reported, and deleted
# by is_account_cookie.
STORED_ACCOUNT_COOKIES = tuple(sorted(_ACCOUNT_COOKIES)) + (
    "__Secure-1PSID", "__Secure-3PSID", "__Secure-1PAPISID", "__Secure-3PAPISID",
    "__Secure-1PSIDTS", "__Secure-3PSIDTS", "__Secure-1PSIDCC", "__Secure-3PSIDCC",
    "__Secure-1PSIDRTS", "__Secure-3PSIDRTS", "__Secure-OSID",
    # Those of the sign-in host itself
    "__Host-1PLSID", "__Host-3PLSID", "__Host-GAPS", "ACCOUNT_CHOOSER", "SMSV", "LSOLH",
)
STORED_ACCOUNT_COOKIE_HOSTS = (
    "www.youtube.com", "accounts.youtube.com", "accounts.google.com", "myaccount.google.com",
)


def _normalized(host: str) -> str:
    """A host or cookie domain in lower case, without the leading and trailing dots."""
    return host.strip().lower().strip(".")


def is_signin_host(host: str) -> bool:
    """True for a host of the sign-in pages."""
    host = _normalized(host)
    return host.startswith("accounts.") and _SIGNIN_HOST.fullmatch(host) is not None


def is_signin_url(url: str) -> bool:
    """True for an address on a host of the sign-in pages."""
    try:
        host = urlsplit(url).hostname or ""
    except ValueError:  # e.g. an invalid port
        return False
    return is_signin_host(host)


def is_account_cookie(name: str, domain: str) -> bool:
    """True for a cookie of a signed-in account on the platform's or the account
    provider's domains; False for the anonymous ones (visitor, consent, preferences)."""
    domain = _normalized(domain)
    if _ACCOUNT_DOMAIN.fullmatch(domain) is None:
        return False
    if is_signin_host(domain):
        return True  # what the sign-in hosts keep belongs to a sign-in
    return name in _ACCOUNT_COOKIES or _ACCOUNT_COOKIE_PREFIX.match(name) is not None


def is_signed_in(cookies: Iterable[tuple[str, str]]) -> bool:
    """Whether ``cookies`` ((name, domain) pairs) hold a signed-in session of the
    platform, as yt-dlp sees it: LOGIN_INFO and one of the SAPISID cookies on
    the platform's domain."""
    names = {name for name, domain in cookies
             if _PLATFORM_DOMAIN.fullmatch(_normalized(domain)) is not None}
    return "LOGIN_INFO" in names and any(key in names for key in _SIGNED_IN_KEYS)

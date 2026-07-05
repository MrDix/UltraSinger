"""Corporate-proxy friendliness for the whole pipeline.

Two independent concerns, both fail-open:

1. ``ensure_localhost_no_proxy`` — when any proxy environment variable is
   set, guarantee that loopback traffic bypasses the proxy. Python's
   ``urllib``/``requests`` route *everything* through ``http_proxy`` unless
   ``no_proxy`` says otherwise, which would send calls to the local
   PO-token provider (http://127.0.0.1:4416) to the corporate proxy and
   break it.

2. ``enable_system_certificates`` — corporate proxies commonly intercept
   TLS with their own root CA. Python's default ``certifi`` bundle does not
   contain it, so every HTTPS call fails. ``truststore`` makes Python use
   the operating system's certificate store (where such CAs are installed),
   fixing this transparently. Optional dependency; silently skipped when
   unavailable.

Proxy configuration itself stays standard: set ``HTTP_PROXY`` /
``HTTPS_PROXY`` / ``NO_PROXY`` (any case) in the environment, or configure
it in the GUI's "Network Proxy" settings card, which applies exactly these
variables via ``apply_proxy_config``.
"""

from __future__ import annotations

import os

_LOOPBACK_ENTRIES = ("localhost", "127.0.0.1", "::1")
_PROXY_VARS = ("http_proxy", "https_proxy", "all_proxy")


def _proxy_is_configured(env) -> bool:
    lowered = {k.lower() for k in env.keys()}
    return any(v in lowered for v in _PROXY_VARS)


def ensure_localhost_no_proxy(env=None):
    """Append loopback hosts to ``no_proxy`` when a proxy is configured.

    Mutates and returns *env* (defaults to ``os.environ``). Existing
    ``no_proxy`` entries are preserved; loopback entries are only appended
    when missing. Both ``no_proxy`` and ``NO_PROXY`` are written because
    different libraries read different casings.
    """
    if env is None:
        env = os.environ
    if not _proxy_is_configured(env):
        return env

    current = env.get("no_proxy") or env.get("NO_PROXY") or ""
    entries = [e.strip() for e in current.split(",") if e.strip()]
    lowered = {e.lower() for e in entries}
    for host in _LOOPBACK_ENTRIES:
        if host not in lowered:
            entries.append(host)
    merged = ",".join(entries)
    env["no_proxy"] = merged
    env["NO_PROXY"] = merged
    return env


def enable_system_certificates() -> bool:
    """Make Python's SSL use the OS certificate store (via ``truststore``).

    Lets HTTPS work behind TLS-intercepting corporate proxies whose root CA
    is installed in the Windows/macOS/Linux system store. Must run before
    the first TLS connection is made. Returns True when active; silently
    fail-open otherwise (missing package, unsupported platform, ...).
    """
    try:
        import truststore

        truststore.inject_into_ssl()
        return True
    except Exception:  # noqa: BLE001 — never let cert setup break startup
        return False


def setup_proxy_environment(env=None) -> bool:
    """Convenience entry-point: loopback bypass + system certificates."""
    ensure_localhost_no_proxy(env)
    return enable_system_certificates()


def apply_proxy_config(config, env=None):
    """Apply the GUI's ``proxy_mode`` / ``proxy_url`` / ``proxy_no_proxy``
    settings to *env* (defaults to ``os.environ``). Mutates and returns *env*.

    Three modes (``config["proxy_mode"]``, case-insensitive; ``"system"``
    is the default for missing/unrecognised values):

    - ``"manual"`` — sets ``http_proxy`` / ``https_proxy`` (both casings)
      from ``proxy_url`` (only when non-empty), merges ``proxy_no_proxy``
      into ``no_proxy``, and appends the loopback bypass via
      :func:`ensure_localhost_no_proxy` so the local PO-token provider
      keeps working.
    - ``"none"`` — strips every ``*_proxy`` variable (any casing) from
      *env*, overriding whatever the OS/environment already configured.
    - ``"system"`` — leaves proxy variables untouched; only the loopback
      bypass is (re-)applied, matching :func:`setup_proxy_environment`.
    """
    if env is None:
        env = os.environ

    mode = str(config.get("proxy_mode", "system") or "system").strip().lower()

    if mode == "none":
        for key in list(env.keys()):
            if key.lower().endswith("_proxy"):
                del env[key]
        return env

    if mode == "manual":
        url = str(config.get("proxy_url", "") or "").strip()
        if url:
            env["http_proxy"] = url
            env["https_proxy"] = url
            env["HTTP_PROXY"] = url
            env["HTTPS_PROXY"] = url

        no_proxy_field = str(config.get("proxy_no_proxy", "") or "").strip()
        if no_proxy_field:
            existing = env.get("no_proxy") or env.get("NO_PROXY") or ""
            entries = [e.strip() for e in existing.split(",") if e.strip()]
            lowered = {e.lower() for e in entries}
            for entry in (e.strip() for e in no_proxy_field.split(",")):
                if entry and entry.lower() not in lowered:
                    entries.append(entry)
                    lowered.add(entry.lower())
            merged_no_proxy = ",".join(entries)
            env["no_proxy"] = merged_no_proxy
            env["NO_PROXY"] = merged_no_proxy

        ensure_localhost_no_proxy(env)
        return env

    # "system" (default) and any unrecognised mode: leave proxy vars alone.
    ensure_localhost_no_proxy(env)
    return env

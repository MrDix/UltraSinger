"""Tests for the corporate-proxy setup helpers."""

import unittest
from unittest.mock import patch

from src.modules.proxy_setup import (
    enable_system_certificates,
    ensure_localhost_no_proxy,
)


class TestEnsureLocalhostNoProxy(unittest.TestCase):
    def test_no_proxy_vars_leaves_env_untouched(self):
        env = {"PATH": "x"}
        ensure_localhost_no_proxy(env)
        self.assertNotIn("no_proxy", env)
        self.assertNotIn("NO_PROXY", env)

    def test_appends_loopback_when_proxy_set(self):
        env = {"http_proxy": "http://proxy.corp:3128"}
        ensure_localhost_no_proxy(env)
        for host in ("localhost", "127.0.0.1", "::1"):
            self.assertIn(host, env["no_proxy"])
        self.assertEqual(env["no_proxy"], env["NO_PROXY"])

    def test_uppercase_proxy_var_detected(self):
        env = {"HTTPS_PROXY": "http://proxy.corp:3128"}
        ensure_localhost_no_proxy(env)
        self.assertIn("127.0.0.1", env["NO_PROXY"])

    def test_existing_entries_preserved_no_duplicates(self):
        env = {
            "http_proxy": "http://proxy.corp:3128",
            "no_proxy": ".corp.example,localhost",
        }
        ensure_localhost_no_proxy(env)
        entries = env["no_proxy"].split(",")
        self.assertIn(".corp.example", entries)
        self.assertEqual(entries.count("localhost"), 1)
        self.assertIn("127.0.0.1", entries)

    def test_uppercase_no_proxy_respected(self):
        env = {"http_proxy": "p", "NO_PROXY": "127.0.0.1,localhost,::1"}
        ensure_localhost_no_proxy(env)
        self.assertEqual(
            env["NO_PROXY"].split(","), ["127.0.0.1", "localhost", "::1"]
        )

    def test_returns_same_mapping(self):
        env = {"http_proxy": "p"}
        self.assertIs(ensure_localhost_no_proxy(env), env)


class TestEnableSystemCertificates(unittest.TestCase):
    def test_returns_true_when_truststore_works(self):
        # truststore is a hard dependency now; on supported platforms this
        # should simply succeed.
        self.assertTrue(enable_system_certificates())

    def test_fail_open_when_inject_raises(self):
        import truststore

        with patch.object(truststore, "inject_into_ssl",
                          side_effect=RuntimeError("boom")):
            self.assertFalse(enable_system_certificates())


if __name__ == "__main__":
    unittest.main()

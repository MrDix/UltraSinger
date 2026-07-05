"""Tests for the corporate-proxy setup helpers."""

import unittest
from unittest.mock import patch

from src.modules.proxy_setup import (
    apply_proxy_config,
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


class TestApplyProxyConfig(unittest.TestCase):
    def test_manual_sets_proxy_vars_and_loopback(self):
        env = {}
        config = {"proxy_mode": "manual", "proxy_url": "http://proxy.corp:3128"}
        apply_proxy_config(config, env)
        for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
            self.assertEqual(env[key], "http://proxy.corp:3128")
        for host in ("localhost", "127.0.0.1", "::1"):
            self.assertIn(host, env["no_proxy"])

    def test_manual_merges_no_proxy_field(self):
        env = {}
        config = {
            "proxy_mode": "manual",
            "proxy_url": "http://proxy.corp:3128",
            "proxy_no_proxy": ".company.com,10.0.0.0/8",
        }
        apply_proxy_config(config, env)
        entries = env["no_proxy"].split(",")
        self.assertIn(".company.com", entries)
        self.assertIn("10.0.0.0/8", entries)
        self.assertIn("localhost", entries)
        self.assertIn("127.0.0.1", entries)
        self.assertEqual(env["no_proxy"], env["NO_PROXY"])

    def test_manual_without_url_skips_proxy_vars(self):
        env = {}
        config = {"proxy_mode": "manual", "proxy_url": ""}
        apply_proxy_config(config, env)
        for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
            self.assertNotIn(key, env)

    def test_none_removes_all_proxy_vars(self):
        env = {
            "http_proxy": "http://old:3128",
            "HTTPS_PROXY": "http://old:3128",
            "no_proxy": "localhost",
            "NO_PROXY": "localhost",
            "all_proxy": "socks5://old:1080",
            "PATH": "/usr/bin",
        }
        apply_proxy_config({"proxy_mode": "none"}, env)
        for key in ("http_proxy", "HTTPS_PROXY", "no_proxy", "NO_PROXY", "all_proxy"):
            self.assertNotIn(key, env)
        self.assertEqual(env["PATH"], "/usr/bin")

    def test_system_mode_leaves_proxy_vars_untouched(self):
        env = {"http_proxy": "http://existing:3128"}
        apply_proxy_config({"proxy_mode": "system"}, env)
        self.assertEqual(env["http_proxy"], "http://existing:3128")
        self.assertIn("127.0.0.1", env["no_proxy"])

    def test_default_mode_is_system(self):
        env = {"http_proxy": "http://existing:3128"}
        apply_proxy_config({}, env)
        self.assertEqual(env["http_proxy"], "http://existing:3128")
        self.assertIn("localhost", env["no_proxy"])

    def test_mode_is_case_insensitive(self):
        env = {}
        apply_proxy_config(
            {"proxy_mode": "MANUAL", "proxy_url": "http://proxy.corp:3128"}, env
        )
        self.assertEqual(env["http_proxy"], "http://proxy.corp:3128")

        env2 = {"http_proxy": "http://old:3128"}
        apply_proxy_config({"proxy_mode": "NONE"}, env2)
        self.assertNotIn("http_proxy", env2)

    def test_returns_same_mapping(self):
        env = {}
        self.assertIs(apply_proxy_config({"proxy_mode": "system"}, env), env)


if __name__ == "__main__":
    unittest.main()

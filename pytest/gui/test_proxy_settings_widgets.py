"""Round-trip tests for the "Network Proxy" settings card in PreferencesTab.

Mirrors ``pytest/gui/test_remote_stt_retry_widgets.py`` in structure: covers
widget defaults, the Manual-mode enable/disable coupling, and that
``collect_all()`` surfaces the three proxy config keys consumed by
``apply_proxy_config`` (``src/modules/proxy_setup.py``).
"""

import os
import unittest

import pytest

# Skip the whole module (instead of failing collection) when PySide6 or its
# native Qt libraries are unavailable, e.g. on a runner without the gui extra.
pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from src.gui.config import _DEFAULTS
from src.gui.preferences_tab import PreferencesTab

_app = QApplication.instance() or QApplication([])


class TestNetworkProxyCardDefaults(unittest.TestCase):
    """Widgets should reflect config values (or defaults when absent)."""

    def test_card_exists_with_default_mode(self):
        tab = PreferencesTab(dict(_DEFAULTS))
        self.assertEqual(tab._proxy_mode.currentData(), _DEFAULTS["proxy_mode"])
        self.assertEqual(tab._proxy_url.text(), _DEFAULTS["proxy_url"])
        self.assertEqual(tab._proxy_no_proxy.text(), _DEFAULTS["proxy_no_proxy"])

    def test_widgets_reflect_stored_config(self):
        config = dict(_DEFAULTS)
        config.update({
            "proxy_mode": "manual",
            "proxy_url": "http://proxy.company.com:3128",
            "proxy_no_proxy": ".company.com,10.0.0.0/8",
        })
        tab = PreferencesTab(config)
        self.assertEqual(tab._proxy_mode.currentData(), "manual")
        self.assertEqual(tab._proxy_url.text(), "http://proxy.company.com:3128")
        self.assertEqual(tab._proxy_no_proxy.text(), ".company.com,10.0.0.0/8")

    def test_unknown_stored_mode_falls_back_to_first_item(self):
        config = dict(_DEFAULTS)
        config["proxy_mode"] = "something_unrecognised"
        tab = PreferencesTab(config)
        self.assertEqual(tab._proxy_mode.currentIndex(), 0)


class TestManualEnableCoupling(unittest.TestCase):
    """proxy_url / proxy_no_proxy are only editable in 'Manual' mode."""

    def test_manual_fields_disabled_by_default_system_mode(self):
        tab = PreferencesTab(dict(_DEFAULTS))
        self.assertFalse(tab._proxy_url.isEnabled())
        self.assertFalse(tab._proxy_no_proxy.isEnabled())

    def test_manual_fields_enabled_when_stored_mode_is_manual(self):
        config = dict(_DEFAULTS)
        config["proxy_mode"] = "manual"
        tab = PreferencesTab(config)
        self.assertTrue(tab._proxy_url.isEnabled())
        self.assertTrue(tab._proxy_no_proxy.isEnabled())

    def test_switching_mode_live_updates_enabled_state(self):
        tab = PreferencesTab(dict(_DEFAULTS))
        self.assertFalse(tab._proxy_url.isEnabled())

        manual_index = tab._proxy_mode.findData("manual")
        tab._proxy_mode.setCurrentIndex(manual_index)
        self.assertTrue(tab._proxy_url.isEnabled())
        self.assertTrue(tab._proxy_no_proxy.isEnabled())

        none_index = tab._proxy_mode.findData("none")
        tab._proxy_mode.setCurrentIndex(none_index)
        self.assertFalse(tab._proxy_url.isEnabled())
        self.assertFalse(tab._proxy_no_proxy.isEnabled())


class TestCollectAll(unittest.TestCase):
    """collect_all() must surface all three proxy config fields."""

    def test_collect_all_includes_proxy_fields(self):
        tab = PreferencesTab(dict(_DEFAULTS))
        manual_index = tab._proxy_mode.findData("manual")
        tab._proxy_mode.setCurrentIndex(manual_index)
        tab._proxy_url.setText("http://proxy.company.com:3128")
        tab._proxy_no_proxy.setText(".company.com,10.0.0.0/8")

        config = tab.collect_all()

        self.assertEqual(config["proxy_mode"], "manual")
        self.assertEqual(config["proxy_url"], "http://proxy.company.com:3128")
        self.assertEqual(config["proxy_no_proxy"], ".company.com,10.0.0.0/8")

    def test_collect_all_defaults_round_trip(self):
        tab = PreferencesTab(dict(_DEFAULTS))
        config = tab.collect_all()

        self.assertEqual(config["proxy_mode"], _DEFAULTS["proxy_mode"])
        self.assertEqual(config["proxy_url"], _DEFAULTS["proxy_url"])
        self.assertEqual(config["proxy_no_proxy"], _DEFAULTS["proxy_no_proxy"])


if __name__ == "__main__":
    unittest.main()

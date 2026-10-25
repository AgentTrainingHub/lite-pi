"""Linux ownership must work in images without procps installed."""

import subprocess
import unittest
from unittest.mock import patch

from lite_pi.process_identity import process_alive, process_identity


class ProcessIdentityContracts(unittest.TestCase):
    def test_live_legacy_owner_keeps_its_slot_until_exit(self):
        with patch("lite_pi.process_identity.sys.platform", "linux"):
            with patch("lite_pi.process_identity.os.kill"):
                self.assertTrue(process_alive(42, "Mon Oct 19 08:00:00 2026"))
            with patch("lite_pi.process_identity.os.kill", side_effect=ProcessLookupError):
                self.assertFalse(process_alive(42, "Mon Oct 19 08:00:00 2026"))

    def test_unreadable_identity_keeps_the_slot_and_ps_timeout_is_reported_as_os_error(self):
        with patch("lite_pi.process_identity.os.kill"):
            with patch("lite_pi.process_identity.process_identity", side_effect=PermissionError):
                self.assertTrue(process_alive(42, "linux:boot-a:12345"))
        with patch("lite_pi.process_identity.sys.platform", "darwin"):
            with patch(
                "lite_pi.process_identity.subprocess.run",
                side_effect=subprocess.TimeoutExpired("ps", 5),
            ):
                with self.assertRaisesRegex(OSError, "ps timed out"):
                    process_identity(42)

    def test_linux_start_ticks_and_boot_identity_distinguish_pid_reuse(self):
        stat = "42 (worker with ) spaces) S " + "0 " * 18 + "12345 0\n"
        with patch("lite_pi.process_identity.sys.platform", "linux"):
            with patch("pathlib.Path.read_text", side_effect=[stat, "boot-a\n"]):
                first = process_identity(42)
            with patch("pathlib.Path.read_text", side_effect=[stat, "boot-b\n"]):
                second = process_identity(42)
            self.assertEqual(first, "linux:boot-a:12345")
            self.assertNotEqual(first, second)
            with patch("pathlib.Path.read_text", side_effect=FileNotFoundError):
                self.assertIsNone(process_identity(42))
            with patch("pathlib.Path.read_text", side_effect=PermissionError):
                with self.assertRaises(PermissionError):
                    process_identity(42)
            with patch("pathlib.Path.read_text", return_value="42 (bad) S 1"):
                with self.assertRaisesRegex(OSError, "process identity"):
                    process_identity(42)

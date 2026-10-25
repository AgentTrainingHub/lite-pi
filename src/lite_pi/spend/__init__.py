"""Shared DeepSeek spend gate with a persistent usage ledger.

Every paid call from lite-pi, tests, and the local proxy in front of upstream Pi
must reserve against the same ledger before the request leaves the machine.
"""

from lite_pi.spend.ledger import Ledger

__all__ = ["Ledger"]

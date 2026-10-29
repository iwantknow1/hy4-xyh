#!/usr/bin/env python3
"""Shared validation helpers for workspace agent-facing scripts.

TODO: Move it to a shared common directory once one is established
      These are currently scattered across each skill and should be
      consolidated into a single shared directory.
"""

from __future__ import annotations

import re

ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class ValidationError(ValueError):
    """Raised for deterministic user-input validation failures."""


def require_env_name(name: str) -> str:
    if not isinstance(name, str) or not ENV_NAME_RE.fullmatch(name):
        raise ValidationError(
            f"invalid env var name: {name!r}; use ASCII [A-Za-z_][A-Za-z0-9_]*"
        )
    return name


def parse_device_csv(value: str | None, *, label: str = "devices") -> list[int] | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{label} must be a non-empty comma-separated device list")
    devices: list[int] = []
    seen: set[int] = set()
    for raw in value.split(","):
        token = raw.strip()
        if not token:
            raise ValidationError(f"{label} contains an empty device id")
        try:
            device = int(token, 10)
        except ValueError as exc:
            raise ValidationError(f"{label} contains a non-integer device id: {token!r}") from exc
        if device < 0:
            raise ValidationError(f"{label} contains a negative device id: {device}")
        if device in seen:
            raise ValidationError(f"{label} contains a duplicate device id: {device}")
        seen.add(device)
        devices.append(device)
    return sorted(devices)

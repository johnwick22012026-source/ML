"""Configuration service providing merged runtime settings."""

from typing import Any, Dict

__all__ = ["ConfigService"]


class ConfigService:
    """Entry point for assembling configuration overrides."""

    def build(self, overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Merges overrides with a minimal default configuration."""
        defaults: Dict[str, Any] = {
            "log_level": "info",
            "timeout": 30,
        }
        config = {**defaults, **(overrides or {})}
        return config

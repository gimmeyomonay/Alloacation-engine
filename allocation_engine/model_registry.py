"""
Model versioning registry.

Tracks trained model versions with metadata (AUC, date, n_records).
Registry is stored as JSON alongside the models directory.

Usage:
    registry = ModelRegistry()
    registry.register("models/xgb_recovery.pkl", auc=0.765, n_records=5000)
    info = registry.get_active()
    registry.rollback("v1")
"""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path
from typing import Optional


_DEFAULT_REGISTRY_PATH = Path("models") / "registry.json"


class ModelRegistry:
    def __init__(self, registry_path: str | Path = _DEFAULT_REGISTRY_PATH):
        self.registry_path = Path(registry_path)
        self._data = self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> dict:
        if self.registry_path.exists():
            with open(self.registry_path) as f:
                return json.load(f)
        return {"active_version": None, "versions": {}}

    def _save(self) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.registry_path, "w") as f:
            json.dump(self._data, f, indent=2)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(
        self,
        model_path: str,
        auc: float,
        n_records: int,
        make_active: bool = True,
        notes: str = "",
    ) -> str:
        """Register a new model version. Returns the version ID."""
        versions = self._data["versions"]
        version_num = len(versions) + 1
        version_id = f"v{version_num}"

        versions[version_id] = {
            "version_id": version_id,
            "model_path": str(model_path),
            "auc": round(auc, 4),
            "n_records": n_records,
            "trained_date": date.today().isoformat(),
            "notes": notes,
        }

        if make_active or self._data["active_version"] is None:
            self._data["active_version"] = version_id

        self._save()
        return version_id

    def get_active(self) -> Optional[dict]:
        """Return metadata for the currently active model version, or None."""
        active = self._data.get("active_version")
        if active is None:
            return None
        return self._data["versions"].get(active)

    def set_active(self, version_id: str) -> None:
        """Set a specific version as active."""
        if version_id not in self._data["versions"]:
            raise ValueError(f"Version '{version_id}' not found in registry.")
        self._data["active_version"] = version_id
        self._save()

    def rollback(self, version_id: str) -> dict:
        """Roll back to a previous version. Returns its metadata."""
        self.set_active(version_id)
        return self.get_active()

    def list_versions(self) -> list[dict]:
        """Return all registered versions, newest first."""
        versions = list(self._data["versions"].values())
        return list(reversed(versions))

    def get_active_model_path(self) -> Optional[str]:
        """Return the file path of the active model, or None if no model registered."""
        info = self.get_active()
        if info is None:
            return None
        return info["model_path"]

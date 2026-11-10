"""Auxiliary model adapter (OpenAI-compatible) for robot-structure suggestions.

Credentials come from the local environment; when unset the adapter reports
`waiting_for_configuration` and still supports structured JSON import plus a
static validation pipeline that builds the candidate in a throwaway copy.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from .schema import ProjectConfig


class AuxModel:
    def __init__(self, base_url=None, api_key_env="AUX_MODEL_API_KEY", model="gpt-4o-mini"):
        self.base_url = base_url or os.environ.get("AUX_MODEL_BASE_URL")
        self.api_key = os.environ.get(api_key_env)
        self.model = model

    def configured(self):
        return bool(self.base_url and self.api_key)

    def status(self):
        return {"configured": self.configured(),
                "base_url": self.base_url, "model": self.model,
                "api_key_env": "AUX_MODEL_API_KEY" if not self.api_key else "<set>",
                "state": "ready" if self.configured() else "waiting_for_configuration"}

    def suggest(self, prompt: str, views: list[str] | None = None):
        """Return structured candidate suggestions. Requires configuration."""
        if not self.configured():
            return {"status": "waiting_for_configuration",
                    "hint": "set AUX_MODEL_BASE_URL and AUX_MODEL_API_KEY, or import structured JSON",
                    "expected_schema": {
                        "links": [{"name": "str", "parent": "str|null", "mass_kg": "number"}],
                        "joints": [{"name": "str", "child": "str", "type": "revolute|prismatic|fixed",
                                    "axis": "[x,y,z]", "range": "[lo,hi]", "origin": "[x,y,z]"}],
                        "assumptions": ["str"], "units": "m/rad", "needs_confirmation": ["field"]}}
        # Real call when configured; text + optional images, structured JSON out.
        import urllib.request
        content = [{"type": "text", "text": prompt}]
        for v in (views or []):
            url = v if v.startswith(("http://", "https://", "data:")) else f"data:image/png;base64,{v}"
            content.append({"type": "image_url", "image_url": {"url": url}})
        body = json.dumps({"model": self.model, "messages": [
            {"role": "system", "content": "Return ONLY JSON candidate robot structure with fields "
                                          "links/joints/assumptions/units/needs_confirmation."},
            {"role": "user", "content": content}]}).encode()
        req = urllib.request.Request(f"{self.base_url.rstrip('/')}/chat/completions", data=body,
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {self.api_key}"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = json.loads(resp.read())
        text = raw.get("choices", [{}])[0].get("message", {}).get("content", "")
        try:
            candidate = json.loads(text[text.index("{"):text.rindex("}") + 1])
        except (ValueError, KeyError):
            return {"status": "unparsed", "raw": text, "hint": "model did not return JSON; use structured import"}
        return {"status": "candidate", "candidate": candidate,
                "assumptions": candidate.get("assumptions", []),
                "needs_confirmation": candidate.get("needs_confirmation", [])}


def validate_suggestion(suggestion: dict, base_config: ProjectConfig, lab_root: Path):
    """Validate a structured suggestion by building it in an isolated copy."""
    from .store import validate_config
    issues = []
    if not isinstance(suggestion, dict) or "robots" not in suggestion:
        return {"valid": False, "issues": ["suggestion must be a full project config with 'robots'"]}
    try:
        candidate = ProjectConfig.model_validate({**base_config.model_dump(), **suggestion,
                                                  "validation_status": "candidate"})
    except Exception as exc:
        return {"valid": False, "issues": [f"schema: {exc}"]}
    issues.extend(validate_config(candidate, lab_root))
    for robot in candidate.robots:
        for joint in robot.joints:
            if joint.type != "fixed" and abs(sum(v * v for v in joint.axis) - 1) > 1e-3:
                issues.append(f"joint {joint.name}: axis not unit length")
            if joint.type != "fixed" and joint.range and joint.range[0] == joint.range[1]:
                issues.append(f"joint {joint.name}: zero-width range")
    return {"valid": not issues, "issues": issues,
            "candidate_sha256": hashlib.sha256(candidate.model_dump_json().encode()).hexdigest(),
            "applied_fields": sorted(suggestion.keys()),
            "note": "candidate built in a throwaway env; user must apply/revert field by field"}
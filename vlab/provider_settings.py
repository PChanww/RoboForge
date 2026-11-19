"""Server-local configuration for the asset providers and auxiliary planner."""

from __future__ import annotations

from dataclasses import dataclass, field
import ipaddress
import json
import os
from pathlib import Path
import tempfile
from threading import Lock
from urllib.error import HTTPError
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from starlette.concurrency import run_in_threadpool


class ProviderUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, hide_input_in_errors=True)

    aholo_api_key: str | None = Field(default=None, min_length=1, max_length=4096)
    aux_base_url: str | None = Field(default=None, min_length=1, max_length=2048)
    aux_api_key: str | None = Field(default=None, min_length=1, max_length=4096)
    aux_model: str | None = Field(default=None, min_length=1, max_length=256)

    @field_validator("aholo_api_key", "aux_api_key")
    @classmethod
    def validate_key(cls, value):
        if value is not None and any(not "!" <= character <= "~" for character in value):
            raise ValueError("Enter the raw API Key without spaces or line breaks")
        return value

    @field_validator("aux_base_url")
    @classmethod
    def normalize_base_url(cls, value):
        if value is None:
            return None
        try:
            parts = urlsplit(value)
            port = parts.port
        except ValueError:
            raise ValueError("Enter a valid HTTP or HTTPS Base URL") from None
        if (parts.scheme not in {"http", "https"} or not parts.hostname or port == 0
                or parts.username is not None or parts.password is not None
                or "?" in value or "#" in value or any(c.isspace() or ord(c) < 32 for c in value)):
            raise ValueError("Enter an HTTP or HTTPS Base URL without credentials or query parameters")
        return value.rstrip("/").removesuffix("/chat/completions")


@dataclass(frozen=True)
class ProviderConfig:
    aholo_api_key: str | None = field(repr=False)
    lux3d_api_key: str | None = field(repr=False)
    aux_base_url: str | None
    aux_api_key: str | None = field(repr=False)
    aux_model: str

    def public(self):
        return {
            "aholo": {"key_configured": bool(self.lux3d_api_key), "world_configured": bool(self.aholo_api_key)},
            "aux": {"base_url": self.aux_base_url, "model": self.aux_model,
                    "key_configured": bool(self.aux_api_key)},
        }


class ProviderSettings:
    def __init__(self, lab_root: Path):
        self.path = Path(lab_root) / ".env.providers.json"
        self._lock = Lock()

    def _read(self):
        try:
            text = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return {}
        try:
            return ProviderUpdate.model_validate_json(text).model_dump(exclude_none=True)
        except ValidationError:
            raise RuntimeError("Cannot read the backend provider settings file") from None

    def resolve(self, update: ProviderUpdate | None = None) -> ProviderConfig:
        values = self._read()
        if update is not None:
            values.update(update.model_dump(exclude_none=True))
        aholo = values.get("aholo_api_key") or os.environ.get("AHOLO_API_KEY")
        return ProviderConfig(
            aholo_api_key=aholo,
            lux3d_api_key=aholo or os.environ.get("LUX3D_CN_API_KEY") or os.environ.get("LUX3D_API_KEY"),
            aux_base_url=values.get("aux_base_url") or os.environ.get("AUX_MODEL_BASE_URL"),
            aux_api_key=values.get("aux_api_key") or os.environ.get("AUX_MODEL_API_KEY"),
            aux_model=values.get("aux_model") or os.environ.get("AUX_MODEL_MODEL") or "gpt-4o-mini",
        )

    def save(self, update: ProviderUpdate) -> ProviderConfig:
        with self._lock:
            values = {**self._read(), **update.model_dump(exclude_none=True)}
            out = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.path.parent,
                                              prefix=".env.providers.", suffix=".tmp", delete=False)
            temporary = Path(out.name)
            try:
                with out:
                    json.dump(values, out, ensure_ascii=False)
                temporary.replace(self.path)
            finally:
                temporary.unlink(missing_ok=True)
            return self.resolve()

    def planner(self):
        from .generation.planner import Planner

        cfg = self.resolve()
        return Planner(base_url=cfg.aux_base_url, api_key=cfg.aux_api_key, model=cfg.aux_model)

    def world_key(self):
        return self.resolve().aholo_api_key

    def lux3d_key(self):
        return self.resolve().lux3d_api_key


def _loopback(host):
    if host == "localhost":
        return True
    try:
        address = ipaddress.ip_address(host or "")
    except ValueError:
        return False
    return address.is_loopback or bool(getattr(address, "ipv4_mapped", None)
                                       and address.ipv4_mapped.is_loopback)


def _require_local(request: Request):
    origin = request.headers.get("origin")
    try:
        valid_origin = origin is None or _loopback(urlsplit(origin).hostname)
    except ValueError:
        valid_origin = False
    if not (request.client and _loopback(request.client.host)
            and _loopback(request.url.hostname) and valid_origin):
        raise HTTPException(403, "请在后端所在电脑通过 localhost 或 127.0.0.1 打开设置页。")


async def _read_update(request: Request) -> ProviderUpdate:
    try:
        return ProviderUpdate.model_validate(await request.json())
    except ValidationError as exc:
        # Report field names without echoing submitted data in validation errors.
        allowed = ProviderUpdate.model_fields
        fields = sorted({error["loc"][0] for error in exc.errors(include_input=False)
                         if error["loc"] and error["loc"][0] in allowed})
        msg = f"请检查这些配置项：{', '.join(fields)}" if fields else "提交的配置格式无效。"
        raise HTTPException(400, msg) from None
    except ValueError:
        raise HTTPException(400, "提交的配置格式无效。") from None


def _check(label, operation, missing):
    if missing:
        return {"service": label, "ok": False, "message": missing}
    try:
        operation()
    except NotImplementedError:
        return {"service": label, "ok": False, "message": "请安装此服务的 SDK 后重试。"}
    except Exception as exc:
        status = exc.code if isinstance(exc, HTTPError) else getattr(exc, "status_code", None)
        suffix = f"（HTTP {status}）" if type(status) is int and 400 <= status < 600 else ""
        return {"service": label, "ok": False, "message": f"连接失败{suffix}，请检查配置、服务权限和网络。"}
    return {"service": label, "ok": True, "message": "连接成功。"}


def _test(settings, update, service):
    from .generation.object_provider import Lux3DProvider
    from .generation.planner import Planner
    from .world_provider import WorldProvider

    cfg = settings.resolve(update)
    if service == "aholo":
        lux = Lux3DProvider(key_provider=lambda: cfg.lux3d_api_key, timeout_ms=10000)
        world = WorldProvider(key_provider=lambda: cfg.aholo_api_key, timeout_ms=10000)
        return [
            _check("Lux3D", lambda: lux.list_remote_tasks(page=1, page_size=1),
                   "请输入 Aholo API Key。" if not lux.configured() else None),
            _check("Aholo World", lambda: world.list_remote_tasks(page_num=0, page_size=1),
                   "请输入 Aholo API Key。" if not world.configured() else None),
        ]
    planner = Planner(base_url=cfg.aux_base_url, api_key=cfg.aux_api_key, model=cfg.aux_model)
    return [_check("辅助 LLM", planner.test_connection,
                   "请填写 Base URL、模型 ID 和 API Key。" if not planner.configured() else None)]


def register_provider_settings_api(app: FastAPI, settings: ProviderSettings):
    route = "/api/v1/settings/providers"

    @app.get(route)
    def read_provider_settings(request: Request):
        _require_local(request)
        return JSONResponse(settings.resolve().public(), headers={"Cache-Control": "no-store"})

    @app.put(route)
    async def save_provider_settings(request: Request):
        _require_local(request)
        update = await _read_update(request)
        try:
            cfg = await run_in_threadpool(settings.save, update)
        except OSError:
            raise HTTPException(500, "无法保存后端配置，请检查项目目录的写入权限。") from None
        return JSONResponse(cfg.public(), headers={"Cache-Control": "no-store"})

    @app.post(route + "/test/{service}")
    async def test_provider_settings(service: str, request: Request):
        _require_local(request)
        if service not in {"aholo", "aux"}:
            raise HTTPException(404, "未知服务。")
        update = await _read_update(request)
        results = await run_in_threadpool(_test, settings, update, service)
        return JSONResponse({"results": results}, headers={"Cache-Control": "no-store"})

"""Transport asset generation and explicitly sourced physical suggestions.

Aholo generates meshes. Numeric mass/friction values are user inputs or estimates,
never claimed to have been measured or predicted by the mesh provider.
"""
from __future__ import annotations

import math
import re
from datetime import datetime, timezone

from .generation.api import SpecPatchBody, apply_user_patch
from .generation.schema import ChatMessage, GenerationSession, FieldSource

DEFAULT_PARAMETERS = {"height_m": .06, "width_m": .05, "depth_m": .055,
                      "mass_kg": .035, "friction": 1.2}
PARAMETER_LIMITS = {"height_m": (.02, .075), "width_m": (.015, .085),
                    "depth_m": (.015, .085), "mass_kg": (.005, .15), "friction": (.2, 3.)}


def _now():
    return datetime.now(timezone.utc).isoformat()


def explicit_parameters(description):
    """Read explicit dimensions without guessing units or treating them as AI."""
    values = {}
    for field, label in (("height_m", "高度|高|height"), ("width_m", "宽度|宽|width"),
                         ("depth_m", "深度|深|长度|长|depth")):
        match = re.search(rf"(?:{label})\s*[:：=为约]?\s*(\d+(?:\.\d+)?)\s*(毫米|厘米|米|mm|cm|m)(?![a-z])", description.lower())
        if match:
            scale = .001 if match[2] in ("mm", "毫米") else .01 if match[2] in ("cm", "厘米") else 1.
            values[field] = float(match[1])*scale
    match = re.search(r"(?:质量|重量|重|mass)\s*[:：=为约]?\s*(\d+(?:\.\d+)?)\s*(千克|公斤|克|kg|g)(?![a-z])", description.lower())
    if match:
        values["mass_kg"] = float(match[1])*(.001 if match[2] in ("g", "克") else 1.)
    match = re.search(r"(?:摩擦系数|摩擦|friction)\s*[:：=为约]?\s*(\d+(?:\.\d+)?)", description.lower())
    if match:
        values["friction"] = float(match[1])
    return values


def check_parameters(parameters):
    clean = {}
    for key, value in parameters.items():
        if value is None:
            continue
        if key not in PARAMETER_LIMITS:
            raise ValueError(f"不支持的物品参数：{key}")
        low, high = PARAMETER_LIMITS[key]
        value = float(value)
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"物品参数{key}需要在{low:g}～{high:g}之间（米／千克／无量纲）。")
        clean[key] = value
    return clean


def suggest_parameters(description, current=None, generation=None):
    explicit = check_parameters(explicit_parameters(description))
    explicit.update(check_parameters(current or {}))
    parameters = dict(DEFAULT_PARAMETERS)
    provenance = {key: {"source": "rules_estimate", "verification": "unverified"}
                  for key in parameters}
    notes = ["尺寸、质量和摩擦均为演示参数，未经实物标定；Aholo只生成外观网格。"]
    planner = (generation or {}).get("planner")
    available = bool(planner and planner.configured())
    # These remain honest local fallbacks if a configured model fails or
    # returns no usable fields. Mesh generation must still be usable offline.
    if any(word in description.lower() for word in ("金属", "钢", "metal", "steel")):
        parameters.update(mass_kg=.09, friction=.6)
    elif any(word in description.lower() for word in ("橡胶", "rubber")):
        parameters.update(mass_kg=.045, friction=1.8)
    if available and set(parameters)-set(explicit):
        now = _now()
        session = GenerationSession(session_id="transport_suggestion", created=now, updated=now,
                                    messages=[ChatMessage(role="user", ts=now, content=(
                                        "为可抓取小型刚体建议尺寸、质量和摩擦，尺寸高2.5～7.5cm、宽深1.8～8.5cm，质量10～120g。"
                                        "用户提供的参数必须保留。需求："+description+"；已给参数："+str(explicit)))])
        try:
            registry = generation["registry"]() if generation.get("registry") else {}
            reply = planner.plan(session, registry)
            proposals = {key: reply.spec_patch.get(key) for key in ("height_m", "width_m", "depth_m")}
            proposals["mass_kg"] = reply.physics_patch.get("mass_kg")
            friction = reply.physics_patch.get("friction")
            proposals["friction"] = friction[0] if isinstance(friction, list) and friction else friction
            for key, value in proposals.items():
                if value is None or key in explicit:
                    continue
                try:
                    value = check_parameters({key: value})[key]
                except (ValueError, TypeError):
                    notes.append(f"模型建议{key}超出本机构范围，保留本地规则值。")
                    continue
                parameters[key] = value
                provenance[key] = {"source": "model_inferred", "verification": "unverified"}
            notes.extend(reply.warnings)
            if reply.reply_zh:
                notes.append(reply.reply_zh)
            if not any(item["source"] == "model_inferred" for item in provenance.values()):
                notes.append("语言模型未返回可用的缺省参数，缺省项使用本地规则推荐。")
        except Exception:
            # Provider response bodies can contain credentials or prompts.
            notes.append("参数建议语言模型暂时不可用，缺省项使用本地规则推荐；已保留用户参数。")
    elif not available:
        notes.append("尚未配置参数建议语言模型，当前返回本地规则推荐；可直接填写参数。")
    else:
        notes.append("参数已齐全，保留现有数值，未请求语言模型。")
    for key, value in explicit.items():
        parameters[key] = value
        provenance[key] = {"source": "user_explicit", "verification": "unverified"}
    source = "model_inferred" if any(item["source"] == "model_inferred" for item in provenance.values()) else "rules_estimate"
    return {"parameters": parameters, "source": source, "provenance": provenance,
            "notes": notes, "ai_available": available}


class GenerationBlocked(ValueError):
    pass


def prepare_assets(row, generation, publish, cancelled, wait, cache_asset=None):
    """Execute durable two-role generation before constructing physics.

    Every paid submit is bound to a stable request id in the existing store.
    Session ids are persisted before patch/confirm, and known job ids reused.
    """
    spec = row["spec"]
    for step in row["asset_steps"]:
        if cancelled():
            return False
        role = step["role"]
        if step["status"] == "succeeded" and step.get("asset_id"):
            # Reconcile a crash after publishing success but before publishing
            # the resolved asset reference; never fall back to the old object.
            spec["object_asset" if role == "object" else "bin_asset"] = step["asset_id"]
            if cache_asset:
                cache_asset(step["asset_id"], role)
            publish()
        if step["status"] in ("reused", "builtin", "succeeded"):
            continue
        if not generation:
            raise GenerationBlocked("Aholo生成上下文不可用。")
        sessions = generation["sessions"]
        if not step.get("session_id"):
            step["session_id"] = sessions.create().session_id
            publish()
        session = sessions.get(step["session_id"])
        if not step.get("patched"):
            if role == "object":
                existing = spec.get("object_parameters", {})
                sources = spec.setdefault("object_parameter_sources", {})
                suggestion = suggest_parameters(spec["object_description"], existing, generation)
                # Existing values may come from an earlier suggestion. Passing
                # them as hard constraints must not relabel them as user input.
                for key in existing:
                    if key in sources:
                        suggestion["provenance"][key]["source"] = sources[key]
                suggestion["source"] = "model_inferred" if any(
                    item["source"] == "model_inferred" for item in suggestion["provenance"].values()) else "rules_estimate"
                params = suggestion["parameters"]
                appearance = "A compact single graspable solid object, no surrounding scene, no support base. " + spec["object_description"]
                patch = dict(kind="task_object", role="dynamic", name=spec.get("object_name") or "transport-object", appearance=appearance,
                             user_words=spec["object_description"], height_m=params["height_m"],
                             width_m=params["width_m"], depth_m=params["depth_m"],
                             radius_m=max(params["width_m"], params["depth_m"])/2, mass_kg=params["mass_kg"],
                             friction=[params["friction"], .003, .0001], graspable=True)
                spec["object_parameters"] = params
                recorded = row.setdefault("parameter_provenance", {})
                for key, provenance in suggestion["provenance"].items():
                    recorded.setdefault(key, provenance)
                    sources[key] = recorded[key]["source"]
                row["parameter_suggestion"] = {key: suggestion[key] for key in ("source", "ai_available", "notes")}
            else:
                width, depth, height = spec["bin_size_m"]
                patch = dict(kind="static_prop", role="visual_only", name=spec.get("bin_name") or "transport-bin", open_top=True,
                             appearance=("A single EMPTY OPEN TOP rectangular storage bin. Thin straight vertical walls, "
                                         "flat bottom, NO lid, NO contents, NO handles sticking into the opening, "
                                         "no surrounding scene. Appearance requirements: "+spec["bin_description"]),
                             user_words=spec["bin_description"], width_m=width, depth_m=depth, height_m=height,
                             mass_kg=.3)
            apply_user_patch(session, SpecPatchBody(expected_revision=session.revision, **patch))
            if role == "object":
                provenance = row["parameter_provenance"]
                for key, dim in (("height_m", "height"), ("width_m", "width"), ("depth_m", "depth")):
                    getattr(session.asset_spec, dim).provenance.source = FieldSource(provenance[key]["source"] if provenance[key]["source"] != "rules_estimate" else "default")
                session.physics_spec.mass.provenance.source = FieldSource(provenance["mass_kg"]["source"] if provenance["mass_kg"]["source"] != "rules_estimate" else "default")
                session.physics_spec.friction_provenance.source = FieldSource(provenance["friction"]["source"] if provenance["friction"]["source"] != "rules_estimate" else "default")
                session.asset_spec.radius.provenance.source = FieldSource("model_inferred" if any(
                    provenance[key]["source"] == "model_inferred" for key in ("width_m", "depth_m")) else "default")
            session.messages.append(ChatMessage(role="user", content=patch["user_words"], ts=_now()))
            sessions.save(session, session.revision)
            step["patched"] = True
            publish()
        if cancelled():
            return False
        if not step.get("job_id"):
            # Register the transport cache before any asynchronous import can
            # finish, so it never briefly appears in the reusable asset list.
            if cache_asset:
                existing_job = generation["objects"].find_by_request(f"{row['id']}:{role}")
                slug = existing_job["slug"] if existing_job else generation["orchestrator"]._slug(session)
                cache_asset("imported/"+slug, role)
            result = generation["orchestrator"].confirm(step["session_id"],
                client_request_id=f"{row['id']}:{role}", provider="lux3d", background=True)
            step["job_id"] = result["job"]["id"]
            step["status"] = result["job"]["state"]
            if cache_asset and result["job"].get("slug"):
                cache_asset("imported/"+result["job"]["slug"], role)
            publish()
        while not cancelled():
            job = generation["objects"].poll(step["job_id"], background=True)
            step.update(status=job["state"], asset_id=job.get("asset_id"), error=job.get("error"), node=job.get("node"))
            publish()
            if job["state"] == "succeeded":
                ref = job.get("asset_id")
                if not ref:
                    raise GenerationBlocked("生成任务结束但没有导入资产。")
                spec["object_asset" if role == "object" else "bin_asset"] = ref
                if cache_asset:
                    cache_asset(ref, role)
                publish()
                break
            if job["state"] in ("failed", "retryable_error", "submit_unknown", "unconfigured"):
                raise GenerationBlocked(f"{role}模型处理未完成：{job.get('error') or job['state']}。可恢复已知远端任务，不会盲目重新收费提交。")
            wait(2.)
        if cancelled():
            return False
    return True

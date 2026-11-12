"""Conversational asset planning and scene-generation loop (round 4).

Modules: schema (data models), capabilities (registry), sessions (SQLite
persistence), planner (LLM adapter), compiler (deterministic prompt compiler),
object_provider / orchestrator (job execution), scene_recipe (layout/validation/
apply). The LLM only understands and suggests; programs validate, orchestrate
and land results.
"""
"""Transactional offline sleep consolidation with deterministic guardrails."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from always_on.config import Settings
from always_on.db.store import AgentStore
from always_on.embeddings.provider import EmbeddingProvider
from always_on.llm.client import HaikuClient
from always_on.llm.prompts import (
    build_consolidation_system_blocks,
    build_consolidation_user_message,
)
from always_on.models import (
    ConsolidationOperation,
    ConsolidationPlan,
    MemoryItem,
    RejectedOperation,
    SleepCycleReport,
    consolidation_json_schema,
    new_id,
)
from always_on.safety.circuit_breaker import AgentCircuitBreaker, CircuitBreakerTripped

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def count_sentences(text: str) -> int:
    """Count sentences conservatively for atomic max-2-sentence enforcement."""
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        return 0
    parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(cleaned) if p.strip()]
    return max(1, len(parts))


@dataclass
class GuardrailValidationResult:
    """Outcome of running deterministic guardrails on a ConsolidationPlan before DB commit."""

    aborted: bool
    abort_reason: str | None
    accepted_ops: list[ConsolidationOperation]
    rejected_ops: list[RejectedOperation]
    cleaned_priorities: list[str]


def validate_consolidation_plan(
    plan: ConsolidationPlan,
    *,
    core_memory_map: dict[str, MemoryItem],
    valid_episode_ids: set[str],
    settings: Settings,
) -> GuardrailValidationResult:
    """Enforce all deterministic consolidation guardrails BEFORE any database write.

    Guardrails enforced:
      1. Churn Circuit Breaker: Abort the entire commit if a single pass attempts
         to ARCHIVE > 25% of active Core Memory (when Core Memory has > 10 items).
      2. Strict Provenance: Reject any operation whose source_episode_ids are empty
         or not a strict subset of current fast_buffer episode IDs.
      3. Pinned Protection: Reject any UPDATE or ARCHIVE targeting a pinned=True item.
      4. Target & Atomicity Validation: Ensure UPDATE/ARCHIVE targets exist, avoid
         conflicting double mutations on the same ID, and enforce max 2 sentences.
    """
    # 1. Churn Circuit Breaker (checked first across all requested ARCHIVE operations)
    core_count = len(core_memory_map)
    if core_count > settings.churn_min_core_items:
        attempted_archive_targets = {
            op.target_id
            for op in plan.operations
            if op.op == "ARCHIVE" and op.target_id and op.target_id in core_memory_map
        }
        archive_ratio = len(attempted_archive_targets) / float(core_count)
        if archive_ratio > settings.churn_max_archive_ratio:
            return GuardrailValidationResult(
                aborted=True,
                abort_reason=(
                    f"Churn circuit breaker tripped: pass attempted to ARCHIVE "
                    f"{len(attempted_archive_targets)}/{core_count} items "
                    f"({archive_ratio:.1%} > {settings.churn_max_archive_ratio:.0%})."
                ),
                accepted_ops=[],
                rejected_ops=[],
                cleaned_priorities=[],
            )

    accepted: list[ConsolidationOperation] = []
    rejected: list[RejectedOperation] = []
    mutated_targets: set[str] = set()

    for op in plan.operations:
        # 2. Strict Provenance: non-empty subset of current fast_buffer episode IDs
        if not op.source_episode_ids:
            rejected.append(
                RejectedOperation(
                    op=op,
                    guardrail="strict_provenance",
                    reason="Operation missing required source_episode_ids.",
                )
            )
            continue

        invalid_ids = [eid for eid in op.source_episode_ids if eid not in valid_episode_ids]
        if invalid_ids:
            rejected.append(
                RejectedOperation(
                    op=op,
                    guardrail="strict_provenance",
                    reason=(
                        f"source_episode_ids {invalid_ids} are not in the current fast_buffer."
                    ),
                )
            )
            continue

        # 3. Operation-specific validation & Pinned Protection
        if op.op == "ADD":
            if op.category is None or not op.content:
                rejected.append(
                    RejectedOperation(
                        op=op,
                        guardrail="schema_integrity",
                        reason="ADD operation requires both category and non-empty content.",
                    )
                )
                continue
            if count_sentences(op.content) > 2:
                rejected.append(
                    RejectedOperation(
                        op=op,
                        guardrail="atomicity_limit",
                        reason="ADD content exceeds atomic limit of 2 sentences.",
                    )
                )
                continue
            accepted.append(op)

        elif op.op in ("UPDATE", "ARCHIVE"):
            if not op.target_id or op.target_id not in core_memory_map:
                rejected.append(
                    RejectedOperation(
                        op=op,
                        guardrail="missing_target",
                        reason=f"{op.op} target_id {op.target_id!r} not found in active Core Memory.",
                    )
                )
                continue

            target_item = core_memory_map[op.target_id]
            if target_item.pinned:
                rejected.append(
                    RejectedOperation(
                        op=op,
                        guardrail="pinned_protection",
                        reason=(
                            f"Rejected {op.op} targeting pinned Core Memory item '{op.target_id}'."
                        ),
                    )
                )
                continue

            if op.target_id in mutated_targets:
                rejected.append(
                    RejectedOperation(
                        op=op,
                        guardrail="duplicate_target_mutation",
                        reason=(
                            f"Rejected duplicate mutation targeting '{op.target_id}' in same pass."
                        ),
                    )
                )
                continue

            if op.op == "UPDATE":
                if not op.content:
                    rejected.append(
                        RejectedOperation(
                            op=op,
                            guardrail="schema_integrity",
                            reason="UPDATE operation requires non-empty content.",
                        )
                    )
                    continue
                if count_sentences(op.content) > 2:
                    rejected.append(
                        RejectedOperation(
                            op=op,
                            guardrail="atomicity_limit",
                            reason="UPDATE content exceeds atomic limit of 2 sentences.",
                        )
                    )
                    continue

            mutated_targets.add(op.target_id)
            accepted.append(op)

    # Clean active priorities (target 3-5 concise statements)
    cleaned_priorities: list[str] = []
    seen_prio: set[str] = set()
    for p in plan.active_priorities:
        norm = " ".join(p.strip().split())
        if norm and norm.lower() not in seen_prio:
            seen_prio.add(norm.lower())
            cleaned_priorities.append(norm)
        if len(cleaned_priorities) >= 5:
            break

    return GuardrailValidationResult(
        aborted=False,
        abort_reason=None,
        accepted_ops=accepted,
        rejected_ops=rejected,
        cleaned_priorities=cleaned_priorities,
    )


async def sleep_and_consolidate(
    *,
    store: AgentStore,
    llm_client: HaikuClient,
    embedder: EmbeddingProvider,
    circuit_breaker: AgentCircuitBreaker,
    settings: Settings,
    trigger_reason: str = "manual",
    fatigue_before: float = 0.0,
) -> SleepCycleReport:
    """Run an offline transactional sleep consolidation pass.

    Steps:
      1. Snapshot current fast_buffer episodes and active Core Memory.
      2. Invoke Claude Haiku 5.5 with adaptive thinking + medium/high effort + JSON Schema.
      3. Enforce deterministic guardrails (Churn Breaker, Strict Provenance, Pinned Protection).
      4. Commit ADD/UPDATE/ARCHIVE + soft-archives + re-embedded priorities + fast_buffer flush
         in a single ACID SQLite transaction.
      5. Never flush fast_buffer if aborted, refused, or failed.
    """
    sleep_cycle_id = new_id("sleep")
    episodes = await store.list_fast_buffer()
    if not episodes:
        return SleepCycleReport(
            sleep_cycle_id=sleep_cycle_id,
            status="skipped",
            trigger_reason=trigger_reason,
            fatigue_before=fatigue_before,
            episodes_processed=0,
            reflection="Fast working buffer is empty; no consolidation needed.",
        )

    await store.create_sleep_cycle(
        sleep_cycle_id=sleep_cycle_id,
        trigger_reason=trigger_reason,
        fatigue_before=fatigue_before,
        episodes_count=len(episodes),
    )

    core_memory = await store.list_core_memory()
    core_memory_map = {m.id: m for m in core_memory}
    active_priorities = await store.list_active_priorities()
    valid_episode_ids = {ep.id for ep in episodes}

    try:
        await circuit_breaker.check_before_call(is_wake_start=False)
    except CircuitBreakerTripped as exc:
        await store.mark_sleep_cycle_aborted_or_failed(
            sleep_cycle_id, status="aborted", reason=str(exc)
        )
        return SleepCycleReport(
            sleep_cycle_id=sleep_cycle_id,
            status="aborted",
            trigger_reason=trigger_reason,
            fatigue_before=fatigue_before,
            episodes_processed=len(episodes),
            abort_reason=str(exc),
        )

    system_blocks = build_consolidation_system_blocks(settings.agent_name)
    user_msg = build_consolidation_user_message(
        core_memory=core_memory,
        episodes=episodes,
        active_priorities=active_priorities,
    )

    usd_cost = 0.0
    try:
        response = await llm_client.create_message(
            system=system_blocks,
            messages=[{"role": "user", "content": user_msg}],
            effort=settings.sleep_effort,
            max_tokens=settings.max_output_tokens_sleep,
            json_schema=consolidation_json_schema(),
        )
        usd_cost = await circuit_breaker.record_usage(response.usage)
    except Exception as exc:
        await circuit_breaker.record_error()
        await store.mark_sleep_cycle_aborted_or_failed(
            sleep_cycle_id, status="failed", reason=f"LLM error: {exc}"
        )
        return SleepCycleReport(
            sleep_cycle_id=sleep_cycle_id,
            status="failed",
            trigger_reason=trigger_reason,
            fatigue_before=fatigue_before,
            episodes_processed=len(episodes),
            abort_reason=f"LLM error: {exc}",
        )

    # Handle Haiku 5.5 stop reasons gracefully
    if response.is_refusal:
        reason = "Consolidation pass refused by model (stop_reason=refusal)."
        await store.mark_sleep_cycle_aborted_or_failed(
            sleep_cycle_id, status="aborted", reason=reason
        )
        return SleepCycleReport(
            sleep_cycle_id=sleep_cycle_id,
            status="aborted",
            trigger_reason=trigger_reason,
            fatigue_before=fatigue_before,
            episodes_processed=len(episodes),
            abort_reason=reason,
            usd_cost=usd_cost,
        )

    if response.is_context_exceeded:
        reason = (
            "Consolidation prompt exceeded model context window "
            "(stop_reason=model_context_window_exceeded)."
        )
        await store.mark_sleep_cycle_aborted_or_failed(
            sleep_cycle_id, status="aborted", reason=reason
        )
        return SleepCycleReport(
            sleep_cycle_id=sleep_cycle_id,
            status="aborted",
            trigger_reason=trigger_reason,
            fatigue_before=fatigue_before,
            episodes_processed=len(episodes),
            abort_reason=reason,
            usd_cost=usd_cost,
        )

    try:
        raw_plan = json.loads(response.text)
        plan = ConsolidationPlan.model_validate(raw_plan)
    except Exception as exc:
        await circuit_breaker.record_error()
        reason = f"Failed to parse structured consolidation JSON: {exc}"
        await store.mark_sleep_cycle_aborted_or_failed(
            sleep_cycle_id, status="failed", reason=reason
        )
        return SleepCycleReport(
            sleep_cycle_id=sleep_cycle_id,
            status="failed",
            trigger_reason=trigger_reason,
            fatigue_before=fatigue_before,
            episodes_processed=len(episodes),
            abort_reason=reason,
            usd_cost=usd_cost,
        )

    # Deterministic Guardrails
    validation = validate_consolidation_plan(
        plan,
        core_memory_map=core_memory_map,
        valid_episode_ids=valid_episode_ids,
        settings=settings,
    )

    if validation.aborted:
        await store.mark_sleep_cycle_aborted_or_failed(
            sleep_cycle_id,
            status="aborted",
            reason=validation.abort_reason or "Guardrail aborted commit",
            ops_summary={"attempted_operations": len(plan.operations)},
        )
        return SleepCycleReport(
            sleep_cycle_id=sleep_cycle_id,
            status="aborted",
            trigger_reason=trigger_reason,
            fatigue_before=fatigue_before,
            episodes_processed=len(episodes),
            abort_reason=validation.abort_reason,
            reflection=plan.reflection,
            usd_cost=usd_cost,
        )

    # Re-embed updated active priorities (if supplied) for local salience scoring
    priorities_with_vectors = []
    if validation.cleaned_priorities:
        vecs = embedder.embed_batch(validation.cleaned_priorities)
        priorities_with_vectors = list(zip(validation.cleaned_priorities, vecs))

    ops_summary = {
        "reflection": plan.reflection,
        "accepted_count": len(validation.accepted_ops),
        "rejected_count": len(validation.rejected_ops),
        "rejected": [
            {
                "op": r.op.op,
                "target_id": r.op.target_id,
                "guardrail": r.guardrail,
                "reason": r.reason,
            }
            for r in validation.rejected_ops
        ],
        "active_priorities": validation.cleaned_priorities,
        "usd_cost": usd_cost,
    }

    try:
        added_ids, updated_ids, archived_ids = await store.apply_consolidation_transaction(
            sleep_cycle_id=sleep_cycle_id,
            validated_ops=validation.accepted_ops,
            priorities_with_vectors=priorities_with_vectors,
            episode_ids_to_flush=[ep.id for ep in episodes],
            ops_summary=ops_summary,
        )
        await circuit_breaker.record_success()
    except Exception as exc:
        await circuit_breaker.record_error()
        reason = f"Transactional commit rolled back: {exc}"
        await store.mark_sleep_cycle_aborted_or_failed(
            sleep_cycle_id, status="failed", reason=reason
        )
        return SleepCycleReport(
            sleep_cycle_id=sleep_cycle_id,
            status="failed",
            trigger_reason=trigger_reason,
            fatigue_before=fatigue_before,
            episodes_processed=len(episodes),
            abort_reason=reason,
            usd_cost=usd_cost,
        )

    return SleepCycleReport(
        sleep_cycle_id=sleep_cycle_id,
        status="committed",
        trigger_reason=trigger_reason,
        fatigue_before=fatigue_before,
        episodes_processed=len(episodes),
        added_ids=added_ids,
        updated_ids=updated_ids,
        archived_ids=archived_ids,
        rejected_operations=validation.rejected_ops,
        active_priorities=validation.cleaned_priorities
        or [p.description for p in active_priorities],
        reflection=plan.reflection,
        usd_cost=usd_cost,
    )

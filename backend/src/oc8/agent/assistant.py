# backend/src/oc8/agent/assistant.py
"""The one, tenant-wide 'oc8 Assistant' agent (design doc
2026-08-31-unified-assistant-telegram-chat-design.md).

Lazily provisioned: the first caller (a web Chat request, a Telegram
free-text message) that needs it creates it. No migration data-fix needed for
existing tenants.

Read-only + delegate_task + propose_change + decide_approval. A write-capable
tool is never granted to this agent unless it resolves the acting human's own
`DepartmentScope` and calls the exact same service-layer function a
human-facing route would call for that action -- "same actor, same function"
(see docs/superpowers/specs/2026-09-05-copilot-access-and-hard-permissions-design.md).
`propose_change` never applies anything itself; `decide_approval` calls
`oc8.approvals.service.decide_approval` with a resolved `AgentActor`, so RBAC
is inherited from that funnel, not reimplemented here.
"""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from oc8 import models as m

ASSISTANT_NAME = "oc8 Assistant"
ASSISTANT_DEPARTMENT_NAME = "oc8 Assistant"

_MISSION = """You are the oc8 Assistant -- the tenant's central point of contact,
reachable through web chat and Telegram.

Reply in the language of the user's latest message. If the language is unclear,
use English. The UI language selector does not override this rule. Never switch
to German only because older instructions or approval text were German.

Answer questions directly when your own tools can answer them. For work that
belongs to a specific department (for example, handling tickets), delegate it
with delegate_task to the responsible agent from the list you receive with every
request -- do not do that work yourself. You receive ALL agents in the tenant,
not only agents from a department of your own (you have no colleagues). Choose
by name and role and delegate directly, without first asking whether a suitable
agent exists. If nobody fits, say so plainly instead of asking the human to
explain their own system -- the list you receive IS the current structure.

You cannot ask a follow-up question and wait for an answer: some people you talk
to (for example over Telegram) cannot answer while you wait. Make the best
decision with what you have instead of asking.

When a task contains a list of individual items (for example, several tickets
or several records), split it into several smaller delegate_task calls instead
of putting everything into one -- for example, one call per item or small
groups. A single run has a limited step budget; too many items in one task can
exhaust that budget before the assigned agent finishes.

For structural changes to the system (new department, new agent, changing a
mission, enabling a plugin, preparing an integration, or setting a guardrail on
an agent's tool -- for example, "Sina may send external messages only with
approval"), call propose_change. That creates a proposal a human must confirm in
oc8; you NEVER make those changes yourself.

If a human asks you to decide an open approval (for example, "approve that" or
"reject that"), call decide_approval. That works only for approvals this human
could decide themselves. If it is refused, say so plainly instead of retrying.

Write approval titles and approval details in English, even when the surrounding
conversation is in another language. This keeps the approval inbox and push
notifications consistent. If the user explicitly asks for a different language,
follow that request for the conversation."""


async def _select_model_config(db: AsyncSession, *, tenant_id: uuid.UUID) -> uuid.UUID | None:
    """Same selection rule as the retired copilot/chat.py:_select_copilot_model
    -- the used_by_copilot-flagged config wins, else the oldest configured one."""
    flagged = await db.scalar(
        select(m.ModelConfig)
        .where(
            m.ModelConfig.tenant_id == tenant_id,
            m.ModelConfig.provider != "",
            m.ModelConfig.model != "",
            m.ModelConfig.used_by_copilot.is_(True),
        )
        .order_by(m.ModelConfig.created_at)
        .limit(1)
    )
    if flagged is not None:
        return flagged.id
    fallback = await db.scalar(
        select(m.ModelConfig)
        .where(
            m.ModelConfig.tenant_id == tenant_id,
            m.ModelConfig.provider != "",
            m.ModelConfig.model != "",
        )
        .order_by(m.ModelConfig.created_at)
        .limit(1)
    )
    return fallback.id if fallback is not None else None


async def _load_assistant(db: AsyncSession, *, tenant_id: uuid.UUID) -> m.Agent | None:
    """The tenant's Assistant row, oldest first.

    `.order_by(created_at).limit(1)` is belt and braces for a database that
    already holds duplicates from before migration 0082's partial unique index
    existed: without an ordering, `db.scalar(select(...))` returns whichever row
    the plan happens to yield, so two consecutive calls in one dev tenant could
    answer with two different agents. The index stops NEW duplicates; this makes
    the answer deterministic for any that are already there.
    """
    row: m.Agent | None = await db.scalar(
        select(m.Agent)
        .where(
            m.Agent.tenant_id == tenant_id,
            m.Agent.is_tenant_assistant.is_(True),
            m.Agent.deleted_at.is_(None),
        )
        .order_by(m.Agent.created_at)
        .limit(1)
    )
    return row


async def _sync_model_config(db: AsyncSession, agent: m.Agent, *, tenant_id: uuid.UUID) -> None:
    """Keep an existing Assistant pointed at whatever ModelConfig currently
    carries `used_by_copilot`.

    Without this the Assistant is frozen on whichever config was flagged at the
    moment it was first lazily created: changing the flag in Settings -> Models
    would change every other copilot surface and silently not this one, for ever.
    Cheap enough to run on the hot path (`get_or_create_assistant` is called on
    every web AND Telegram interaction) -- one indexed lookup, and an assignment
    only when the answer actually moved.
    """
    wanted = await _select_model_config(db, tenant_id=tenant_id)
    # `wanted is None` means the tenant has no usable ModelConfig at all --
    # never clear a working pin because of a transiently empty answer.
    if wanted is None or wanted == agent.model_config_id:
        return
    agent.model_config_id = wanted
    await db.flush()


async def get_or_create_assistant(db: AsyncSession, *, tenant_id: uuid.UUID) -> m.Agent:
    existing = await _load_assistant(db, tenant_id=tenant_id)
    if existing is not None:
        await _sync_model_config(db, existing, tenant_id=tenant_id)
        return existing

    # Create inside a SAVEPOINT, exactly as `runtime/intake.enqueue_run` does
    # for `idempotency_key`. There are five concurrent first-call entrypoints
    # now (GET /assistant, the three /chat/sessions... routes via
    # `_assistant_visible`, and `bind_from_free_text`), so two tabs on a fresh
    # tenant genuinely race here. The partial unique index from migration 0082
    # turns the loser into an IntegrityError; only the savepoint unwinds, which
    # leaves the outer transaction -- and its transaction-local RLS tenant
    # binding -- alive, so the follow-up SELECT can still see the winner.
    try:
        async with db.begin_nested():
            dept = m.Department(
                tenant_id=tenant_id,
                name=ASSISTANT_DEPARTMENT_NAME,
                goal="",
                is_assistant_department=True,
            )
            db.add(dept)
            await db.flush()

            agent = m.Agent(
                tenant_id=tenant_id,
                department_id=dept.id,
                name=ASSISTANT_NAME,
                mission=_MISSION,
                is_team_lead=True,
                is_tenant_assistant=True,
                status="running",
                model_config_id=await _select_model_config(db, tenant_id=tenant_id),
            )
            db.add(agent)
            await db.flush()

            # Same convention as capas/service.py's department-with-a-lead
            # creation: collab/intake.py's route_to_team_lead reads
            # dept.team_lead_agent_id and treats None as "no lead -- drop the
            # work silently," so the department this Assistant just got created
            # in must point back at it.
            dept.team_lead_agent_id = agent.id
            await db.flush()
    except IntegrityError:
        # The savepoint took the half-built Department down with the Agent, so
        # the loser leaves nothing behind -- it just adopts the winner's row.
        winner = await _load_assistant(db, tenant_id=tenant_id)
        if winner is None:  # pragma: no cover - not the index we raced on
            raise
        await _sync_model_config(db, winner, tenant_id=tenant_id)
        return winner
    return agent

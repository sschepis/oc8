"""Turn a human's decision into work the agent performs.

The agent that raised the decision did not wait for it -- see
`control_tools.REQUEST_DECISION` -- so nothing is sitting there to hand it to.
Without this the inbox would be a dead end: the operator clicks, the row says
"approved", and the customer waits forever.

The decision therefore becomes a fresh run through the ordinary intake path, the
same one a cron trigger or an operator uses. That has a consequence worth
stating: the new run starts a NEW conversation and remembers nothing, so the
instruction has to carry its own context. Hence the original question and basis
travel with it, not just "approved".
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from oc8 import models as m
from oc8.runtime.intake import enqueue_run


class UnknownOption(ValueError):
    """The chosen option is not one the agent offered."""


def option_labels(approval: m.ApprovalRequest) -> dict[str, dict[str, Any]]:
    raw = (approval.payload or {}).get("options") or []
    return {str(o["key"]): o for o in raw if isinstance(o, dict) and o.get("key")}


def instruction_for(approval: m.ApprovalRequest, *, decision: str, reason: str) -> str:
    """What the agent is told to do now.

    Written as an instruction, not as a record: the agent reads this cold, with
    no memory of having asked.
    """
    lines = [
        "A human has made the decision you asked for. Carry it out now.",
        "",
        f"YOUR QUESTION WAS: {approval.title}",
    ]
    if approval.detail:
        lines += ["", f"WHAT IT WAS ABOUT: {approval.detail}"]

    chosen = option_labels(approval).get(approval.decision_option or "")
    if decision == "reject":
        lines += ["", "DECISION: REJECTED — do NOT perform the proposed action."]
    elif chosen is not None:
        detail = f" ({chosen['detail']})" if chosen.get("detail") else ""
        lines += ["", f"DECISION: {chosen['label']}{detail}"]
    else:
        lines += ["", "DECISION: APPROVED."]
    if reason:
        # Last, so it overrides a chosen option it contradicts: a human who
        # writes an instruction means it more than a button they also pressed.
        lines += ["", f"HUMAN INSTRUCTION (overrides everything else): {reason}"]

    lines += [
        "",
        "Carry out exactly that and then inform the customer in the chat. "
        "Do not repeat steps you already completed.",
    ]
    return "\n".join(lines)


async def carry_out_decision(
    db: AsyncSession,
    *,
    approval: m.ApprovalRequest,
    decision: str,
    reason: str,
    option: str | None,
) -> uuid.UUID:
    """Record which option was picked and enqueue the run that acts on it.

    COMMITS `db` (via `enqueue_run`); see its docstring.
    """
    options = option_labels(approval)
    if option is not None:
        if option not in options:
            raise UnknownOption(option)
        approval.decision_option = option
    await db.flush()

    run, _published = await enqueue_run(
        db,
        tenant_id=approval.tenant_id,
        agent_id=approval.agent_id,
        context={"task": instruction_for(approval, decision=decision, reason=reason)},
        source="decision",
        # One run per decision, forever: a double-click on approve must not have
        # the agent act twice on an external system.
        idempotency_key=f"decision:{approval.id}",
    )
    return run.id

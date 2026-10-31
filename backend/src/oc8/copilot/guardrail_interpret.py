"""Single-shot LLM translation from a free-text guardrail definition into the
generic 4-state policy shape (`Self-sufficient` / `With limits` / `Approval
required` / `Not allowed`) the guardrails UI and `authz/pdp.py` both speak --
`With limits` decomposing into structured `Condition`s (`authz/pdp.py`), never
a euro-only special case.

Deliberately NOT a chat turn and NOT the tenant Assistant: this never sees a
conversation, never calls `propose_change`, and never writes anything -- it
answers exactly one structured question and returns, ONCE, at authoring time.
The result is only ever a pre-filled suggestion an operator still reviews
(and must explicitly accept) inside the normal narrowing editor and saves
through the existing `PUT /agents/{id}/narrowing` path; `authz/pdp.py` is
never touched from here, and free text is never stored or re-interpreted --
only the structured `Condition`s this module derives are. Fails closed
(`GuardrailNotUnderstood`) rather than guessing whenever the model doesn't
call the tool or names something this module doesn't recognise.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Literal

from sqlalchemy.ext.asyncio import AsyncSession

from oc8 import models as m
from oc8.authz.pdp import Condition, ConditionDatatype, ConditionOperator, Effect
from oc8.capas.manifest import GuardrailAttribute
from oc8.config import get_settings
from oc8.modelrouter.fallback import complete_with_fallback
from oc8.modelrouter.router import get_model_router
from oc8.modelrouter.types import ModelParams, NeutralMessage, NeutralTool

Decision = Literal["self_sufficient", "with_limits", "approval_required", "not_allowed"]

_DECISIONS: tuple[Decision, ...] = (
    "self_sufficient",
    "with_limits",
    "approval_required",
    "not_allowed",
)
_OPERATORS = tuple(o.value for o in ConditionOperator)
_THEN_VALUES = (Effect.REQUIRE_APPROVAL.value, Effect.DENY.value)


class GuardrailNotUnderstood(ValueError):
    """The free-text definition could not be mapped to a supported guardrail
    shape. Value-free like `copilot.capabilities.InvalidOperation` -- the
    definition text itself is never echoed back in an error."""


@dataclass(frozen=True)
class GuardrailInterpretation:
    decision: Decision
    #: Only non-empty when `decision == "with_limits"`. Each entry is already
    #: a valid `authz.pdp.Condition` -- `attribute` checked against the
    #: function's declared `GuardrailAttribute`s and `value` coerced against
    #: that attribute's own `datatype`, never trusted from the model as-is.
    conditions: list[Condition]


def attributes_for_function(
    attributes: list[GuardrailAttribute], function: str
) -> list[GuardrailAttribute]:
    return [a for a in attributes if not a.tools or function in a.tools]


def _build_tool(attributes: list[GuardrailAttribute]) -> NeutralTool:
    decisions = list(_DECISIONS) if attributes else [d for d in _DECISIONS if d != "with_limits"]
    condition_schema: dict[str, Any] = {
        "type": "array",
        "description": (
            "Only set when decision is 'with_limits'. Each entry: IF "
            "attribute OPERATOR value THEN then. Every attribute value the "
            "definition doesn't name falls through to allowed -- never add a "
            "condition for the 'otherwise' case."
        ),
        "items": {
            "type": "object",
            "properties": {
                "attribute": {
                    "type": "string",
                    "enum": [a.key for a in attributes] or [""],
                },
                "operator": {"type": "string", "enum": list(_OPERATORS)},
                "value": {
                    "description": (
                        "The comparison value, typed to match the chosen attribute "
                        "(a number, a string/enum member, or an array of those for "
                        "'in'/'not_in')."
                    )
                },
                "then": {
                    "type": "string",
                    "enum": list(_THEN_VALUES),
                    "description": (
                        "'require_approval': a human must approve a call whose "
                        "value matches. 'deny': such a call is blocked outright."
                    ),
                },
            },
            "required": ["attribute", "operator", "value", "then"],
        },
    }
    return NeutralTool(
        name="set_guardrail_interpretation",
        description=(
            "Report the one guardrail shape that best matches the operator's "
            "free-text definition. Call this exactly once, with your best "
            "reading -- never ask a follow-up question, there is nobody who "
            "can answer one."
        ),
        parameters={
            "type": "object",
            "properties": {
                "decision": {
                    "type": "string",
                    "enum": decisions,
                    "description": (
                        "'not_allowed': the function must never run. "
                        "'approval_required': the function may run but a human "
                        "must approve EVERY call, unconditionally. 'with_limits': "
                        "the function runs on its own up to some limit named in "
                        "the definition, above which conditions apply -- express "
                        "that limit as one or more entries in 'conditions'. "
                        "'self_sufficient': no restriction beyond what the "
                        "department already permits."
                    ),
                },
                "conditions": condition_schema,
            },
            "required": ["decision", "conditions"],
        },
    )


def _build_prompt(
    *, connection_name: str, function: str, definition: str, attributes: list[GuardrailAttribute]
) -> str:
    prompt = (
        "An operator describes a guardrail in free text for the tool "
        f"'{function}' on connection '{connection_name}':\n\n"
        f'"{definition}"\n\n'
        "Map it to the one matching rule shape by calling "
        "set_guardrail_interpretation exactly once."
    )
    if attributes:
        catalog = ", ".join(f"{a.key} ({a.datatype})" for a in attributes)
        prompt += (
            f" For 'with_limits' only these attributes are available: "
            f"{catalog}. Do not invent another attribute."
        )
    else:
        prompt += (
            " This function declares no comparable attributes -- 'with_limits' "
            "cannot be used, even if the definition mentions a number; choose "
            "'approval_required' or 'not_allowed' instead."
        )
    return prompt


def _coerce_value(raw: Any, datatype: str, operator: str) -> Any:
    is_set_op = operator in (ConditionOperator.IN.value, ConditionOperator.NOT_IN.value)
    if is_set_op:
        if not isinstance(raw, list) or not raw:
            raise GuardrailNotUnderstood()
        return [_coerce_scalar(v, datatype) for v in raw]
    return _coerce_scalar(raw, datatype)


def _coerce_scalar(raw: Any, datatype: str) -> Any:
    if datatype == ConditionDatatype.NUMBER.value:
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise GuardrailNotUnderstood()
        return raw
    if datatype == ConditionDatatype.BOOLEAN.value:
        if not isinstance(raw, bool):
            raise GuardrailNotUnderstood()
        return raw
    # string / enum
    if not isinstance(raw, str) or not raw:
        raise GuardrailNotUnderstood()
    return raw


def parse_conditions(raw_conditions: Any, attributes: list[GuardrailAttribute]) -> list[Condition]:
    if not isinstance(raw_conditions, list) or not raw_conditions:
        raise GuardrailNotUnderstood()
    by_key = {a.key: a for a in attributes}
    out: list[Condition] = []
    for raw in raw_conditions:
        if not isinstance(raw, dict):
            raise GuardrailNotUnderstood()
        attribute = raw.get("attribute")
        operator = raw.get("operator")
        then = raw.get("then")
        if (
            not isinstance(attribute, str)
            or attribute not in by_key
            or operator not in _OPERATORS
            or then not in _THEN_VALUES
        ):
            raise GuardrailNotUnderstood()
        spec = by_key[attribute]
        if spec.datatype == ConditionDatatype.ENUM.value:
            values = raw.get("value")
            candidates = values if isinstance(values, list) else [values]
            if any(v not in spec.enum_values for v in candidates):
                raise GuardrailNotUnderstood()
        value = _coerce_value(raw.get("value"), spec.datatype, operator)
        out.append(
            Condition(
                attribute=attribute,
                datatype=spec.datatype,
                operator=operator,
                value=tuple(value) if isinstance(value, list) else value,
                then=Effect(then),
            )
        )
    return out


#: A batch entry never reports `self_sufficient` explicitly -- omitting a
#: function from the model's answer already means "leave it at the default".
_BATCH_DECISIONS: tuple[Decision, ...] = tuple(d for d in _DECISIONS if d != "self_sufficient")


@dataclass(frozen=True)
class FunctionGuardrailInterpretation:
    function: str
    decision: Decision
    conditions: list[Condition]


def _build_batch_tool(tool_catalog: list[str], attributes: list[GuardrailAttribute]) -> NeutralTool:
    decisions = (
        list(_BATCH_DECISIONS)
        if attributes
        else [d for d in _BATCH_DECISIONS if d != "with_limits"]
    )
    condition_schema: dict[str, Any] = {
        "type": "array",
        "description": (
            "Only set when this entry's decision is 'with_limits'. Each item: "
            "IF attribute OPERATOR value THEN then."
        ),
        "items": {
            "type": "object",
            "properties": {
                "attribute": {"type": "string", "enum": [a.key for a in attributes] or [""]},
                "operator": {"type": "string", "enum": list(_OPERATORS)},
                "value": {"description": "The comparison value, typed to match the attribute."},
                "then": {"type": "string", "enum": list(_THEN_VALUES)},
            },
            "required": ["attribute", "operator", "value", "then"],
        },
    }
    return NeutralTool(
        name="set_guardrail_interpretations",
        description=(
            "Report every function whose access should be restricted, based on "
            "the agent's own instructions. Omit any function you would leave "
            "unrestricted -- that already means self-sufficient, no entry "
            "needed for it. Call this exactly once, with your best reading; "
            "never ask a follow-up question, there is nobody who can answer one."
        ),
        parameters={
            "type": "object",
            "properties": {
                "assignments": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "function": {"type": "string", "enum": tool_catalog},
                            "decision": {
                                "type": "string",
                                "enum": decisions,
                                "description": (
                                    "'not_allowed': must never run. "
                                    "'approval_required': may run but a human must "
                                    "approve EVERY call, unconditionally. "
                                    "'with_limits': runs on its own up to some limit "
                                    "named in the instructions, expressed as one or "
                                    "more entries in 'conditions'."
                                ),
                            },
                            "conditions": condition_schema,
                        },
                        "required": ["function", "decision", "conditions"],
                    },
                },
            },
            "required": ["assignments"],
        },
    )


def _build_batch_prompt(
    *,
    connection_name: str,
    instructions: str,
    tool_catalog: list[str],
    attributes: list[GuardrailAttribute],
) -> str:
    functions = ", ".join(sorted(tool_catalog))
    prompt = (
        "Ein Agent hat die folgende Instruction/Mission:\n\n"
        f'"{instructions}"\n\n'
        f"Er hat Zugriff auf die Verbindung '{connection_name}' mit diesen Funktionen: "
        f"{functions}.\n\n"
        "For every function whose use should be restricted by the instruction "
        "(for example because it does not appear there, a limit is mentioned, or "
        "approval seems appropriate), propose a rule by calling "
        "set_guardrail_interpretations exactly once. Omit every function that "
        "should remain freely usable."
    )
    if attributes:
        catalog = ", ".join(f"{a.key} ({a.datatype})" for a in attributes)
        prompt += (
            f" For 'with_limits' only these attributes are available: "
            f"{catalog}. Do not invent another attribute, and use only the "
            "attributes that actually belong to each function."
        )
    else:
        prompt += (
            " No function on this connection declares comparable attributes -- "
            "'with_limits' cannot be used for any of them."
        )
    return prompt


def parse_batch_assignments(
    raw_assignments: Any, tool_catalog: list[str], guardrail_attributes: list[GuardrailAttribute]
) -> list[FunctionGuardrailInterpretation]:
    """Parses a batch tool-call's `assignments`, dropping any single entry
    that doesn't check out (unknown function, unsupported decision, or a
    `with_limits` whose conditions don't resolve) rather than failing the
    whole batch -- one bad entry shouldn't cost the operator every other
    good suggestion the model made in the same call."""
    if not isinstance(raw_assignments, list):
        raise GuardrailNotUnderstood()
    catalog = set(tool_catalog)
    seen: set[str] = set()
    out: list[FunctionGuardrailInterpretation] = []
    for raw in raw_assignments:
        if not isinstance(raw, dict):
            continue
        function = raw.get("function")
        decision = raw.get("decision")
        if not isinstance(function, str) or function not in catalog or function in seen:
            continue
        if decision not in _BATCH_DECISIONS:
            continue
        attributes = attributes_for_function(guardrail_attributes, function)
        if decision == "with_limits":
            if not attributes:
                continue
            try:
                conditions = parse_conditions(raw.get("conditions"), attributes)
            except GuardrailNotUnderstood:
                continue
        else:
            conditions = []
        seen.add(function)
        out.append(
            FunctionGuardrailInterpretation(
                function=function, decision=decision, conditions=conditions
            )
        )
    return out


async def interpret_guardrails_from_instruction(
    db: AsyncSession,
    *,
    tenant_id: uuid.UUID,
    agent: m.Agent,
    connection_name: str,
    tool_catalog: list[str],
    guardrail_attributes: list[GuardrailAttribute],
) -> list[FunctionGuardrailInterpretation]:
    """Same one-shot, single-LLM-call contract as `interpret_guardrail_definition`,
    but reads the agent's own instructions (`agent.mission`) instead of an
    operator-typed definition, and proposes rules for every function of one
    connection in a single call rather than one function at a time. An empty
    result is a legitimate answer ("nothing here needs restricting"); only a
    failed/absent tool call raises `GuardrailNotUnderstood`."""
    instructions = (agent.mission or "").strip()
    if not instructions or not tool_catalog:
        raise GuardrailNotUnderstood()

    primary: m.ModelConfig | None = None
    if agent.model_config_id is not None:
        primary = await db.get(m.ModelConfig, agent.model_config_id)

    settings = get_settings()
    try:
        result = await complete_with_fallback(
            db,
            get_model_router(),
            tenant_id=tenant_id,
            agent_id=agent.id,
            primary=primary,
            no_config_provider=settings.default_model_provider,
            no_config_model=settings.default_model,
            messages=[
                NeutralMessage(
                    role="user",
                    content=_build_batch_prompt(
                        connection_name=connection_name,
                        instructions=instructions,
                        tool_catalog=tool_catalog,
                        attributes=guardrail_attributes,
                    ),
                )
            ],
            tools=[_build_batch_tool(tool_catalog, guardrail_attributes)],
            params=ModelParams(max_tokens=1024),
            request_id=uuid.uuid4(),
            contains_restricted=False,
        )
    except Exception as exc:
        raise GuardrailNotUnderstood() from exc

    call = next((c for c in result.tool_calls if c.name == "set_guardrail_interpretations"), None)
    if call is None:
        raise GuardrailNotUnderstood()
    return parse_batch_assignments(
        call.arguments.get("assignments"), tool_catalog, guardrail_attributes
    )


async def interpret_guardrail_definition(
    db: AsyncSession,
    *,
    tenant_id: uuid.UUID,
    agent: m.Agent,
    connection_name: str,
    function: str,
    definition: str,
    guardrail_attributes: list[GuardrailAttribute],
) -> GuardrailInterpretation:
    definition = definition.strip()
    if not definition:
        raise GuardrailNotUnderstood()

    attributes = attributes_for_function(guardrail_attributes, function)

    primary: m.ModelConfig | None = None
    if agent.model_config_id is not None:
        primary = await db.get(m.ModelConfig, agent.model_config_id)

    settings = get_settings()
    try:
        result = await complete_with_fallback(
            db,
            get_model_router(),
            tenant_id=tenant_id,
            agent_id=agent.id,
            primary=primary,
            no_config_provider=settings.default_model_provider,
            no_config_model=settings.default_model,
            messages=[
                NeutralMessage(
                    role="user",
                    content=_build_prompt(
                        connection_name=connection_name,
                        function=function,
                        definition=definition,
                        attributes=attributes,
                    ),
                )
            ],
            tools=[_build_tool(attributes)],
            params=ModelParams(max_tokens=512),
            request_id=uuid.uuid4(),
            contains_restricted=False,
        )
    except Exception as exc:
        raise GuardrailNotUnderstood() from exc

    call = next((c for c in result.tool_calls if c.name == "set_guardrail_interpretation"), None)
    if call is None:
        raise GuardrailNotUnderstood()
    decision = call.arguments.get("decision")
    if decision not in _DECISIONS:
        raise GuardrailNotUnderstood()
    if decision == "with_limits":
        if not attributes:
            raise GuardrailNotUnderstood()
        conditions = parse_conditions(call.arguments.get("conditions"), attributes)
    else:
        conditions = []
    return GuardrailInterpretation(decision=decision, conditions=conditions)

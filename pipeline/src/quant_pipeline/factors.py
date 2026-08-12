"""Auditable OHLCV factor candidates for controlled ETF experiments.

These are original research candidates assembled for this pipeline.  The label
does not assert that an economically similar signal has never been published or
traded.  Every expression is intentionally point-in-time: Qlib ``Ref`` offsets
must be positive, so only the current bar and already observed bars are used.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from numbers import Real
from typing import Any, Iterable, Literal


FactorDirection = Literal[-1, 1]

FACTOR_CATALOG_VERSION = "orc_ohlcv_v1"

FACTOR_FAMILIES = (
    "trend_crowding",
    "volume_impact",
    "compression_release",
    "price_volume_divergence",
    "session_structure",
)

_ALLOWED_FIELDS = {"open", "high", "low", "close", "volume"}
_ELEMENT_OPERATORS = {"Abs", "Sign", "Log"}
_PAIR_OPERATORS = {"Greater", "Less"}
_ROLLING_OPERATORS = {"Mean", "Sum", "Std", "Max", "Min", "Slope", "Rsquare"}
_PAIR_ROLLING_OPERATORS = {"Corr"}
_ALLOWED_OPERATORS = (
    _ELEMENT_OPERATORS | _PAIR_OPERATORS | _ROLLING_OPERATORS | _PAIR_ROLLING_OPERATORS | {"Ref"}
)
_NAME_PATTERN = re.compile(r"ORC_[A-Z0-9_]+\Z")
_FIELD_PATTERN = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")


@dataclass(frozen=True, slots=True)
class FactorDefinition:
    """A raw Qlib feature and its testable prior.

    ``direction`` is the expected monotonic relationship with the next-period
    return: ``1`` means larger values are expected to rank higher and ``-1``
    means the inverse.  It is metadata for diagnostics, not a transformation
    applied to the feature.  ``lookback`` is a conservative warm-up in prior
    bars and must cover the maximum historical lag required by the expression.
    """

    name: str
    family: str
    expression: str
    direction: FactorDirection
    hypothesis: str
    lookback: int


@dataclass(frozen=True, slots=True)
class FactorResearchProtocol:
    """Predeclared controls that separate discovery from confirmation."""

    protocol_version: str
    catalog_version: str
    stage_order: tuple[str, ...]
    discovery_scope: str
    primary_metric: str
    multiplicity_control: str
    locked_holdout_policy: str
    required_robustness_checks: tuple[str, ...]


RESEARCH_PROTOCOL = FactorResearchProtocol(
    protocol_version="purged_family_ablation_v1",
    catalog_version=FACTOR_CATALOG_VERSION,
    stage_order=("family_ablation", "candidate_confirmation", "locked_holdout"),
    discovery_scope="purged rolling train and validation folds only",
    primary_metric="mean validation daily rank IC in the declared direction",
    multiplicity_control="Benjamini-Hochberg false-discovery rate at q=0.10 within the frozen catalog",
    locked_holdout_policy=(
        "Do not inspect or tune on the final chronological holdout until families, candidates, model, and strategy "
        "parameters are frozen; evaluate that holdout once."
    ),
    required_robustness_checks=(
        "same sign in a majority of validation folds",
        "incremental result versus the unchanged Alpha158 baseline",
        "positive net excess at the configured 10 bps slippage stress",
        "no material concentration in one instrument or one fold",
        "research-only classification while the ETF universe is not point-in-time",
    ),
)


ORIGINAL_RESEARCH_CANDIDATES: tuple[FactorDefinition, ...] = (
    FactorDefinition(
        name="ORC_TREND_PATH_CROWD_20",
        family="trend_crowding",
        expression=(
            "($close/Ref($close,20)-1)*Abs($close/Ref($close,20)-1)"
            "/(Sum(Abs($close/Ref($close,1)-1),20)+1e-12)"
            "*(Mean($volume,5)/(Mean($volume,20)+1e-12))"
        ),
        direction=-1,
        hypothesis=(
            "A directionally efficient 20-bar move with accelerating participation is crowded and is more likely "
            "to partially unwind."
        ),
        lookback=20,
    ),
    FactorDefinition(
        name="ORC_TREND_ACCEL_UNCROWDED_5_20",
        family="trend_crowding",
        expression=(
            "(Slope($close,5)/(Mean($close,5)+1e-12)"
            "-Slope($close,20)/(Mean($close,20)+1e-12))*(1-Rsquare($close,20))"
        ),
        direction=1,
        hypothesis=(
            "Fast trend acceleration that is not yet explained by a highly linear slow trend represents a fresh "
            "rotation rather than a mature consensus trade."
        ),
        lookback=20,
    ),
    FactorDefinition(
        name="ORC_TREND_STREAK_EXHAUST_12",
        family="trend_crowding",
        expression=(
            "Mean(Sign($close/Ref($close,1)-1),12)*Rsquare($close,12)"
            "*($volume/(Mean($volume,12)+1e-12))"
        ),
        direction=-1,
        hypothesis=(
            "A one-sided, linear return streak accompanied by a current volume spike is late-stage demand or "
            "supply and should mean revert."
        ),
        lookback=12,
    ),
    FactorDefinition(
        name="ORC_SIGNED_IMPACT_SHOCK_20",
        family="volume_impact",
        expression=(
            "($close/Ref($close,1)-1)*($volume/(Mean(Ref($volume,1),20)+1e-12))"
            "*Abs($close-$open)/($high-$low+1e-12)"
        ),
        direction=-1,
        hypothesis=(
            "A large signed close-to-close move, delivered as a full-bodied bar on exceptional volume, contains "
            "transient market impact that should decay."
        ),
        lookback=20,
    ),
    FactorDefinition(
        name="ORC_RETURN_PER_VOLUME_SURPRISE_20",
        family="volume_impact",
        expression=(
            "($close/Ref($close,1)-1)"
            "/(Abs($volume/(Mean(Ref($volume,1),20)+1e-12)-1)+0.05)"
        ),
        direction=1,
        hypothesis=(
            "Directional price discovery achieved without an extreme volume dislocation is less impact-driven and "
            "is more likely to persist."
        ),
        lookback=20,
    ),
    FactorDefinition(
        name="ORC_IMPACT_CONCENTRATION_20",
        family="volume_impact",
        expression=(
            "Corr(Abs($close/Ref($close,1)-1),Log($volume+1),20)"
            "*(Std($volume,20)/(Mean($volume,20)+1e-12))*Sign($close/Ref($close,5)-1)"
        ),
        direction=-1,
        hypothesis=(
            "A signed short trend is fragile when absolute returns are tightly coupled to volume and volume itself "
            "is unstable, indicating concentrated liquidity impact."
        ),
        lookback=20,
    ),
    FactorDefinition(
        name="ORC_RANGE_RELEASE_5_20",
        family="compression_release",
        expression=(
            "(($close-$open)/($high-$low+1e-12))"
            "*(($high-$low)/(Mean(Ref($high-$low,1),20)+1e-12))"
            "*(Mean(Ref($high-$low,1),20)/(Mean(Ref($high-$low,1),5)+1e-12))"
        ),
        direction=1,
        hypothesis=(
            "A directional body that expands out of a five-bar range squeeze relative to the prior 20 bars begins "
            "a short-lived volatility release."
        ),
        lookback=20,
    ),
    FactorDefinition(
        name="ORC_VOLATILITY_RELEASE_20",
        family="compression_release",
        expression=(
            "($close/Ref($close,1)-1)"
            "/(Std(Ref($close/Ref($close,1)-1,1),20)+1e-12)"
        ),
        direction=1,
        hypothesis=(
            "A return that escapes the volatility distribution of the preceding 20 completed returns tends to "
            "continue briefly in its release direction."
        ),
        lookback=21,
    ),
    FactorDefinition(
        name="ORC_COIL_EDGE_5_20",
        family="compression_release",
        expression=(
            "(2*($close-Min($low,20))/(Max($high,20)-Min($low,20)+1e-12)-1)"
            "*(1-Mean($high-$low,5)/(Mean($high-$low,20)+1e-12))"
        ),
        direction=1,
        hypothesis=(
            "During short-range compression, a close parked near a 20-bar edge identifies the likely direction of "
            "the eventual range expansion."
        ),
        lookback=20,
    ),
    FactorDefinition(
        name="ORC_WICK_RELEASE_20",
        family="compression_release",
        expression=(
            "(2*$close-$high-$low)/($high-$low+1e-12)"
            "*(($high-$low)/(Mean(Ref($high-$low,1),20)+1e-12))"
        ),
        direction=1,
        hypothesis=(
            "A range expansion that closes toward one edge rather than leaving a rejection wick carries follow-on "
            "pressure in the close-location direction."
        ),
        lookback=20,
    ),
    FactorDefinition(
        name="ORC_TREND_VOLUME_DIVERGENCE_10_20",
        family="price_volume_divergence",
        expression=(
            "($close/Ref($close,10)-1)"
            "*(1-$volume/(Mean(Ref($volume,1),20)+1e-12))"
        ),
        direction=-1,
        hypothesis=(
            "A 10-bar price move unsupported by current participation relative to prior volume is vulnerable to a "
            "symmetric reversal."
        ),
        lookback=20,
    ),
    FactorDefinition(
        name="ORC_VOLUME_WEIGHTED_DIRECTION_10",
        family="price_volume_divergence",
        expression=(
            "Mean(Sign($close/Ref($close,1)-1)*Log($volume+1),10)"
            "/(Mean(Log($volume+1),10)+1e-12)"
            "-Mean(Sign($close/Ref($close,1)-1),10)"
        ),
        direction=1,
        hypothesis=(
            "The gap between volume-weighted return direction and the unweighted up/down hit rate isolates "
            "participation by the more informative bars."
        ),
        lookback=10,
    ),
    FactorDefinition(
        name="ORC_LAGGED_VOLUME_LEAD_20",
        family="price_volume_divergence",
        expression=(
            "Corr(Ref($volume/Ref($volume,1)-1,1),$close/Ref($close,1)-1,20)"
            "*($volume/Ref($volume,1)-1)"
        ),
        direction=1,
        hypothesis=(
            "When an ETF has recently transmitted volume changes into next-bar returns, the current volume impulse "
            "provides a signed lead signal."
        ),
        lookback=21,
    ),
    FactorDefinition(
        name="ORC_PV_COHERENCE_SHIFT_5_20",
        family="price_volume_divergence",
        expression=(
            "($close/Ref($close,5)-1)"
            "*(Corr($close/Ref($close,1)-1,$volume/Ref($volume,1)-1,5)"
            "-Corr($close/Ref($close,1)-1,$volume/Ref($volume,1)-1,20))"
        ),
        direction=1,
        hypothesis=(
            "A recent increase in price-volume coherence confirms a five-bar trend before that relationship is "
            "visible in its slower baseline."
        ),
        lookback=20,
    ),
    FactorDefinition(
        name="ORC_GAP_CONFIRMATION_20",
        family="session_structure",
        expression=(
            "(($open/Ref($close,1)-1)+($close/$open-1))"
            "*(1+Sign(($open/Ref($close,1)-1)*($close/$open-1)))/2"
            "*Abs($open/Ref($close,1)-1)/(Std($close/Ref($close,1)-1,20)+1e-12)"
        ),
        direction=1,
        hypothesis=(
            "An opening gap confirmed by the intraday move is information arrival rather than a temporary ETF "
            "premium or discount and should continue in the combined direction."
        ),
        lookback=20,
    ),
    FactorDefinition(
        name="ORC_GAP_ABSORPTION_20",
        family="session_structure",
        expression=(
            "($close/$open-1)*Abs($open/Ref($close,1)-1)"
            "/(Std(Ref($close/Ref($close,1)-1,1),20)+1e-12)"
        ),
        direction=1,
        hypothesis=(
            "The intraday direction after a volatility-scaled opening dislocation reveals which side absorbed the "
            "gap and is expected to persist."
        ),
        lookback=21,
    ),
    FactorDefinition(
        name="ORC_OVERNIGHT_DOMINANCE_20",
        family="session_structure",
        expression=(
            "(($open/Ref($close,1)-1)-($close/$open-1))"
            "/(Std($close/Ref($close,1)-1,20)+1e-12)"
        ),
        direction=-1,
        hypothesis=(
            "Returns concentrated at the open but not sustained intraday are more consistent with a temporary ETF "
            "premium or discount and should fade."
        ),
        lookback=20,
    ),
    FactorDefinition(
        name="ORC_CLOSE_PRESSURE_VOLUME_20",
        family="session_structure",
        expression=(
            "(2*$close-$high-$low)/($high-$low+1e-12)"
            "*($volume/(Mean(Ref($volume,1),20)+1e-12)-1)"
        ),
        direction=1,
        hypothesis=(
            "An edge-of-range close backed by unusual volume proxies late-session pressure that carries into the "
            "next decision period."
        ),
        lookback=20,
    ),
    FactorDefinition(
        name="ORC_TREND_RSQR_STRESS_20",
        family="trend_crowding",
        expression=(
            "Rsquare($close,20)"
            "*(Std($close/Ref($close,1)-1,5)/(Std($close/Ref($close,1)-1,20)+1e-12))"
        ),
        direction=-1,
        hypothesis=(
            "A highly linear 20-bar trend with unusually volatile short returns is crowded and is more likely to "
            "partially unwind than to continue cleanly."
        ),
        lookback=20,
    ),
    FactorDefinition(
        name="ORC_TREND_ACCEL_GAP_10_30",
        family="trend_crowding",
        expression=(
            "(Slope($close,10)/$close)/(Slope($close,30)/$close+1e-12)-1"
        ),
        direction=-1,
        hypothesis=(
            "A short-term normalized trend running far ahead of its medium-term trend is late acceleration rather "
            "than fresh trend strength and is more likely to revert."
        ),
        lookback=30,
    ),
    FactorDefinition(
        name="ORC_VOLUME_CLIMAX_10",
        family="volume_impact",
        expression=(
            "($volume/Mean(Ref($volume,1),20))"
            "*(2*$close-$high-$low)/($high-$low+1e-12)"
        ),
        direction=-1,
        hypothesis=(
            "Exceptional volume combined with a lower-close candle indicates absorption or distribution that is "
            "more likely to be followed by a short-term fade."
        ),
        lookback=20,
    ),
    FactorDefinition(
        name="ORC_VOLUME_STABILITY_TREND_20",
        family="volume_impact",
        expression=(
            "Mean($close/Ref($close,1)-1,20)"
            "/(Std($volume,20)/(Mean($volume,20)+1e-12)+1e-12)"
        ),
        direction=1,
        hypothesis=(
            "A directional return trend achieved with stable, orderly volume is more sustainable than the same "
            "trend delivered with unstable participation."
        ),
        lookback=20,
    ),
    FactorDefinition(
        name="ORC_NET_VOLUME_PRESSURE_20",
        family="price_volume_divergence",
        expression=(
            "(Sum(Greater($close,Ref($close,1))*$volume,20)"
            "-Sum(Greater(Ref($close,1),$close)*$volume,20))"
            "/(Sum($volume,20)+1e-12)"
        ),
        direction=1,
        hypothesis=(
            "Up-bar volume dominance over down-bar volume is a signed participation measure; positive net pressure "
            "is expected to precede positive returns."
        ),
        lookback=20,
    ),
    FactorDefinition(
        name="ORC_VOLUME_TREND_DIVERGENCE_10",
        family="price_volume_divergence",
        expression=(
            "($close/Ref($close,10)-1)"
            "*(1-$volume/(Mean(Ref($volume,1),10)+1e-12))"
        ),
        direction=-1,
        hypothesis=(
            "A 10-bar price move unsupported by current participation relative to its own 10-bar volume profile is "
            "vulnerable to a symmetric reversal."
        ),
        lookback=10,
    ),
    FactorDefinition(
        name="ORC_INTRADAY_RANGE_RETURN_10",
        family="session_structure",
        expression=(
            "Mean(($close-$open)/$open,10)"
            "/(Std(($close-$open)/$open,20)+1e-12)"
        ),
        direction=1,
        hypothesis=(
            "A stable positive intraday drift relative to the volatility of its own recent open-to-close moves "
            "reflects persistent session-level buying pressure."
        ),
        lookback=20,
    ),
)


def _literal_integer(node: ast.AST, *, context: str) -> int:
    sign = 1
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        sign = -1 if isinstance(node.op, ast.USub) else 1
        node = node.operand
    if not isinstance(node, ast.Constant) or isinstance(node.value, bool) or not isinstance(node.value, Real):
        raise ValueError(f"{context} must be a finite integer literal")
    value = float(node.value) * sign
    if not math.isfinite(value) or not value.is_integer():
        raise ValueError(f"{context} must be a finite integer literal")
    return int(value)


def _parse_expression(expression: str, *, factor_name: str) -> ast.Expression:
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError(f"{factor_name}: expression must be a non-empty string")
    parsed_text = _FIELD_PATTERN.sub(lambda match: f"FIELD_{match.group(1).lower()}", expression)
    try:
        tree = ast.parse(parsed_text, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"{factor_name}: invalid Qlib expression syntax") from exc

    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Call,
        ast.Name,
        ast.Load,
        ast.Constant,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.UAdd,
        ast.USub,
    )
    call_function_nodes = {id(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}
    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            raise ValueError(f"{factor_name}: unsupported expression element {type(node).__name__}")
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, Real):
                raise ValueError(f"{factor_name}: expression constants must be numeric")
            if not math.isfinite(float(node.value)):
                raise ValueError(f"{factor_name}: expression contains a non-finite constant")
        if isinstance(node, ast.Name):
            if node.id.startswith("FIELD_"):
                field = node.id.removeprefix("FIELD_")
                if field not in _ALLOWED_FIELDS:
                    raise ValueError(f"{factor_name}: unsupported source field ${field}")
            elif node.id not in _ALLOWED_OPERATORS:
                raise ValueError(f"{factor_name}: unsupported operator or token {node.id}")
            elif id(node) not in call_function_nodes:
                raise ValueError(f"{factor_name}: operator {node.id} must be called")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_OPERATORS:
                operator = node.func.id if isinstance(node.func, ast.Name) else type(node.func).__name__
                raise ValueError(f"{factor_name}: unsupported operator {operator}")
            if node.keywords:
                raise ValueError(f"{factor_name}: Qlib operators do not accept keyword arguments here")
            operator = node.func.id
            expected_args = 1 if operator in _ELEMENT_OPERATORS else 2
            if operator in _PAIR_ROLLING_OPERATORS:
                expected_args = 3
            if len(node.args) != expected_args:
                raise ValueError(f"{factor_name}: {operator} requires {expected_args} arguments")
            if operator == "Ref":
                offset = _literal_integer(node.args[1], context=f"{factor_name}: Ref offset")
                if offset < 0:
                    raise ValueError(f"{factor_name}: future Ref offset {offset} is forbidden")
                if offset == 0:
                    raise ValueError(f"{factor_name}: Ref offset zero is ambiguous; use the field directly")
            elif operator in _ROLLING_OPERATORS | _PAIR_ROLLING_OPERATORS:
                window = _literal_integer(node.args[-1], context=f"{factor_name}: {operator} window")
                if window <= 0:
                    raise ValueError(f"{factor_name}: rolling windows must be positive")
    return tree


def _required_historical_lag(node: ast.AST) -> int:
    if isinstance(node, ast.Expression):
        return _required_historical_lag(node.body)
    if isinstance(node, (ast.Constant, ast.Name)):
        return 0
    if isinstance(node, ast.UnaryOp):
        return _required_historical_lag(node.operand)
    if isinstance(node, ast.BinOp):
        return max(_required_historical_lag(node.left), _required_historical_lag(node.right))
    if isinstance(node, ast.Call):
        operator = node.func.id
        if operator == "Ref":
            return _required_historical_lag(node.args[0]) + _literal_integer(
                node.args[1], context="Ref offset"
            )
        if operator in _ROLLING_OPERATORS:
            return _required_historical_lag(node.args[0]) + _literal_integer(
                node.args[1], context=f"{operator} window"
            ) - 1
        if operator in _PAIR_ROLLING_OPERATORS:
            child_lag = max(_required_historical_lag(node.args[0]), _required_historical_lag(node.args[1]))
            return child_lag + _literal_integer(node.args[2], context=f"{operator} window") - 1
        return max((_required_historical_lag(argument) for argument in node.args), default=0)
    raise TypeError(f"unsupported AST node: {type(node).__name__}")


def validate_factor_definitions(
    definitions: Iterable[FactorDefinition] = ORIGINAL_RESEARCH_CANDIDATES,
) -> tuple[FactorDefinition, ...]:
    """Validate factor metadata and point-in-time expression safety.

    The validated tuple is returned so callers can validate and consume an
    arbitrary candidate set in one operation.
    """

    validated = tuple(definitions)
    if not validated:
        raise ValueError("at least one factor definition is required")

    seen_names: set[str] = set()
    for factor in validated:
        if not isinstance(factor, FactorDefinition):
            raise TypeError("factor definitions must be FactorDefinition instances")
        if not isinstance(factor.name, str) or not _NAME_PATTERN.fullmatch(factor.name):
            raise ValueError(f"invalid factor name {factor.name!r}; expected ORC_[A-Z0-9_]+")
        canonical_name = factor.name.casefold()
        if canonical_name in seen_names:
            raise ValueError(f"duplicate factor name: {factor.name}")
        seen_names.add(canonical_name)
        if factor.family not in FACTOR_FAMILIES:
            raise ValueError(f"{factor.name}: unknown factor family {factor.family!r}")
        if not isinstance(factor.hypothesis, str) or not factor.hypothesis.strip():
            raise ValueError(f"{factor.name}: hypothesis must be a non-empty string")
        if isinstance(factor.direction, bool) or not isinstance(factor.direction, Real):
            raise ValueError(f"{factor.name}: direction must be finite and equal to -1 or 1")
        if not math.isfinite(float(factor.direction)) or factor.direction not in (-1, 1):
            raise ValueError(f"{factor.name}: direction must be finite and equal to -1 or 1")
        if isinstance(factor.lookback, bool) or not isinstance(factor.lookback, Real):
            raise ValueError(f"{factor.name}: lookback must be a finite positive integer")
        numeric_lookback = float(factor.lookback)
        if not math.isfinite(numeric_lookback) or not numeric_lookback.is_integer() or numeric_lookback <= 0:
            raise ValueError(f"{factor.name}: lookback must be a finite positive integer")

        tree = _parse_expression(factor.expression, factor_name=factor.name)
        required_lag = _required_historical_lag(tree)
        if int(numeric_lookback) < required_lag:
            raise ValueError(
                f"{factor.name}: lookback {factor.lookback} understates required historical lag {required_lag}"
            )
    return validated


def select_factor_definitions(families: Iterable[str] | None = None) -> tuple[FactorDefinition, ...]:
    """Return a stable subset for family-level ablation experiments."""

    candidates = validate_factor_definitions()
    if families is None:
        return candidates
    if isinstance(families, str):
        requested = (families,)
    else:
        requested = tuple(dict.fromkeys(families))
    if not requested:
        raise ValueError("at least one factor family is required")
    unknown = sorted(set(requested) - set(FACTOR_FAMILIES))
    if unknown:
        raise ValueError(f"unknown factor families: {unknown}")
    selected = tuple(factor for factor in candidates if factor.family in requested)
    return validate_factor_definitions(selected)


def factor_config(families: Iterable[str] | None = None) -> tuple[list[str], list[str]]:
    """Return Qlib ``(fields, names)`` lists for a loader configuration."""

    selected = select_factor_definitions(families)
    return [factor.expression for factor in selected], [factor.name for factor in selected]


def factors_by_family() -> dict[str, tuple[FactorDefinition, ...]]:
    """Expose deterministic ablation groups in the declared family order."""

    candidates = validate_factor_definitions()
    return {
        family: tuple(factor for factor in candidates if factor.family == family) for family in FACTOR_FAMILIES
    }


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)


def factor_catalog_manifest(families: Iterable[str] | None = None) -> dict[str, Any]:
    """Return serializable, fingerprinted metadata for a run manifest.

    The digest covers the selected definitions, their stable ordering, and the
    predeclared research protocol.  Changing an expression, prior, family, or
    protocol therefore creates a different experimental catalog identity.
    """

    selected = select_factor_definitions(families)
    payload = {
        "catalog_version": FACTOR_CATALOG_VERSION,
        "protocol": asdict(RESEARCH_PROTOCOL),
        "families": list(dict.fromkeys(factor.family for factor in selected)),
        "factors": [asdict(factor) for factor in selected],
    }
    canonical = _canonical_json(payload)
    payload = json.loads(canonical)
    payload["sha256"] = hashlib.sha256(canonical.encode("ascii")).hexdigest()
    return payload


def combined_alpha158_feature_config(
    families: Iterable[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Return Alpha158 followed by a validated original-candidate subset.

    Qlib is imported lazily so catalog validation and audit tooling remain
    usable without the compiled Qlib runtime.  Exact duplicate names are a hard
    error; this prevents silent column replacement if Alpha158 later adds a
    feature with a candidate's name.
    """

    from qlib.contrib.data.loader import Alpha158DL

    alpha_fields, alpha_names = Alpha158DL.get_feature_config(
        {
            "kbar": {},
            "price": {"windows": [0], "feature": ["OPEN", "HIGH", "LOW", "VWAP"]},
            "rolling": {},
        }
    )
    candidate_fields, candidate_names = factor_config(families)
    names = [*alpha_names, *candidate_names]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Alpha158 and candidate feature names overlap: {duplicates}")
    return [*alpha_fields, *candidate_fields], names


def build_alpha158_factor_handler(
    *,
    instruments: str = "csi500",
    start_time: str | None = None,
    end_time: str | None = None,
    fit_start_time: str | None = None,
    fit_end_time: str | None = None,
    label: tuple[list[str], list[str]] | None = None,
    families: Iterable[str] | None = None,
    **handler_kwargs: Any,
):
    """Build an Alpha158-compatible handler with candidate columns appended.

    This is the safe runner integration point: pass the same arguments currently
    sent to ``Alpha158`` and optionally select complete factor families.  The
    official Alpha158 processors and infer/learn behavior remain unchanged.
    """

    from qlib.contrib.data.handler import Alpha158

    fields, names = combined_alpha158_feature_config(families)

    class _Alpha158WithFrozenCandidates(Alpha158):
        def get_feature_config(self):
            return fields, names

    kwargs: dict[str, Any] = {
        "instruments": instruments,
        "start_time": start_time,
        "end_time": end_time,
        "fit_start_time": fit_start_time,
        "fit_end_time": fit_end_time,
        **handler_kwargs,
    }
    if label is not None:
        kwargs["label"] = label
    return _Alpha158WithFrozenCandidates(**kwargs)

"""Experiment results schema — DRAFT, needs TCA hardening.

This is the initial sketch. It works but violates TCA principles:

- int for prompt_id and sample_id — should be newtypes (semantic wrappers)
- str for raw_value and field_name — should be newtypes or enums
- dict returns on cached_property indexes — should be RootModel collections
  with O(1) lookup as a method or property, not raw dicts
- dict[str, object] for raw_json — should be a proper model or at minimum
  a newtype

The structure is right. The types need to be completed.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from functools import cached_property

from pydantic import BaseModel


class ModelId(StrEnum):
    CLAUDE_SONNET_4_6 = "claude_sonnet_4_6"
    CHATGPT_5_2 = "chatgpt_5_2"
    GRANITE_19B = "granite_19b"


class VariantId(StrEnum):
    BASELINE = "baseline"
    VACUOUS = "vacuous"
    MISLEADING = "misleading"


class EnumResult(BaseModel, frozen=True):
    raw_value: str          # should be a newtype
    ordinal: int            # should be a newtype (EnumOrdinal)
    field_name: str         # should be a newtype or enum (FieldPosition)


class NormalizedResponse(BaseModel, frozen=True):
    enum_fields: tuple[EnumResult, ...]
    confidence: float
    interventions: tuple[str, ...]   # str items should be a newtype
    reasoning: str                   # should be a newtype
    raw_json: dict[str, object]      # should be a model or newtype


class ExperimentRun(BaseModel, frozen=True):
    model: ModelId
    variant: VariantId
    prompt_id: int              # should be PromptId newtype
    sample_id: int              # should be SampleId newtype
    response: NormalizedResponse
    latency_ms: int             # should be a newtype (LatencyMs)
    timestamp: datetime
    token_count: int            # should be a newtype (TokenCount)


class ExperimentResults(BaseModel, frozen=True):
    runs: tuple[ExperimentRun, ...]

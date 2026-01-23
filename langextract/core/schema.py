# Copyright 2025 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Core schema abstractions for LangExtract."""
from __future__ import annotations

import abc
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from langextract.core import data
from langextract.core import format_handler as fh
from langextract.core import types

if TYPE_CHECKING:
    from pydantic import BaseModel as PydanticBaseModel

__all__ = [
    "ConstraintType",
    "Constraint",
    "BaseSchema",
    "FormatModeSchema",
]

# Backward compat re-exports
ConstraintType = types.ConstraintType
Constraint = types.Constraint


class BaseSchema(abc.ABC):
    """Abstract base class for structured schema constraints."""

    @classmethod
    @abc.abstractmethod
    def from_examples(
        cls,
        examples_data: Sequence[data.ExampleData],
        attribute_suffix: str = data.ATTRIBUTE_SUFFIX,
    ) -> BaseSchema:
        """Factory method to build a schema instance from example data."""

    @classmethod
    def from_pydantic(
        cls,
        model_class: type["PydanticBaseModel"],
        schema_name: str | None = None,
    ) -> "BaseSchema":
        """Build a schema instance from a Pydantic model.

        Converts a Pydantic BaseModel class into a provider-specific
        schema that constrains the model's output structure.

        Args:
            model_class: A Pydantic BaseModel class defining the
                output structure.
            schema_name: Optional name for the schema. Defaults to
                the model class name if not provided.

        Returns:
            A schema instance configured for the Pydantic model.

        Raises:
            NotImplementedError: If the provider doesn't support
                Pydantic schemas. Subclasses should override this
                method to provide support.

        Example:
            >>> from pydantic import BaseModel
            >>> class Person(BaseModel):
            ...     name: str
            ...     age: int
            >>> schema = SomeProviderSchema.from_pydantic(Person)
        """
        raise NotImplementedError(
            f"{cls.__name__} does not support Pydantic schema generation. "
            "Use from_examples() with ExampleData objects instead, or use a "
            "provider that supports Pydantic schemas (e.g., OpenAI, Gemini)."
        )

    @abc.abstractmethod
    def to_provider_config(self) -> dict[str, Any]:
        """Convert schema to provider-specific configuration.

        Returns:
          Dictionary of provider kwargs (e.g., response_schema for Gemini).
          Should be a pure data mapping with no side effects.
        """

    @property
    @abc.abstractmethod
    def requires_raw_output(self) -> bool:
        """Whether this schema outputs raw JSON/YAML without fence markers.

        When True, the provider emits syntactically valid JSON directly.
        When False, the provider needs fence markers for structure.
        """

    def validate_format(self, format_handler: fh.FormatHandler) -> None:
        """Validate format compatibility and warn about issues.

        Override in subclasses to check format settings.
        Default implementation does nothing (no validation needed).

        Args:
          format_handler: The format configuration to validate.
        """

    def sync_with_provider_kwargs(self, kwargs: dict[str, Any]) -> None:
        """Hook to update schema state based on provider kwargs.

        This allows schemas to adjust their behavior based on caller overrides.
        For example, FormatModeSchema uses this to sync its format when the
        caller overrides it, ensuring requires_raw_output stays accurate.

        Default implementation does nothing. Override if your schema needs to
        respond to provider kwargs.

        Args:
          kwargs: The effective provider kwargs after merging.
        """


class FormatModeSchema(BaseSchema):
    """Generic schema for providers that support format modes (JSON/YAML).

    This schema doesn't enforce structure, only output format. Useful for
    providers that can guarantee syntactically valid JSON or YAML but don't
    support field-level constraints.
    """

    def __init__(self, format_type: types.FormatType = types.FormatType.JSON):
        """Initialize with a format type."""
        self.format_type = format_type
        # Keep _format for backward compatibility with tests
        is_json = format_type == types.FormatType.JSON
        self._format = "json" if is_json else "yaml"

    @classmethod
    def from_examples(
        cls,
        examples_data: Sequence[data.ExampleData],
        attribute_suffix: str = data.ATTRIBUTE_SUFFIX,
    ) -> FormatModeSchema:
        """Factory method to build a schema instance from example data."""
        # Default to JSON format
        return cls(format_type=types.FormatType.JSON)

    def to_provider_config(self) -> dict[str, Any]:
        """Convert schema to provider-specific configuration."""
        return {"format": self._format}

    @property
    def requires_raw_output(self) -> bool:
        """JSON format outputs raw JSON without fences, YAML does not."""
        return self._format == "json"

    def sync_with_provider_kwargs(self, kwargs: dict[str, Any]) -> None:
        """Sync format type with provider kwargs."""
        if "format_type" in kwargs:
            self.format_type = kwargs["format_type"]
            self._format = (
                "json" if self.format_type == types.FormatType.JSON else "yaml"
            )
        if "format" in kwargs:
            self._format = kwargs["format"]
            self.format_type = (
                types.FormatType.JSON
                if self._format == "json"
                else types.FormatType.YAML
            )
            )

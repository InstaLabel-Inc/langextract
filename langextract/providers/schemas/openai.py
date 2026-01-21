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

"""OpenAI provider schema implementation for Structured Outputs."""
# pylint: disable=duplicate-code

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
from typing import Any
import warnings

from langextract.core import data
from langextract.core import format_handler as fh
from langextract.core import schema


@dataclasses.dataclass
class OpenAISchema(schema.BaseSchema):
    """Schema implementation for OpenAI Structured Outputs.

    Converts ExampleData objects into a JSON Schema definition that OpenAI
    can interpret via the 'response_format' parameter with type 'json_schema'.

    OpenAI Structured Outputs has specific requirements:
    - All properties must be listed in 'required' array
    - 'additionalProperties' must be False at every object level
    - Optional fields use type union with null: ["object", "null"]
    - Schema must be wrapped in json_schema object with name and strict: true
    """

    _schema_dict: dict[str, Any]
    _schema_name: str = "extraction_response"

    @property
    def schema_dict(self) -> dict[str, Any]:
        """Returns the schema dictionary."""
        return self._schema_dict

    @schema_dict.setter
    def schema_dict(self, schema_dict: dict[str, Any]) -> None:
        """Sets the schema dictionary."""
        self._schema_dict = schema_dict

    @property
    def schema_name(self) -> str:
        """Returns the schema name used in the API call."""
        return self._schema_name

    def to_provider_config(self) -> dict[str, Any]:
        """Convert schema to OpenAI-specific configuration.

        Returns:
          Dictionary with response_format for OpenAI API containing:
          - type: "json_schema"
          - json_schema: object with name, strict, and schema
        """
        return {
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": self._schema_name,
                    "strict": True,
                    "schema": self._schema_dict,
                },
            },
        }

    @property
    def requires_raw_output(self) -> bool:
        """OpenAI Structured Outputs returns raw JSON without fence markers."""
        return True

    def validate_format(self, format_handler: fh.FormatHandler) -> None:
        """Validate OpenAI's format requirements.

        OpenAI Structured Outputs:
        - Outputs raw JSON directly (no fence markers needed)
        - Expects wrapper with EXTRACTIONS_KEY (built into schema)
        """
        # Check for fence usage with raw JSON output
        if format_handler.use_fences:
            warnings.warn(
                "OpenAI Structured Outputs returns native JSON via response_format. "
                "Using fence_output=True may cause parsing issues. "
                "Set fence_output=False.",
                UserWarning,
                stacklevel=3,
            )

        # Verify wrapper is enabled with correct key
        if (
            not format_handler.use_wrapper
            or format_handler.wrapper_key != data.EXTRACTIONS_KEY
        ):
            warnings.warn(
                "OpenAI's response_format schema expects "
                f"wrapper_key='{data.EXTRACTIONS_KEY}'. Current settings: "
                f"use_wrapper={format_handler.use_wrapper}, "
                f"wrapper_key='{format_handler.wrapper_key}'",
                UserWarning,
                stacklevel=3,
            )

    @classmethod
    def from_examples(
        cls,
        examples_data: Sequence[data.ExampleData],
        attribute_suffix: str = data.ATTRIBUTE_SUFFIX,
    ) -> OpenAISchema:
        """Creates an OpenAISchema from example extractions.

        Builds a JSON Schema with a top-level "extractions" array. Each element
        in that array is an object containing the extraction class name and an
        accompanying "<class>_attributes" object for its attributes.

        OpenAI Structured Outputs requires:
        - All properties must be in the 'required' array
        - 'additionalProperties': false at every object level
        - Optional/nullable fields use type union: ["object", "null"]

        Args:
          examples_data: A sequence of ExampleData objects containing extraction
            classes and attributes.
          attribute_suffix: String appended to each class name to form the
            attributes field name (defaults to "_attributes").

        Returns:
          An OpenAISchema with internal dictionary representing the JSON Schema.
        """
        # Track attribute types for each category
        extraction_categories: dict[str, dict[str, set[type]]] = {}
        for example in examples_data:
            for extraction in example.extractions:
                category = extraction.extraction_class
                if category not in extraction_categories:
                    extraction_categories[category] = {}

                if extraction.attributes:
                    for attr_name, attr_value in extraction.attributes.items():
                        if attr_name not in extraction_categories[category]:
                            extraction_categories[category][attr_name] = set()
                        extraction_categories[category][attr_name].add(type(attr_value))

        extraction_properties: dict[str, dict[str, Any]] = {}
        extraction_required: list[str] = []

        for category, attrs in extraction_categories.items():
            # Add the extraction class field
            extraction_properties[category] = {"type": "string"}
            extraction_required.append(category)

            # Build attributes field
            attributes_field = f"{category}{attribute_suffix}"
            attr_properties: dict[str, Any] = {}
            attr_required: list[str] = []

            # Default property for categories without attributes
            if not attrs:
                attr_properties["_unused"] = {"type": "string"}
                attr_required.append("_unused")
            else:
                for attr_name, attr_types in attrs.items():
                    # List attributes become arrays
                    if list in attr_types:
                        attr_properties[attr_name] = {
                            "type": "array",
                            "items": {"type": "string"},
                        }
                    else:
                        attr_properties[attr_name] = {"type": "string"}
                    attr_required.append(attr_name)

            # OpenAI requires nullable fields to use type union syntax
            # and all nested objects must have additionalProperties: false
            extraction_properties[attributes_field] = {
                "type": ["object", "null"],  # OpenAI nullable syntax
                "properties": attr_properties,
                "required": attr_required,
                "additionalProperties": False,
            }
            extraction_required.append(attributes_field)

        # Build the extraction item schema
        extraction_schema = {
            "type": "object",
            "properties": extraction_properties,
            "required": extraction_required,
            "additionalProperties": False,
        }

        # Build the root schema
        schema_dict = {
            "type": "object",
            "properties": {
                data.EXTRACTIONS_KEY: {
                    "type": "array",
                    "items": extraction_schema,
                }
            },
            "required": [data.EXTRACTIONS_KEY],
            "additionalProperties": False,
        }

        return cls(_schema_dict=schema_dict)

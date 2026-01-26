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

"""Gemini provider schema implementation."""
# pylint: disable=duplicate-code

from __future__ import annotations

import copy
import dataclasses
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from langextract.core import data, schema
from langextract.core import format_handler as fh

if TYPE_CHECKING:
    from pydantic import BaseModel as PydanticBaseModel


@dataclasses.dataclass
class GeminiSchema(schema.BaseSchema):
    """Schema implementation for Gemini structured output.

    Converts ExampleData objects or Pydantic models into an OpenAPI/JSON-schema
    definition that Gemini can interpret via 'response_schema'.
    """

    _schema_dict: dict[str, Any]
    _schema_name: str = "extraction_response"
    _model_class: type["PydanticBaseModel"] | None = dataclasses.field(
        default=None, repr=False
    )

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
        """Returns the schema name."""
        return self._schema_name

    def to_provider_config(self) -> dict[str, Any]:
        """Convert schema to Gemini-specific configuration.

        Returns:
          Dictionary with response_schema and response_mime_type for Gemini API.
        """
        return {
            "response_schema": self._schema_dict,
            "response_mime_type": "application/json",
        }

    @property
    def requires_raw_output(self) -> bool:
        """Gemini outputs raw JSON via response_mime_type."""
        return True

    def validate_format(self, format_handler: fh.FormatHandler) -> None:
        """Validate Gemini's format requirements.

        Gemini requires:
        - No fence markers (outputs raw JSON via response_mime_type)
        - Wrapper with EXTRACTIONS_KEY (built into response_schema)
        """
        # Check for fence usage with raw JSON output
        if format_handler.use_fences:
            warnings.warn(
                "Gemini outputs native JSON via"
                " response_mime_type='application/json'. Using fence_output=True may"
                " cause parsing issues. Set fence_output=False.",
                UserWarning,
                stacklevel=3,
            )

        # Verify wrapper is enabled with correct key
        if (
            not format_handler.use_wrapper
            or format_handler.wrapper_key != data.EXTRACTIONS_KEY
        ):
            warnings.warn(
                "Gemini's response_schema expects"
                f" wrapper_key='{data.EXTRACTIONS_KEY}'. Current settings:"
                f" use_wrapper={format_handler.use_wrapper},"
                f" wrapper_key='{format_handler.wrapper_key}'",
                UserWarning,
                stacklevel=3,
            )

    @classmethod
    def from_examples(
        cls,
        examples_data: Sequence[data.ExampleData],
        attribute_suffix: str = data.ATTRIBUTE_SUFFIX,
    ) -> GeminiSchema:
        """Creates a GeminiSchema from example extractions.

        Builds a JSON-based schema with a top-level "extractions" array. Each
        element in that array is an object containing the extraction class name
        and an accompanying "<class>_attributes" object for its attributes.

        Args:
          examples_data: A sequence of ExampleData objects containing extraction
            classes and attributes.
          attribute_suffix: String appended to each class name to form the
            attributes field name (defaults to "_attributes").

        Returns:
          A GeminiSchema with internal dictionary represents the JSON constraint.
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

        for category, attrs in extraction_categories.items():
            extraction_properties[category] = {"type": "string"}

            attributes_field = f"{category}{attribute_suffix}"
            attr_properties = {}

            # Default property for categories without attributes
            if not attrs:
                attr_properties["_unused"] = {"type": "string"}
            else:
                for attr_name, attr_types in attrs.items():
                    # List attributes become arrays
                    if list in attr_types:
                        attr_properties[attr_name] = {
                            "type": "array",
                            "items": {"type": "string"},  # type: ignore[dict-item]
                        }
                    else:
                        attr_properties[attr_name] = {"type": "string"}

            extraction_properties[attributes_field] = {
                "type": "object",
                "properties": attr_properties,
                "nullable": True,
            }

        extraction_schema = {
            "type": "object",
            "properties": extraction_properties,
        }

        schema_dict = {
            "type": "object",
            "properties": {
                data.EXTRACTIONS_KEY: {"type": "array", "items": extraction_schema}
            },
            "required": [data.EXTRACTIONS_KEY],
        }

        return cls(_schema_dict=schema_dict)

    @classmethod
    def from_pydantic(
        cls,
        model_class: type["PydanticBaseModel"],
        schema_name: str | None = None,
    ) -> "GeminiSchema":
        """Creates a GeminiSchema from a Pydantic model.

        Gemini accepts JSON schema via response_schema. This method converts
        Pydantic's model_json_schema() output to a Gemini-compatible format:
        - Wraps the model in an "extractions" array for pipeline compatibility
        - Resolves $defs references inline
        - Converts anyOf (Optional) to nullable: true syntax
        - Removes unsupported keys (title, default, examples)

        Args:
            model_class: A Pydantic BaseModel class.
            schema_name: Optional name for the schema (unused for Gemini).

        Returns:
            A GeminiSchema configured for the Pydantic model.
        """
        del schema_name  # Unused for Gemini
        raw_schema = model_class.model_json_schema()
        defs = raw_schema.pop("$defs", {})
        processed_item = cls._process_node_for_gemini(raw_schema, defs)

        # Wrap in extractions array for pipeline compatibility
        schema_dict = {
            "type": "object",
            "properties": {
                data.EXTRACTIONS_KEY: {
                    "type": "array",
                    "items": processed_item,
                }
            },
            "required": [data.EXTRACTIONS_KEY],
        }

        return cls(
            _schema_dict=schema_dict,
            _model_class=model_class,
        )

    @classmethod
    def _process_node_for_gemini(
        cls,
        node: dict[str, Any],
        defs: dict[str, Any],
    ) -> dict[str, Any]:
        """Recursively process a schema node for Gemini compatibility.

        Args:
            node: A JSON schema node to process.
            defs: The $defs dictionary for resolving references.

        Returns:
            A processed schema node compatible with Gemini.
        """
        if not isinstance(node, dict):
            return node

        result = copy.copy(node)

        # Handle $ref - resolve reference
        if "$ref" in result:
            ref_path = result["$ref"]
            if ref_path.startswith("#/$defs/"):
                def_name = ref_path.split("/")[-1]
                if def_name in defs:
                    resolved = copy.deepcopy(defs[def_name])
                    return cls._process_node_for_gemini(resolved, defs)
            return result

        # Handle anyOf (Pydantic uses this for Optional types)
        if "anyOf" in result:
            result = cls._handle_any_of_for_gemini(result, defs)

        # Handle allOf by merging
        if "allOf" in result:
            result = cls._handle_all_of_for_gemini(result, defs)

        # Handle object type
        if result.get("type") == "object" and "properties" in result:
            props = result.get("properties", {})
            result["properties"] = {
                k: cls._process_node_for_gemini(v, defs) for k, v in props.items()
            }

        # Handle array type
        if result.get("type") == "array" and "items" in result:
            result["items"] = cls._process_node_for_gemini(result["items"], defs)

        # Remove unsupported keys
        result.pop("title", None)
        result.pop("default", None)
        result.pop("examples", None)
        result.pop("$defs", None)

        return result

    @classmethod
    def _handle_any_of_for_gemini(
        cls,
        node: dict[str, Any],
        defs: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle anyOf schemas for Gemini (converts to nullable syntax).

        Args:
            node: A schema node containing anyOf.
            defs: The $defs dictionary for resolving references.

        Returns:
            Processed schema with Gemini-compatible nullable syntax.
        """
        any_of = node["anyOf"]
        non_null = [x for x in any_of if x.get("type") != "null"]
        has_null = any(x.get("type") == "null" for x in any_of)

        if has_null and len(non_null) == 1:
            # Convert to Gemini nullable syntax
            inner = cls._process_node_for_gemini(non_null[0], defs)
            inner["nullable"] = True
            return inner
        elif has_null and len(non_null) > 1:
            # Multiple types + null - keep anyOf but process each
            processed = [cls._process_node_for_gemini(opt, defs) for opt in non_null]
            processed.append({"type": "null"})
            result = dict(node)
            result["anyOf"] = processed
            return result
        else:
            # No null - just process each option
            result = dict(node)
            result["anyOf"] = [
                cls._process_node_for_gemini(opt, defs) for opt in any_of
            ]
            return result

    @classmethod
    def _handle_all_of_for_gemini(
        cls,
        node: dict[str, Any],
        defs: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle allOf schemas by merging them.

        Args:
            node: A schema node containing allOf.
            defs: The $defs dictionary for resolving references.

        Returns:
            Merged schema.
        """
        all_of = node["allOf"]
        merged: dict[str, Any] = {}

        for sub_schema in all_of:
            processed = cls._process_node_for_gemini(sub_schema, defs)
            if "properties" in processed:
                if "properties" not in merged:
                    merged["properties"] = {}
                merged["properties"].update(processed["properties"])
            if "required" in processed:
                if "required" not in merged:
                    merged["required"] = []
                merged["required"].extend(processed["required"])
            if "type" in processed:
                merged["type"] = processed["type"]

        for key, value in node.items():
            if key != "allOf" and key not in merged:
                merged[key] = value

        return cls._process_node_for_gemini(merged, defs)
        return cls._process_node_for_gemini(merged, defs)
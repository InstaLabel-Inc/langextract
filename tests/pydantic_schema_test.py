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

"""Tests for Pydantic schema support."""

from typing import Optional

from absl.testing import absltest
from pydantic import BaseModel, Field

from langextract.core import schema
from langextract.providers.schemas import gemini as gemini_schema
from langextract.providers.schemas import openai as openai_schema


# Test Pydantic models
class SimplePerson(BaseModel):
    """Simple model with required fields only."""

    name: str
    age: int


class PersonWithOptional(BaseModel):
    """Model with optional fields."""

    name: str
    age: Optional[int] = None
    email: Optional[str] = None


class Address(BaseModel):
    """Nested model for address."""

    street: str
    city: str


class PersonWithAddress(BaseModel):
    """Model with nested object."""

    name: str
    address: Address


class PersonWithTags(BaseModel):
    """Model with list field."""

    name: str
    tags: list[str]


class PersonWithDescription(BaseModel):
    """Model with field descriptions."""

    name: str = Field(description="The person's full name")
    age: int = Field(description="Age in years")


class BaseSchemaFromPydanticTest(absltest.TestCase):
    """Tests for BaseSchema.from_pydantic() default behavior."""

    def test_format_mode_schema_raises_not_implemented(self):
        """Test that FormatModeSchema.from_pydantic() raises NotImplementedError."""
        with self.assertRaises(NotImplementedError) as cm:
            schema.FormatModeSchema.from_pydantic(SimplePerson)

        self.assertIn(
            "does not support Pydantic schema generation",
            str(cm.exception),
            msg="Error message should mention Pydantic not supported",
        )


class OpenAISchemaFromPydanticTest(absltest.TestCase):
    """Tests for OpenAISchema.from_pydantic()."""

    def test_simple_model(self):
        """Test schema generation from simple Pydantic model."""
        result = openai_schema.OpenAISchema.from_pydantic(SimplePerson)

        self.assertIsNotNone(result.schema_dict)
        self.assertEqual(result.schema_name, "SimplePerson")
        self.assertEqual(result._model_class, SimplePerson)

    def test_all_fields_required(self):
        """Test that all properties are in required array (OpenAI requirement)."""
        result = openai_schema.OpenAISchema.from_pydantic(SimplePerson)
        schema_dict = result.schema_dict

        self.assertIn("required", schema_dict)
        self.assertIn("name", schema_dict["required"])
        self.assertIn("age", schema_dict["required"])

    def test_additional_properties_false(self):
        """Test that additionalProperties is false (OpenAI requirement)."""
        result = openai_schema.OpenAISchema.from_pydantic(SimplePerson)
        schema_dict = result.schema_dict

        self.assertFalse(
            schema_dict.get("additionalProperties", True),
            msg="additionalProperties should be False",
        )

    def test_optional_fields_nullable(self):
        """Test that Optional fields use type union with null (OpenAI syntax)."""
        result = openai_schema.OpenAISchema.from_pydantic(PersonWithOptional)
        schema_dict = result.schema_dict

        age_type = schema_dict["properties"]["age"]["type"]
        self.assertIsInstance(
            age_type, list, msg="Optional field type should be a list"
        )
        self.assertIn("null", age_type, msg="Optional field should include null type")

    def test_nested_model(self):
        """Test schema generation with nested models."""
        result = openai_schema.OpenAISchema.from_pydantic(PersonWithAddress)
        schema_dict = result.schema_dict

        address_props = schema_dict["properties"]["address"]
        self.assertIn("properties", address_props)
        self.assertFalse(
            address_props.get("additionalProperties", True),
            msg="Nested object should have additionalProperties: false",
        )
        self.assertIn(
            "required",
            address_props,
            msg="Nested object should have required array",
        )

    def test_list_fields(self):
        """Test schema generation with list fields."""
        result = openai_schema.OpenAISchema.from_pydantic(PersonWithTags)
        schema_dict = result.schema_dict

        tags_schema = schema_dict["properties"]["tags"]
        self.assertEqual(tags_schema["type"], "array")
        self.assertIn("items", tags_schema)

    def test_to_provider_config(self):
        """Test that to_provider_config returns correct OpenAI format."""
        result = openai_schema.OpenAISchema.from_pydantic(SimplePerson)
        config = result.to_provider_config()

        self.assertIn("response_format", config)
        self.assertEqual(config["response_format"]["type"], "json_schema")
        self.assertTrue(config["response_format"]["json_schema"]["strict"])
        self.assertIn("schema", config["response_format"]["json_schema"])
        self.assertEqual(
            config["response_format"]["json_schema"]["name"], "SimplePerson"
        )

    def test_custom_schema_name(self):
        """Test custom schema name."""
        result = openai_schema.OpenAISchema.from_pydantic(
            SimplePerson, schema_name="custom_name"
        )

        self.assertEqual(result.schema_name, "custom_name")
        config = result.to_provider_config()
        self.assertEqual(
            config["response_format"]["json_schema"]["name"], "custom_name"
        )

    def test_requires_raw_output(self):
        """Test that schema requires raw output (no fence markers)."""
        result = openai_schema.OpenAISchema.from_pydantic(SimplePerson)
        self.assertTrue(result.requires_raw_output)

    def test_removes_title_and_default(self):
        """Test that unsupported keys like title and default are removed."""
        result = openai_schema.OpenAISchema.from_pydantic(PersonWithOptional)
        schema_dict = result.schema_dict

        # Pydantic adds title by default, should be removed
        self.assertNotIn("title", schema_dict)

        # Check properties don't have title either
        for prop in schema_dict.get("properties", {}).values():
            if isinstance(prop, dict):
                self.assertNotIn("title", prop)


class GeminiSchemaFromPydanticTest(absltest.TestCase):
    """Tests for GeminiSchema.from_pydantic()."""

    def test_simple_model(self):
        """Test schema generation from simple Pydantic model."""
        result = gemini_schema.GeminiSchema.from_pydantic(SimplePerson)

        self.assertIsNotNone(result.schema_dict)
        self.assertEqual(result._model_class, SimplePerson)

    def test_to_provider_config(self):
        """Test that to_provider_config returns correct Gemini format."""
        result = gemini_schema.GeminiSchema.from_pydantic(SimplePerson)
        config = result.to_provider_config()

        self.assertIn("response_schema", config)
        self.assertEqual(config["response_mime_type"], "application/json")

    def test_optional_fields_use_nullable(self):
        """Test that Optional fields use nullable: true syntax for Gemini."""
        result = gemini_schema.GeminiSchema.from_pydantic(PersonWithOptional)
        schema_dict = result.schema_dict

        age_prop = schema_dict["properties"]["age"]
        self.assertTrue(
            age_prop.get("nullable", False),
            msg="Optional field should have nullable: true",
        )

    def test_nested_model(self):
        """Test schema generation with nested models."""
        result = gemini_schema.GeminiSchema.from_pydantic(PersonWithAddress)
        schema_dict = result.schema_dict

        address_props = schema_dict["properties"]["address"]
        self.assertIn("properties", address_props)
        self.assertIn("street", address_props["properties"])
        self.assertIn("city", address_props["properties"])

    def test_list_fields(self):
        """Test schema generation with list fields."""
        result = gemini_schema.GeminiSchema.from_pydantic(PersonWithTags)
        schema_dict = result.schema_dict

        tags_schema = schema_dict["properties"]["tags"]
        self.assertEqual(tags_schema["type"], "array")
        self.assertIn("items", tags_schema)

    def test_preserves_descriptions(self):
        """Test that field descriptions are preserved."""
        result = gemini_schema.GeminiSchema.from_pydantic(PersonWithDescription)
        schema_dict = result.schema_dict

        name_prop = schema_dict["properties"]["name"]
        self.assertEqual(
            name_prop.get("description"),
            "The person's full name",
            msg="Field descriptions should be preserved",
        )

    def test_requires_raw_output(self):
        """Test that schema requires raw output (no fence markers)."""
        result = gemini_schema.GeminiSchema.from_pydantic(SimplePerson)
        self.assertTrue(result.requires_raw_output)

    def test_removes_defs(self):
        """Test that $defs are resolved and removed."""
        result = gemini_schema.GeminiSchema.from_pydantic(PersonWithAddress)
        schema_dict = result.schema_dict

        self.assertNotIn("$defs", schema_dict)
        # Address should be inlined, not a reference
        address_props = schema_dict["properties"]["address"]
        self.assertNotIn("$ref", address_props)


if __name__ == "__main__":
    absltest.main()

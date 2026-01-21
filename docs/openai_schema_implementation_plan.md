# OpenAI Schema Constraints Implementation Plan

> **Goal**: Extend `use_schema_constraints` support to OpenAI models in LangExtract

## Table of Contents

1. [Overview](#overview)
2. [Current State Analysis](#current-state-analysis)
3. [OpenAI API Requirements](#openai-api-requirements)
4. [Implementation Plan](#implementation-plan)
5. [File Changes Summary](#file-changes-summary)
6. [Testing Strategy](#testing-strategy)
7. [Work Log](#work-log)

---

## Overview

### What We're Building

Enable LangExtract users to use `use_schema_constraints=True` with OpenAI models, providing the same structured output guarantees currently available for Gemini models.

### User Experience After Implementation

```python
import langextract as lx

# This will work with schema constraints (currently requires use_schema_constraints=False)
result = lx.extract(
    text_or_documents="John has diabetes and takes metformin",
    prompt_description="Extract medical conditions and medications",
    examples=[...],
    model_id="gpt-4o",
    use_schema_constraints=True,  # ✅ Will now work!
)
```

### Benefits

- **Reliable outputs**: OpenAI's Structured Outputs guarantee JSON schema adherence
- **No validation needed**: Model will always return valid schema-conforming JSON
- **Consistent API**: Same `use_schema_constraints=True` works for both Gemini and OpenAI

---

## Current State Analysis

### How Gemini Schema Support Works (Reference)

```
User calls lx.extract(use_schema_constraints=True)
    │
    ▼
extraction.py: extract()
    │ Passes examples and use_schema_constraints to factory
    ▼
factory.py: _create_model_with_schema()
    │
    ├─► router.resolve("gemini-2.5-flash") → GeminiLanguageModel
    │
    ├─► GeminiLanguageModel.get_schema_class() → GeminiSchema
    │
    ├─► GeminiSchema.from_examples(examples) → schema_instance
    │
    ├─► schema_instance.to_provider_config() → {"response_schema": ..., "response_mime_type": ...}
    │
    ├─► GeminiLanguageModel(**kwargs) → model instance
    │
    └─► model.apply_schema(schema_instance)
            │
            ▼
        During inference: _process_single_prompt()
            │
            └─► Gemini API receives response_schema in config
```

### Current OpenAI Provider State

| Aspect | Current State |
|--------|---------------|
| `get_schema_class()` | Returns `None` (inherits from base) |
| `apply_schema()` | Uses base implementation (stores but doesn't use) |
| `_process_single_prompt()` | Only uses `{"type": "json_object"}` (JSON mode, not Structured Outputs) |
| Schema file | Does not exist |

### Key Files to Modify/Create

| File | Action |
|------|--------|
| `langextract/providers/schemas/openai.py` | **CREATE** |
| `langextract/providers/schemas/__init__.py` | MODIFY |
| `langextract/providers/openai.py` | MODIFY |
| `langextract/_compat/schema.py` | MODIFY (optional) |
| `langextract/schema.py` | MODIFY (optional) |
| `tests/provider_schema_test.py` | MODIFY |
| `README.md` | MODIFY |

---

## OpenAI API Requirements

### Structured Outputs API Format

OpenAI's Structured Outputs uses `response_format` with `type: "json_schema"`:

```python
response = client.chat.completions.create(
    model="gpt-4o-2024-08-06",
    messages=[...],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "extraction_response",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "extractions": {
                        "type": "array",
                        "items": {...}
                    }
                },
                "required": ["extractions"],
                "additionalProperties": False
            }
        }
    }
)
```

### Critical OpenAI Schema Constraints

| Requirement | Description |
|-------------|-------------|
| **All fields required** | Every property must be in `required` array |
| **additionalProperties: false** | Must be set at EVERY object level |
| **Nullable fields** | Use `"type": ["string", "null"]` instead of omitting from required |
| **Root must be object** | Cannot use `anyOf` at root level |
| **Nesting limit** | Max 10 levels of nesting |
| **Property limit** | Max 5000 total object properties |
| **Supported types** | String, Number, Boolean, Integer, Object, Array, Enum, anyOf |

### Supported Models

- `gpt-4o-mini` and later
- `gpt-4o-2024-08-06` and later
- Older models (gpt-4-turbo, gpt-3.5-turbo) only support JSON mode (not strict schema)

### Comparison: Gemini vs OpenAI Schema Format

| Aspect | Gemini | OpenAI |
|--------|--------|--------|
| **API Parameter** | `response_schema` + `response_mime_type` | `response_format` with `json_schema` |
| **Strict mode** | Implicit | Explicit `"strict": true` |
| **Required fields** | Optional (uses `nullable`) | **All must be required** |
| **Additional props** | Optional | **Must be `false` everywhere** |
| **Nullable syntax** | `"nullable": true` | `"type": ["string", "null"]` |
| **Schema wrapper** | Direct schema dict | Wrapped in `json_schema` object with `name` |

---

## Implementation Plan

### Phase 1: Create OpenAISchema Class

**File**: `langextract/providers/schemas/openai.py`

#### 1.1 Class Structure

```python
@dataclasses.dataclass
class OpenAISchema(schema.BaseSchema):
    """Schema implementation for OpenAI Structured Outputs."""
    
    _schema_dict: dict[str, Any]
    _schema_name: str = "extraction_response"
    
    @classmethod
    def from_examples(cls, examples_data, attribute_suffix) -> "OpenAISchema":
        """Build schema from examples - OpenAI compliant."""
        ...
    
    def to_provider_config(self) -> dict[str, Any]:
        """Return OpenAI-specific response_format."""
        ...
    
    @property
    def requires_raw_output(self) -> bool:
        """OpenAI Structured Outputs returns raw JSON."""
        return True
```

#### 1.2 Key Implementation Details

**`from_examples()` must ensure:**

1. All properties are in `required` array
2. `additionalProperties: false` at every object level
3. Optional fields use `"type": ["object", "null"]` pattern
4. Schema follows LangExtract's extraction format:
   ```json
   {
     "extractions": [
       {
         "extraction_class": "...",
         "extraction_class_attributes": {...}
       }
     ]
   }
   ```

**`to_provider_config()` returns:**

```python
{
    "response_format": {
        "type": "json_schema",
        "json_schema": {
            "name": self._schema_name,
            "strict": True,
            "schema": self._schema_dict
        }
    }
}
```

#### 1.3 Schema Generation Logic

```python
# For each extraction class found in examples:
extraction_properties = {}
extraction_required = []

for category in extraction_categories:
    # Add the class name field
    extraction_properties[category] = {"type": "string"}
    extraction_required.append(category)
    
    # Add attributes field (nullable object)
    attr_field = f"{category}{attribute_suffix}"
    extraction_properties[attr_field] = {
        "type": ["object", "null"],  # OpenAI nullable syntax
        "properties": {...},
        "required": [...],  # ALL attribute fields
        "additionalProperties": False  # Required by OpenAI
    }
    extraction_required.append(attr_field)

# Final schema
schema_dict = {
    "type": "object",
    "properties": {
        "extractions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": extraction_properties,
                "required": extraction_required,
                "additionalProperties": False
            }
        }
    },
    "required": ["extractions"],
    "additionalProperties": False
}
```

---

### Phase 2: Update OpenAI Provider

**File**: `langextract/providers/openai.py`

#### 2.1 Add Schema Class Method

```python
from langextract.providers.schemas import openai as openai_schema

class OpenAILanguageModel(base_model.BaseLanguageModel):
    
    @classmethod
    def get_schema_class(cls) -> type[schema.BaseSchema] | None:
        """Return OpenAISchema for structured output support."""
        return openai_schema.OpenAISchema
```

#### 2.2 Override apply_schema()

```python
def apply_schema(self, schema_instance: schema.BaseSchema | None) -> None:
    """Apply schema and store response_format for API calls."""
    super().apply_schema(schema_instance)
    if isinstance(schema_instance, openai_schema.OpenAISchema):
        config = schema_instance.to_provider_config()
        self._response_format = config.get("response_format")
    else:
        self._response_format = None
```

#### 2.3 Add Instance Variable

```python
@dataclasses.dataclass(init=False)
class OpenAILanguageModel(base_model.BaseLanguageModel):
    # ... existing fields ...
    _response_format: dict[str, Any] | None = dataclasses.field(
        default=None, repr=False, compare=False
    )
```

#### 2.4 Update _process_single_prompt()

```python
def _process_single_prompt(self, prompt: str, config: dict) -> core_types.ScoredOutput:
    # ... existing setup code ...
    
    api_params = {
        'model': self.model_id,
        'messages': messages,
        'n': 1,
    }
    
    # Priority: Use schema-based response_format if available
    if self._response_format is not None:
        api_params['response_format'] = self._response_format
    elif self.format_type == data.FormatType.JSON:
        # Fallback to basic JSON mode
        api_params.setdefault('response_format', {'type': 'json_object'})
    
    # ... rest of existing code ...
```

#### 2.5 Update requires_fence_output Property

```python
@property
def requires_fence_output(self) -> bool:
    """OpenAI returns raw JSON when using structured outputs or JSON mode."""
    # Check for explicit override first
    if hasattr(self, '_fence_output_override') and self._fence_output_override is not None:
        return self._fence_output_override
    
    # Schema-based: raw JSON output
    if self._response_format is not None:
        return False
    
    # JSON mode: raw JSON output
    if self.format_type == data.FormatType.JSON:
        return False
    
    return super().requires_fence_output
```

---

### Phase 3: Update Schema Package Exports

**File**: `langextract/providers/schemas/__init__.py`

```python
"""Provider-specific schema implementations."""
from __future__ import annotations

from langextract.providers.schemas import gemini
from langextract.providers.schemas import openai  # ADD THIS

GeminiSchema = gemini.GeminiSchema  # Backward compat
OpenAISchema = openai.OpenAISchema  # ADD THIS

__all__ = ["GeminiSchema", "OpenAISchema"]  # UPDATE
```

---

### Phase 4: Update Compatibility Layer (Optional)

**File**: `langextract/_compat/schema.py`

```python
def __getattr__(name: str):
    moved = {
        # ... existing entries ...
        "GeminiSchema": ("langextract.providers.schemas.gemini", "GeminiSchema"),
        "OpenAISchema": ("langextract.providers.schemas.openai", "OpenAISchema"),  # ADD
    }
    # ... rest unchanged ...
```

**File**: `langextract/schema.py`

```python
def __getattr__(name: str):
    # ... existing code ...
    
    elif name in ("GeminiSchema", "OpenAISchema"):  # UPDATE
        return schema.__getattr__(name)
    
    raise AttributeError(...)
```

---

### Phase 5: Model Compatibility Check (Optional Enhancement)

Some OpenAI models don't support Structured Outputs. Consider adding validation:

```python
# In OpenAISchema or OpenAILanguageModel

STRUCTURED_OUTPUT_MODELS = {
    "gpt-4o",
    "gpt-4o-mini", 
    "gpt-4o-2024-08-06",
    "gpt-4o-mini-2024-07-18",
    # Add more as released
}

def _supports_structured_outputs(model_id: str) -> bool:
    """Check if model supports Structured Outputs (not just JSON mode)."""
    # Check exact match or prefix match for dated versions
    for supported in STRUCTURED_OUTPUT_MODELS:
        if model_id == supported or model_id.startswith(supported):
            return True
    return False
```

---

## File Changes Summary

### Files to CREATE

| File | Description |
|------|-------------|
| `langextract/providers/schemas/openai.py` | OpenAISchema class implementation |

### Files to MODIFY

| File | Changes |
|------|---------|
| `langextract/providers/openai.py` | Add `get_schema_class()`, `apply_schema()`, update `_process_single_prompt()` |
| `langextract/providers/schemas/__init__.py` | Export OpenAISchema |
| `langextract/_compat/schema.py` | Add OpenAISchema to moved dict |
| `langextract/schema.py` | Add OpenAISchema handling |
| `tests/provider_schema_test.py` | Add OpenAI schema tests |
| `README.md` | Remove `use_schema_constraints=False` requirement for OpenAI |

---

## Testing Strategy

### Unit Tests

#### Test 1: Schema Class Discovery
```python
def test_openai_returns_openai_schema(self):
    """Test that OpenAILanguageModel returns OpenAISchema."""
    schema_class = openai.OpenAILanguageModel.get_schema_class()
    self.assertEqual(schema_class, schemas.openai.OpenAISchema)
```

#### Test 2: Schema Generation from Examples
```python
def test_openai_schema_from_examples(self):
    """Test that OpenAISchema correctly generates from examples."""
    examples = [
        data.ExampleData(
            text="Patient has diabetes",
            extractions=[
                data.Extraction(
                    extraction_class="condition",
                    extraction_text="diabetes",
                    attributes={"severity": "moderate"},
                )
            ],
        )
    ]
    
    schema = OpenAISchema.from_examples(examples)
    config = schema.to_provider_config()
    
    # Verify structure
    self.assertIn("response_format", config)
    self.assertEqual(config["response_format"]["type"], "json_schema")
    self.assertTrue(config["response_format"]["json_schema"]["strict"])
    
    # Verify additionalProperties: false
    json_schema = config["response_format"]["json_schema"]["schema"]
    self.assertFalse(json_schema.get("additionalProperties", True))
```

#### Test 3: All Fields Required
```python
def test_openai_schema_all_fields_required(self):
    """Test that all properties are in required array."""
    examples = [...]
    schema = OpenAISchema.from_examples(examples)
    json_schema = schema._schema_dict
    
    # Check root level
    self.assertEqual(
        set(json_schema["properties"].keys()),
        set(json_schema["required"])
    )
```

#### Test 4: Provider Config Integration
```python
def test_openai_forwards_schema_to_api(self):
    """Test that schema is passed to OpenAI API."""
    with mock.patch("openai.OpenAI") as mock_client:
        # Setup mock
        mock_instance = mock_client.return_value
        mock_instance.chat.completions.create.return_value = mock_response
        
        # Create model with schema
        model = OpenAILanguageModel(model_id="gpt-4o", api_key="test")
        schema = OpenAISchema.from_examples(examples)
        model.apply_schema(schema)
        
        # Run inference
        list(model.infer(["Test prompt"]))
        
        # Verify API call includes response_format
        call_kwargs = mock_instance.chat.completions.create.call_args[1]
        self.assertIn("response_format", call_kwargs)
        self.assertEqual(call_kwargs["response_format"]["type"], "json_schema")
```

#### Test 5: requires_raw_output
```python
def test_openai_schema_requires_raw_output(self):
    """Test that OpenAISchema requires raw output (no fences)."""
    schema = OpenAISchema.from_examples([])
    self.assertTrue(schema.requires_raw_output)
```

### Integration Tests

#### Test with factory.create_model()
```python
def test_factory_creates_openai_with_schema(self):
    """Test that factory correctly applies schema to OpenAI model."""
    with mock.patch("openai.OpenAI"):
        config = factory.ModelConfig(
            model_id="gpt-4o",
            provider_kwargs={"api_key": "test"}
        )
        
        model = factory.create_model(
            config=config,
            examples=examples,
            use_schema_constraints=True,
        )
        
        self.assertIsNotNone(model._response_format)
        self.assertFalse(model.requires_fence_output)
```

### Live API Test (Optional)
```python
@unittest.skipUnless(os.getenv("OPENAI_API_KEY"), "Requires API key")
def test_live_openai_structured_output(self):
    """Test actual API call with structured output."""
    result = lx.extract(
        text_or_documents="John has diabetes",
        prompt_description="Extract medical conditions",
        examples=[...],
        model_id="gpt-4o",
        use_schema_constraints=True,
    )
    
    self.assertIsNotNone(result)
    self.assertTrue(len(result.extractions) > 0)
```

---

## Work Log

### Status Legend
- ⬜ Not Started
- 🟡 In Progress  
- ✅ Completed
- ❌ Blocked

---

### Phase 1: OpenAISchema Class
| Task | Status | Date | Notes |
|------|--------|------|-------|
| Create `langextract/providers/schemas/openai.py` | ✅ | 2026-01-21 | File created with full implementation |
| Implement `OpenAISchema` dataclass | ✅ | 2026-01-21 | Includes `_schema_dict` and `_schema_name` fields |
| Implement `from_examples()` method | ✅ | 2026-01-21 | Generates OpenAI-compliant schema with all required fields |
| Implement `to_provider_config()` method | ✅ | 2026-01-21 | Returns `response_format` with `json_schema` type |
| Implement `requires_raw_output` property | ✅ | 2026-01-21 | Returns `True` for raw JSON output |
| Implement `validate_format()` method | ✅ | 2026-01-21 | Warns about fence/wrapper issues |
| Add unit tests for schema generation | ⬜ | | Pending - will add in testing phase |

### Phase 2: OpenAI Provider Updates
| Task | Status | Date | Notes |
|------|--------|------|-------|
| Add import for openai schema | ✅ | 2026-01-21 | `from langextract.providers.schemas import openai as openai_schema` |
| Add `_response_format` field | ✅ | 2026-01-21 | Dataclass field + initialized in `__init__` |
| Implement `get_schema_class()` | ✅ | 2026-01-21 | Returns `openai_schema.OpenAISchema` |
| Implement `apply_schema()` | ✅ | 2026-01-21 | Extracts `response_format` from schema config |
| Update `_process_single_prompt()` | ✅ | 2026-01-21 | Uses schema response_format with priority over JSON mode |
| Update `requires_fence_output` | ✅ | 2026-01-21 | Checks `_response_format`, override, and format_type |
| Add unit tests for provider | ✅ | 2026-01-21 | Added in Phase 4 - all tests passing |

### Phase 3: Package Exports
| Task | Status | Date | Notes |
|------|--------|------|-------|
| Update `schemas/__init__.py` | ✅ | 2026-01-21 | Added `OpenAISchema` export |
| Update `_compat/schema.py` | ✅ | 2026-01-21 | Added `OpenAISchema` to moved dict |
| Update `schema.py` | ✅ | 2026-01-21 | Added `OpenAISchema` to provider schema handling |

### Phase 4: Testing
| Task | Status | Date | Notes |
|------|--------|------|-------|
| Schema discovery test | ✅ | 2026-01-21 | `test_openai_returns_openai_schema` - updated existing test |
| Schema generation test | ✅ | 2026-01-21 | `test_from_examples_creates_schema` |
| All fields required test | ✅ | 2026-01-21 | `test_all_fields_required` |
| additionalProperties test | ✅ | 2026-01-21 | `test_additional_properties_false` |
| Nullable fields test | ✅ | 2026-01-21 | `test_nullable_fields_use_type_union` |
| Provider config test | ✅ | 2026-01-21 | `test_to_provider_config_returns_response_format` |
| apply_schema test | ✅ | 2026-01-21 | `test_openai_apply_schema_stores_response_format` |
| requires_fence_output test | ✅ | 2026-01-21 | `test_openai_requires_fence_output_false_with_schema` |
| API forwarding test | ✅ | 2026-01-21 | `test_openai_forwards_schema_to_api` |
| Factory integration test | ✅ | 2026-01-21 | `test_factory_creates_openai_with_schema` |
| Schema shim test | ✅ | 2026-01-21 | `test_provider_schema_imports` - added OpenAISchema check |
| Live API test (optional) | ✅ | 2026-01-21 | All 3 live tests passed with gpt-4o |

### Phase 5: Documentation
| Task | Status | Date | Notes |
|------|--------|------|-------|
| Update README.md | ✅ | 2026-01-21 | Updated OpenAI example to use `use_schema_constraints=True` |
| Update provider README | ✅ | 2026-01-21 | Provider README already has generic schema docs |
| Add docstrings | ✅ | 2026-01-21 | Docstrings already present in implementation |

### Phase 6: Final Review
| Task | Status | Date | Notes |
|------|--------|------|-------|
| Run full test suite | ✅ | 2026-01-21 | 402 passed, 78 warnings in 4.28s |
| Run linter | ✅ | 2026-01-21 | All tests pass after cleanup |
| Code review | ✅ | 2026-01-21 | Implementation reviewed and verified |
| Merge to main | ⬜ | | Ready for merge |

---

## Appendix

### A. Example Generated Schema

For these examples:
```python
examples = [
    ExampleData(
        text="Patient John has diabetes and hypertension",
        extractions=[
            Extraction(
                extraction_class="condition",
                extraction_text="diabetes",
                attributes={"severity": "moderate", "onset": "recent"}
            ),
            Extraction(
                extraction_class="condition", 
                extraction_text="hypertension",
                attributes={"severity": "mild"}
            ),
        ]
    )
]
```

Generated OpenAI-compatible schema:
```json
{
    "type": "object",
    "properties": {
        "extractions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "condition": {
                        "type": "string"
                    },
                    "condition_attributes": {
                        "type": ["object", "null"],
                        "properties": {
                            "severity": {"type": "string"},
                            "onset": {"type": "string"}
                        },
                        "required": ["severity", "onset"],
                        "additionalProperties": false
                    }
                },
                "required": ["condition", "condition_attributes"],
                "additionalProperties": false
            }
        }
    },
    "required": ["extractions"],
    "additionalProperties": false
}
```

### B. API Call Example

Final API call to OpenAI:
```python
client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant..."},
        {"role": "user", "content": "Extract from: Patient John has diabetes..."}
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "extraction_response",
            "strict": True,
            "schema": {
                # ... schema from above ...
            }
        }
    }
)
```

### C. References

- [OpenAI Structured Outputs Documentation](https://platform.openai.com/docs/guides/structured-outputs)
- [JSON Schema Specification](https://json-schema.org/)
- [LangExtract Gemini Schema Implementation](langextract/providers/schemas/gemini.py)

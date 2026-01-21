#!/usr/bin/env python3
"""Live test script for OpenAI schema constraints.

Usage:
    export OPENAI_API_KEY="your-api-key"
    python test_openai_schema_live.py

Or run with the key as an argument:
    python test_openai_schema_live.py --api-key "your-api-key"
"""

import argparse
import json
import os
import sys
import textwrap

import dotenv
dotenv.load_dotenv()

import langextract as lx
from langextract.core import data


def test_basic_extraction_with_schema(api_key: str):
    """Test basic extraction with use_schema_constraints=True."""
    print("\n" + "=" * 60)
    print("TEST 1: Basic Medication Extraction (with schema)")
    print("=" * 60)

    prompt = textwrap.dedent("""\
        Extract medication information including medication name, dosage, 
        route, and frequency from the text.""")

    examples = [
        data.ExampleData(
            text="Patient was given 250 mg IV Cefazolin TID.",
            extractions=[
                data.Extraction(
                    extraction_class="dosage",
                    extraction_text="250 mg",
                ),
                data.Extraction(
                    extraction_class="route",
                    extraction_text="IV",
                ),
                data.Extraction(
                    extraction_class="medication",
                    extraction_text="Cefazolin",
                ),
                data.Extraction(
                    extraction_class="frequency",
                    extraction_text="TID",
                ),
            ],
        )
    ]

    input_text = "Patient took 400 mg PO Ibuprofen every 4 hours for pain."

    print(f"\nInput text: {input_text}")
    print(f"Model: gpt-4o")
    print(f"use_schema_constraints: True")
    print("\nCalling lx.extract()...")

    try:
        result = lx.extract(
            text_or_documents=input_text,
            prompt_description=prompt,
            examples=examples,
            model_id="gpt-4o",
            api_key=api_key,
            use_schema_constraints=True,  # NEW: Testing with schema!
            language_model_params={"temperature": 0.0},
        )

        print("\n✅ SUCCESS!")
        print(f"\nResult type: {type(result).__name__}")
        print(f"Number of extractions: {len(result.extractions)}")

        print("\nExtractions:")
        for ext in result.extractions:
            print(f"  - {ext.extraction_class}: '{ext.extraction_text}'")
            if ext.attributes:
                print(f"    Attributes: {ext.attributes}")

        return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_extraction_with_attributes(api_key: str):
    """Test extraction with attributes using schema constraints."""
    print("\n" + "=" * 60)
    print("TEST 2: Extraction with Attributes (with schema)")
    print("=" * 60)

    prompt = textwrap.dedent("""\
        Extract medical conditions with their severity and type.""")

    examples = [
        data.ExampleData(
            text="Patient has severe diabetes and mild hypertension.",
            extractions=[
                data.Extraction(
                    extraction_class="condition",
                    extraction_text="diabetes",
                    attributes={"severity": "severe", "type": "chronic"},
                ),
                data.Extraction(
                    extraction_class="condition",
                    extraction_text="hypertension",
                    attributes={"severity": "mild", "type": "chronic"},
                ),
            ],
        )
    ]

    input_text = "The patient presents with moderate asthma and acute bronchitis."

    print(f"\nInput text: {input_text}")
    print(f"Model: gpt-4o")
    print(f"use_schema_constraints: True")
    print("\nCalling lx.extract()...")

    try:
        result = lx.extract(
            text_or_documents=input_text,
            prompt_description=prompt,
            examples=examples,
            model_id="gpt-4o",
            api_key=api_key,
            use_schema_constraints=True,
            language_model_params={"temperature": 0.0},
        )

        print("\n✅ SUCCESS!")
        print(f"\nResult type: {type(result).__name__}")
        print(f"Number of extractions: {len(result.extractions)}")

        print("\nExtractions:")
        for ext in result.extractions:
            print(f"  - {ext.extraction_class}: '{ext.extraction_text}'")
            if ext.attributes:
                print(f"    Attributes: {ext.attributes}")

        return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_compare_with_without_schema(api_key: str):
    """Compare results with and without schema constraints."""
    print("\n" + "=" * 60)
    print("TEST 3: Compare With/Without Schema Constraints")
    print("=" * 60)

    prompt = "Extract product names and prices from the text."

    examples = [
        data.ExampleData(
            text="The laptop costs $999 and the mouse is $25.",
            extractions=[
                data.Extraction(
                    extraction_class="product",
                    extraction_text="laptop",
                    attributes={"price": "$999"},
                ),
                data.Extraction(
                    extraction_class="product",
                    extraction_text="mouse",
                    attributes={"price": "$25"},
                ),
            ],
        )
    ]

    input_text = "We bought a keyboard for $75 and a monitor for $350."

    print(f"\nInput text: {input_text}")

    # Test WITHOUT schema constraints
    print("\n--- Without schema constraints ---")
    try:
        result_no_schema = lx.extract(
            text_or_documents=input_text,
            prompt_description=prompt,
            examples=examples,
            model_id="gpt-4o",
            api_key=api_key,
            use_schema_constraints=False,
            language_model_params={"temperature": 0.0},
        )
        print(f"✅ Extractions: {len(result_no_schema.extractions)}")
        for ext in result_no_schema.extractions:
            print(f"   {ext.extraction_class}: '{ext.extraction_text}' {ext.attributes or ''}")
    except Exception as e:
        print(f"❌ Failed: {e}")

    # Test WITH schema constraints
    print("\n--- With schema constraints ---")
    try:
        result_with_schema = lx.extract(
            text_or_documents=input_text,
            prompt_description=prompt,
            examples=examples,
            model_id="gpt-4o",
            api_key=api_key,
            use_schema_constraints=True,
            language_model_params={"temperature": 0.0},
        )
        print(f"✅ Extractions: {len(result_with_schema.extractions)}")
        for ext in result_with_schema.extractions:
            print(f"   {ext.extraction_class}: '{ext.extraction_text}' {ext.attributes or ''}")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Test OpenAI schema constraints")
    parser.add_argument(
        "--api-key",
        help="OpenAI API key (or set OPENAI_API_KEY env var)",
    )
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: Please provide an API key via --api-key or OPENAI_API_KEY env var")
        sys.exit(1)

    print("=" * 60)
    print("OpenAI Schema Constraints Live Test")
    print("=" * 60)
    print(f"Testing use_schema_constraints=True with OpenAI gpt-4o")

    results = []
    results.append(("Basic extraction", test_basic_extraction_with_schema(api_key)))
    results.append(("Attributes extraction", test_extraction_with_attributes(api_key)))
    results.append(("Compare with/without", test_compare_with_without_schema(api_key)))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_passed = True
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n⚠️ Some tests failed")
        sys.exit(1)


if __name__ == "__main__":
    main()

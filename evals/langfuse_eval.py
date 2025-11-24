#!/usr/bin/env python3
"""
Langfuse Evaluation Framework for Multi-Agent System

This script creates datasets and runs evaluations on the multi-agent system.
"""

import os
import sys
import json
from typing import List, Dict
from langfuse import Langfuse

# Initialize Langfuse
langfuse = Langfuse(
    secret_key=os.getenv("LANGFUSE_SECRET_KEY", "sk-lf-secret"),
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY", "pk-lf-public"),
    host=os.getenv("LANGFUSE_HOST", "http://localhost:3000"),
)

# Test datasets
BOM_QUERIES = [
    {"input": "Get BOM for 3.5L V6 Engine", "expected_components": ["Piston Assembly", "Spark Plug", "Fuel Injector"]},
    {"input": "Show parts for 5.3L V8 Engine", "expected_components": []},
    {"input": "Get BOM for CVT Transmission", "expected_components": []},
]

SUPPLIER_QUERIES = [
    {"input": "Analyze risk for Denso in Japan", "expected_score_range": (10, 20)},
    {"input": "Analyze risk for Bosch in Germany", "expected_score_range": (20, 30)},
    {"input": "Analyze risk for BorgWarner in USA", "expected_score_range": (25, 35)},
]

MATERIALS_QUERIES = [
    {"input": "Find details about brake pads", "expected_fields": ["oem_status", "manufacturer", "average_price"]},
    {"input": "Find details about spark plugs", "expected_fields": ["oem_status", "manufacturer"]},
    {"input": "Find details about turbocharger", "expected_fields": ["manufacturer", "average_price"]},
]

ORCHESTRATION_QUERIES = [
    {
        "input": "Who supplies the turbocharger for engines and what are their risks?",
        "expected_agents": ["find-material", "analyze-risk"],
        "description": "Multi-step orchestration"
    },
]


def create_dataset(name: str, items: List[Dict]):
    """Create or update a Langfuse dataset"""
    try:
        dataset = langfuse.create_dataset(name=name)
        print(f"✅ Created dataset: {name}")
    except Exception as e:
        print(f"ℹ️  Dataset '{name}' already exists or error: {e}")
        return
    
    for idx, item in enumerate(items):
        try:
            dataset.create_item(
                input=item.get("input"),
                expected_output=item.get("expected_output", item),
                metadata=item.get("metadata", {})
            )
            print(f"  ✓ Added item {idx + 1}")
        except Exception as e:
            print(f"  ✗ Error adding item {idx + 1}: {e}")


def create_all_datasets():
    """Create all evaluation datasets"""
    print("=" * 60)
    print("Creating Langfuse Evaluation Datasets")
    print("=" * 60)
    print()
    
    create_dataset("bom-agent-tests", BOM_QUERIES)
    create_dataset("supplier-agent-tests", SUPPLIER_QUERIES)
    create_dataset("materials-agent-tests", MATERIALS_QUERIES)
    create_dataset("orchestration-tests", ORCHESTRATION_QUERIES)
    
    print()
    print("=" * 60)
    print("Datasets Created Successfully!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Access Langfuse UI at http://localhost:3000")
    print("2. Navigate to 'Datasets' to view created datasets")
    print("3. Run agents and traces will be linked to dataset items")
    print("4. Use the Playground to test prompts")


def score_trace(trace_id: str, name: str, value: float, comment: str = ""):
    """Add a score to a trace"""
    try:
        langfuse.score(
            trace_id=trace_id,
            name=name,
            value=value,
            comment=comment
        )
        print(f"✓ Scored trace {trace_id}: {name}={value}")
    except Exception as e:
        print(f"✗ Error scoring trace: {e}")


def create_prompt_configs():
    """Create prompt configurations in Langfuse"""
    print("=" * 60)
    print("Creating Prompt Configurations")
    print("=" * 60)
    print()
    
    prompts = [
        {
            "name": "orchestrator-system",
            "prompt": """You are a smart Orchestrator Agent for a supply chain system.

Your goal is to answer user questions by routing them to the correct tools.

STRATEGY:
{{strategy_instructions}}

Provide a complete answer based on the user's specific request.""",
            "config": {
                "model": "gpt-4o-mini",
                "temperature": 0,
            }
        },
        {
            "name": "materials-expert",
            "prompt": """You are a Materials Expert. Use the search_web tool to find details about car parts.

Find the following information:
- OEM Status (Is it typically OEM or Aftermarket?)
- Manufacturer (Top manufacturers)
- Origin Country (Where is it mostly made?)
- Average Price (Provide a price range or average)""",
            "config": {
                "model": "gpt-4o",
                "temperature": 0.3,
            }
        },
        {
            "name": "pestel-analysis",
            "prompt": """You are a Supply Chain Risk Analyst. Perform a PESTEL analysis for supplier '{{supplier_name}}' in '{{country}}'.

Analyze:
1. Political: Stability, trade policies
2. Economic: GDP, inflation, exchange rates
3. Social: Labor market, demographics
4. Technological: Innovation, infrastructure
5. Environmental: Regulations, climate risks
6. Legal: Labor laws, IP protection

Output JSON with risk_score (0-100) and detailed breakdown.""",
            "config": {
                "model": "gemini-2.0-flash",
                "temperature": 0.2,
            }
        }
    ]
    
    for prompt_config in prompts:
        print(f"Creating prompt: {prompt_config['name']}")
        try:
            prompt = langfuse.create_prompt(
                name=prompt_config["name"],
                prompt=prompt_config["prompt"],
                config=prompt_config["config"],
                labels=["production"]
            )
            print(f"  ✓ Prompt created with ID: {prompt.name}")
        except Exception as e:
            print(f"  ℹ️  Prompt might already exist or error: {e}")
    
    print()
    print("=" * 60)
    print("Prompts Created!")
    print("=" * 60)


def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "datasets":
            create_all_datasets()
        elif command == "prompts":
            create_prompt_configs()
        elif command == "all":
            create_all_datasets()
            print()
            create_prompt_configs()
        else:
            print(f"Unknown command: {command}")
            print("Usage: python langfuse_eval.py [datasets|prompts|all]")
    else:
        # Default: create everything
        create_all_datasets()
        print()
        create_prompt_configs()


if __name__ == "__main__":
    main()

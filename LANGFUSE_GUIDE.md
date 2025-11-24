# Langfuse Integration Guide

## Overview
This multi-agent system now includes full Langfuse integration for LLM observability, prompt management, and automated evaluations.

## Quick Start

### 1. Start the System
```bash
docker-compose up --build -d
```

### 2. Access Langfuse
Open Langfuse UI: **http://localhost:3000**

Default credentials will be created on first access.

### 3. Create Evaluation Datasets
```bash
python evals/langfuse_eval.py all
```

This creates:
- BOM agent test dataset
- Supplier agent test dataset
- Materials agent test dataset
- Orchestration test dataset
- Prompt configurations

## What's Integrated

### ✅ Orchestrator Agent (LangChain)
- **File**: `agents/orchestrator/react_agent.py`
- **Tracing**: Full LangChain execution traces
- **Captures**: Agent decisions, tool calls, LLM interactions
- **Callback**: `langfuse.langchain.CallbackHandler`

### ✅ Materials Agent (PydanticAI + OpenAI)
- **File**: `agents/materials-agent/main.py`
- **Tracing**: Request-level tracing with `@observe()` decorator
- **Captures**: Part searches, LLM calls, results
- **Metadata**: Part names, search providers

### ✅ Supplier Agent (RAG)
- **File**: `agents/supplier-agent/rag.py`
- **Tracing**: RAG pipeline steps
- **Captures**: World Bank API calls, WTO data, Gemini analysis
- **Observability**: Each RAG step traced separately

## Features

### 1. LLM Observability
Every AI interaction is traced:
- **Input prompts**
- **Model parameters**
- **Token usage**
- **Latency**
- **Output results**
- **Costs** (calculated automatically)

### 2. Prompt Management
Centralized prompt versioning:
```python
from langfuse import Langfuse

langfuse = Langfuse()
prompt = langfuse.get_prompt("orchestrator-system")
```

Prompts are versioned and can be updated without code deployments.

### 3. Evaluation Framework
Automated testing with datasets:
```bash
# Create datasets and prompts
python evals/langfuse_eval.py all

# Create only datasets
python evals/langfuse_eval.py datasets

# Create only prompts
python evals/langfuse_eval.py prompts
```

## Langfuse UI Walkthrough

### Traces Tab
View all LLM executions:
1. Click "Traces" in left sidebar
2. See real-time traces from all agents
3. Click any trace to see details:
   - Agent decisions
   - Tool calls
   - LLM interactions
   - Timing breakdown

### Datasets Tab
Manage evaluation datasets:
1. Click "Datasets"
2. View test queries for each agent
3. Link traces to dataset items
4. Run experiments

### Prompts Tab
Manage prompt versions:
1. Click "Prompts"
2. View all prompt configs
3. Edit prompts in UI
4. Deploy new versions

### Playground
Test prompts interactively:
1. Click "Playground"
2. Select a prompt
3. Test with different inputs
4. Compare model outputs

## Environment Variables

All agents have these Langfuse env vars:
```yaml
LANGFUSE_SECRET_KEY: sk-lf-secret
LANGFUSE_PUBLIC_KEY: pk-lf-public
LANGFUSE_HOST: http://langfuse:3000
LANGFUSE_ENABLED: "true"  # Set to "false" to disable
```

##  Configuration

### Disable Langfuse
Set environment variable:
```bash
export LANGFUSE_ENABLED=false
```

### Change API Keys
Update in `docker-compose.yml` or `.env`:
```yaml
LANGFUSE_SECRET_KEY: your-secret-key
LANGFUSE_PUBLIC_KEY: your-public-key
```

## Testing with Langfuse

### Run Test Script
```bash
./test_agents.sh
```

All 30 test queries will create traces in Langfuse! 

### View Traces
1. Open http://localhost:3000
2. Navigate to "Traces"
3. See each test execution
4. Analyze performance metrics

### Score Traces
In Langfuse UI:
1. Click on a trace
2. Click "Add Score"
3. Add scores like:
   - `accuracy`: Did it return correct info?
   - `latency`: Was it fast enough?
   - `quality`: LLM-as-judge score

## Evaluation Metrics

### Auto-Tracked
- ✅ **Latency**: Response time for each call
- ✅ **Token Usage**: Input/output tokens
- ✅ **Cost**: Calculated from token usage
- ✅ **Success Rate**: Error tracking

### Manual Scoring
Add custom scores via UI or SDK:
```python
langfuse.score(
    trace_id="...",
    name="accuracy",
    value=0.95,
    comment="Correct supplier identified"
)
```

## Prompt Experiments

Compare prompt versions:
1. Create multiple prompt versions in UI
2. Label them (e.g., "v1", "v2", "production")
3. Run tests with different versions
4. Compare metrics in dashboard

## Best Practices

### 1. Name Your Traces
```python
langfuse_context.update_current_trace(
    name="risk-analysis-bosch",
    user_id="test-user"
)
```

### 2. Add Metadata
```python
langfuse_context.update_current_trace(
    metadata={
        "supplier": "Denso",
        "country": "Japan",
        "test_id": "eval-001"
    }
)
```

### 3. Track Errors
```python
try:
    result = agent.run(...)
except Exception as e:
    langfuse_context.update_current_trace(level="ERROR")
    raise
```

### 4. Group by Sessions
```python
langfuse_context.update_current_trace(
    session_id="user-session-123"
)
```

## Troubleshooting

### Langfuse UI Not Loading
```bash
docker-compose logs langfuse
docker-compose restart langfuse
```

### Traces Not Appearing
1. Check Langfuse is running: `docker-compose ps`
2. Verify env vars in agents
3. Check agent logs for Langfuse errors
4. Ensure `LANGFUSE_ENABLED=true`

### Database Issues
```bash
# Reset Langfuse database
docker-compose down -v
docker-compose up --build -d
```

## Port Configuration

- **Langfuse UI**: 3000
- **Frontend**: 3001 (changed from 3000)
- **Orchestrator**: 8013
- **Materials Agent**: 8012
- **Supplier Agent**: 8011
- **BOM Agent**: 8014

## Next Steps

1. ✅ Run test queries to generate traces
2. ✅ Explore Langfuse UI
3. ✅ Create custom prompts
4. ✅ Build evaluation datasets
5. ✅ Set up automated scoring
6. ✅ Compare prompt versions
7. ✅ Monitor costs and latency

## Resources

- **Langfuse Docs**: https://langfuse.com/docs
- **Python SDK**: https://langfuse.com/docs/sdk/python
- **LangChain Integration**: https://langfuse.com/docs/integrations/langchain
- **Prompt Management**: https://langfuse.com/docs/prompts

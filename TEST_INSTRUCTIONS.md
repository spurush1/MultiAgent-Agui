# Multi-Agent System Test Script

## Overview
This test script validates all agents in the multi-agent system with 30 comprehensive queries.

## Prerequisites
- Docker containers running: `docker-compose up -d`
- `jq` installed for JSON formatting: `brew install jq` (optional but recommended)
- `bc` for calculations (usually pre-installed on macOS)

## Running the Tests

### Quick Start
```bash
./test_agents.sh
```

### What It Tests

#### Category 1: BOM Agent (10 tests)
Tests Neo4j-based Bill of Materials lookups for:
- Various engine types (1.5L Turbo, 3.5L V6, 5.3L V8, etc.)
- Different transmission types (6-speed, 8-speed, CVT, etc.)
- Hybrid powertrains

#### Category 2: Supplier Risk Analysis (10 tests)
Tests Postgres-based supplier risk analysis for:
- **Japanese suppliers**: Denso, Aisin
- **German suppliers**: Bosch, Continental AG, ZF Friedrichshafen
- **American suppliers**: BorgWarner, Delphi, Tenneco
- **Other**: Magna (Canada), Valeo (France)

#### Category 3: Materials Discovery (10 tests)
Tests PydanticAI-powered materials agent for:
- Brake system components
- Engine components
- Electrical parts
- Suspension components

## Expected Output

The script provides:
- ✓ **PASS** (green) for successful tests
- ✗ **FAIL** (red) for failed tests
- HTTP status codes
- JSON responses (prettified with jq)
- Summary with success rate

## Example Output
```
======================================
Test 11: Analyze risk for Denso in Japan
======================================
✓ PASS (HTTP 200)
{
  "supplier_name": "Denso",
  "country": "Japan",
  "risk_score": 15,
  "summary": "...",
  "pestel_data": {...}
}
```

## Test Coverage

- ✅ BOM Agent (Neo4j)
- ✅ Supplier Agent (Postgres)
- ✅ Materials Agent (PydanticAI + Google Search)
- ⚠️ Orchestrator (tested indirectly via agent calls)

## Troubleshooting

### Port Issues
If you changed ports in docker-compose, update these variables at the top of the script:
```bash
ORCHESTRATOR_URL="http://localhost:8013"
BOM_URL="http://localhost:8014"
SUPPLIER_URL="http://localhost:8011"
MATERIALS_URL="http://localhost:8012"
```

### Containers Not Running
```bash
docker-compose ps
docker-compose up -d
```

### See Logs
```bash
docker-compose logs -f [service-name]
# Example: docker-compose logs -f materials-agent
```

## Manual Browser Testing

You can also test via the frontend at `http://localhost:3001`:

### Sample Queries
1. "Analyze risk for Denso in Japan"
2. "Find details about brake pads"
3. "Show me parts for the 3.5L V6 Engine"
4. "Analyze risk for Bosch in Germany"
5. "Find information on turbochargers"

## Success Criteria

- All 30 tests should pass (100% success rate)
- HTTP 200 responses from all agents
- Valid JSON responses with expected data structure

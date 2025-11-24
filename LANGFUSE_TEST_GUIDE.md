# Langfuse Integration Test Report

## Pre-Test Setup

### 1. Start Docker Desktop
Ensure Docker Desktop is running before proceeding.

### 2. Kill Conflicting Processes
```bash
# Kill any process on port 3000 (Langfuse)
lsof -ti:3000 | xargs kill -9

# Kill any process on port 3001 (Frontend)
lsof -ti:3001 | xargs kill -9
```

### 3. Start the System
```bash
# Clean start
docker-compose down -v
docker-compose up --build -d
```

### 4. Wait for Langfuse to Initialize
```bash
# Check logs
docker-compose logs -f langfuse

# Wait for message: "Server listening on port 3000"
```

## Testing Checklist

### ✅ Phase 1: Infrastructure
- [ ] Docker Desktop running
- [ ] All containers started: `docker-compose ps`
- [ ] Langfuse UI accessible: http://localhost:3000
- [ ] Frontend accessible: http://localhost:3001

### ✅ Phase 2: Setup Evaluations
```bash
# From project root
python evals/langfuse_eval.py all
```

Expected output:
```
Creating Langfuse Evaluation Datasets
✅ Created dataset: bom-agent-tests
✅ Created dataset: supplier-agent-tests
✅ Created dataset: materials-agent-tests
✅ Created dataset: orchestration-tests
Creating Prompt Configurations
✓ Prompt created: orchestrator-system
✓ Prompt created: materials-expert
✓ Prompt created: pestel-analysis
```

### ✅ Phase 3: Test Queries with Tracing

#### Test 1: Supplier Risk Analysis
```bash
curl -X POST http://localhost:8011/analyze-risk \
  -H "Content-Type: application/json" \
  -d '{"supplier_name":"Denso","country":"Japan"}'
```

**Expected Result:**
- HTTP 200 response
- JSON with risk_score, summary, pestel_data
- **Trace appears in Langfuse UI**

**Verify in Langfuse:**
1. Open http://localhost:3000
2. Navigate to "Traces"
3. Look for trace with:
   - Name: `generate_pestel_analysis`
   - Metadata: `{\"supplier\": \"Denso\", \"country\": \"Japan\"}`
   - Observations: `fetch_world_bank_data`

#### Test 2: Materials Discovery
```bash
curl -X POST http://localhost:8012/find-material \
  -H "Content-Type: application/json" \
  -d '{"part_name":"brake pads"}'
```

**Expected Result:**
- HTTP 200 response
- MaterialResult JSON with oem_status, manufacturer, etc.
- **Trace appears in Langfuse UI**

**Verify in Langfuse:**
1. Look for trace named `find-material`
2. Metadata should show: `{\"part_name\": \"brake pads\"}`
3. Should see PydanticAI execution

#### Test 3: Orchestrator (End-to-End)
```bash
curl -X POST http://localhost:8013/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Analyze risk for Bosch in Germany"}'
```

**Expected Result:**
- HTTP 200 streaming response
- **Multiple traces in Langfuse**:
  - Main orchestrator trace
  - Tool call to `analyze-risk`
  - Nested supplier-agent trace

**Verify in Langfuse:**
1. Look for orchestrator trace
2. Should see:
   - Agent reasoning
   - Tool selection (`analyze-risk`)
   - Tool execution
   - Final answer
3. Nested trace for supplier agent

### ✅ Phase 4: Run Full Test Suite
```bash
./test_agents.sh
```

**Expected:**
- 30 tests execute
- All create traces in Langfuse
- Success rate displayed

**Verify in Langfuse:**
1. Navigate to "Traces"
2. Filter by last hour
3. Should see 30+ traces
4. Mix of:
   - BOM queries
   - Supplier analytics
   - Materials discoveries

### ✅ Phase 5: Verify Datasets

**In Langfuse UI:**
1. Navigate to "Datasets"
2. Verify 4 datasets exist:
   - `bom-agent-tests`
   - `supplier-agent-tests`
   - `materials-agent-tests`
   - `orchestration-tests`
3. Each should have test items

### ✅ Phase 6: Verify Prompts

**In Langfuse UI:**
1. Navigate to "Prompts"
2. Verify 3 prompts exist:
   - `orchestrator-system`
   - `materials-expert`
   - `pestel-analysis`
3. Each should be labeled "production"

### ✅ Phase 7: Analyze Metrics

**In Langfuse UI:**
1. Click on any trace
2. Verify metrics shown:
   - ✅ Total latency (ms)
   - ✅ Token usage (input/output)
   - ✅ Estimated cost
   - ✅ Timestamp
3. Expand trace to see:
   - Individual LLM calls
   - Tool executions
   - Nested observations

## Troubleshooting

### Issue: Langfuse UI Not Loading
```bash
# Check Langfuse logs
docker-compose logs langfuse

# Check Langfuse DB
docker-compose logs langfuse-db

# Restart Langfuse
docker-compose restart langfuse
```

### Issue: No Traces Appearing
```bash
# Check agent logs
docker-compose logs orchestrator | grep -i langfuse
docker-compose logs materials-agent | grep -i langfuse
docker-compose logs supplier-agent | grep -i langfuse

# Verify env vars
docker-compose exec orchestrator env | grep LANGFUSE
```

### Issue: Agent Errors
```bash
# Check all agent logs
docker-compose logs --tail=50 orchestrator
docker-compose logs --tail=50 materials-agent
docker-compose logs --tail=50 supplier-agent
```

### Issue: Port Conflicts
```bash
# Free up ports
lsof -ti:3000 | xargs kill -9  # Langfuse
lsof -ti:3001 | xargs kill -9  # Frontend
lsof -ti:8011 | xargs kill -9  # Supplier
lsof -ti:8012 | xargs kill -9  # Materials
lsof -ti:8013 | xargs kill -9  # Orchestrator
lsof -ti:8014 | xargs kill -9  # BOM
```

## Success Criteria

### All Green ✅
- [ ] All Docker containers running
- [ ] Langfuse UI accessible
- [ ] 4 datasets created
- [ ] 3 prompts created
- [ ] Test queries return HTTP 200
- [ ] Traces appear in Langfuse UI with:
  - Correct metadata
  - Token usage
  - Latency metrics
  - Nested observations
- [ ] Test suite runs successfully
- [ ] Cost tracking working

## Expected Trace Structure

### Supplier Agent Trace
```
generate_pestel_analysis
├── fetch_world_bank_data
│   └── HTTP request to World Bank API
└── Gemini LLM call
    ├── Input tokens: ~500
    ├── Output tokens: ~300
    └── Cost: $0.001
```

### Materials Agent Trace
```
find-material
├── Metadata: {part_name: "brake pads"}
└── PydanticAI run
    ├── Search tool execution
    └── OpenAI GPT-4o call
        ├── Input tokens: ~400
        ├── Output tokens: ~200
        └── Cost: $0.002
```

### Orchestrator Trace
```
Agent Execution
├── Agent reasoning (GPT-4o-mini)
├── Tool selection: analyze-risk
├── Tool execution
│   └── [Nested supplier-agent trace]
└── Final answer generation
    ├── Total tokens: ~1000
    └── Total cost: $0.005
```

## Next Steps After Testing

1. ✅ Review traces in Langfuse UI
2. ✅ Create custom scores for test traces
3. ✅ Set up automated evaluations
4. ✅ Compare prompt versions
5. ✅ Monitor cost trends
6. ✅ Optimize based on metrics

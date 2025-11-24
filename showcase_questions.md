# Google Antigravity Showcase Questions

Use these questions to demonstrate the full capabilities of the multi-agent system, including orchestration, graph traversal, and risk analysis.

## 1. Multi-Step Orchestration (BOM + Risk)
**Query:** 
> "Who supplies the **Turbocharger** for the Ford F-150, and what are the geopolitical risks associated with that supplier in their country?"

**What it demonstrates:**
- **Chaining:** The Orchestrator must first identify the supplier (BorgWarner) using the `bom-agent` and then analyze them using the `supplier-agent`.
- **Context Passing:** Passing the output of one agent as the input to another.

## 2. Graph Exploration (BOM Agent)
**Query:**
> "Show me the full Bill of Materials for the **3.5L V6 Engine**, including all sub-components and their suppliers."

**What it demonstrates:**
- **Graph Querying:** Direct interaction with the Neo4j graph database via the `bom-agent`.
- **Structured Data:** Returns a complex JSON object with nested children and suppliers, visible in the AGUI JSON component.

## 3. Comparative Analysis (Parallel Execution)
**Query:**
> "Compare the supply chain risks for **Denso** in **Japan** versus **Magna** in **Canada**."

**What it demonstrates:**
- **Parallel/Multiple Tool Calls:** The agent executes the `analyze-risk` tool twice for different entities.
- **Synthesis:** The final response combines data from two separate analysis reports into a coherent comparison.

## 4. PESTEL Risk Analysis (Supplier Agent)
**Query:**
> "What is the impact of recent trade tariffs on **Hitachi**'s ability to supply **Electronics** from **Japan**?"

**What it demonstrates:**
- **Generative Analysis:** Triggers the `supplier-agent`'s RAG/PESTEL generation capability.
- **Domain Specificity:** Extracts specific "Political" and "Economic" factors from the structured risk report.

## 5. Edge Case / Error Handling
**Query:**
> "Analyze the risk for **Tesla** in **Mars**."

**What it demonstrates:**
- **Robustness:** Shows how the system gracefully handles missing data or unknown entities without crashing the UI.


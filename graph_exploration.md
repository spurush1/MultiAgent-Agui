# Graph Exploration (BOM Agent)

```mermaid
sequenceDiagram
    title Graph Exploration (BOM Agent)
    participant User
    participant Frontend
    participant Orchestrator
    participant BOM_Agent

    User->>Frontend: Submit query "Show full Bill of Materials for the V6 Engine"
    Frontend->>Orchestrator: SSE /chat (ReAct LLM)
    Orchestrator->>BOM_Agent: get-bom(part="V6 Engine")
    BOM_Agent-->>Orchestrator: Hierarchical BOM JSON with suppliers
    Orchestrator-->>Frontend: Stream JSON component
    Frontend->>User: Render nested BOM tree in Thinking Drawer
```

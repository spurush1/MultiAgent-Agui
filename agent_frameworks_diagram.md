# Agent Development Frameworks Diagram

```mermaid
flowchart LR
    subgraph Frontend[Frontend]
        FE["React UI (Next.js)"]
    end
    subgraph Orchestrator[Orchestrator]
        ORCH["FastAPI + LangChain (ReAct Agent)"]
    end
    subgraph Materials[Materials Agent]
        MAT[FastAPI Service]
    end
    subgraph BOM[BOM Agent]
        BOMS[FastAPI Service]
    end
    subgraph Supplier[Supplier Agent]
        SUP[FastAPI Service]
    end
    
    FE -->|SSE request| ORCH
    ORCH -->|tool call| MAT
    ORCH -->|tool call| BOMS
    ORCH -->|tool call| SUP
    
    classDef fastapi fill:#f9f,stroke:#333,stroke-width:2px;
    classDef nextjs fill:#bbf,stroke:#333,stroke-width:2px;
    class ORCH,MAT,BOMS,SUP fastapi;
    class FE nextjs;
```

The diagram shows that **all agents are built with FastAPI**, while the **frontend** uses **Next.js (React)**. The Orchestrator combines FastAPI with LangChain to run the ReAct LLM agent.

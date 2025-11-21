from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn
import os

from .database import seed_bom_data, get_db, close_db
from shared.utils import register_agent

app = FastAPI(title="BOM Agent")

@app.on_event("startup")
def on_startup():
    # Wait for Neo4j to be ready in real prod, but for now just try seed
    try:
        seed_bom_data()
    except Exception as e:
        print(f"Startup seeding failed (Neo4j might be warming up): {e}")
        
    # Register with Orchestrator using Agent Card Skills
    skills = [
        {
            "id": "get-bom",
            "name": "Get Bill of Materials",
            "description": "Retrieves the Bill of Materials (BOM) for a specific car part, including its children and suppliers.",
            "inputModes": ["text"],
            "outputModes": ["text", "json"],
            "parameters": {
                "type": "object",
                "properties": {
                    "part_name": {"type": "string", "description": "Name of the part (e.g., 'V6 Engine')"}
                },
                "required": ["part_name"]
            },
            "instructions": "Use this skill FIRST to get the composition and suppliers of a part. Only if this fails or returns no results should you try other discovery tools."
        }
    ]
    register_agent("bom-agent", 8004, skills)

@app.get("/.well-known/agent.json")
def get_agent_card():
    return {
        "name": "bom-agent",
        "description": "Bill of Materials Graph Agent",
        "url": "http://bom-agent:8004",
        "version": "1.0.0",
        "skills": [{"id": "get-bom", "name": "Get Bill of Materials"}]
    }


@app.on_event("shutdown")
def on_shutdown():
    close_db()

class BOMRequest(BaseModel):
    part_name: str

class BOMResponse(BaseModel):
    part_name: str
    children: List[Dict[str, Any]]
    suppliers: List[Dict[str, Any]]

@app.post("/get-bom", response_model=BOMResponse)
def get_bom(request: BOMRequest):
    """
    Retrieves the immediate children and suppliers of a given part.
    """
    query = """
    MATCH (p:Part {name: $part_name})
    OPTIONAL MATCH (p)-[:COMPOSED_OF]->(child:Part)
    OPTIONAL MATCH (p)-[:SUPPLIED_BY]->(s:Supplier)
    RETURN collect(DISTINCT child) as children, collect(DISTINCT s) as suppliers
    """
    
    with get_db() as session:
        result = session.run(query, part_name=request.part_name)
        record = result.single()
        
        if not record:
            raise HTTPException(status_code=404, detail="Part not found")
            
        children = [{"name": c["name"], "type": c["type"]} for c in record["children"] if c]
        suppliers = [{"name": s["name"], "country": s["country"]} for s in record["suppliers"] if s]
        
        return BOMResponse(
            part_name=request.part_name,
            children=children,
            suppliers=suppliers
        )

@app.get("/health")
def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8004)))

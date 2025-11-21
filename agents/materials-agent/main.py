from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn
import os
from openai import OpenAI
import json

from shared.utils import register_agent
from shared.tools.search import get_search_tool

app = FastAPI(title="Materials Agent")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.on_event("startup")
def on_startup():
    # Register with Orchestrator using Agent Card Skills
    skills = [
        {
            "id": "find-material",
            "name": "Find Material Details",
            "description": "Finds details about a car part including OEM status, manufacturer, origin, and average price.",
            "instructions": "Use this skill when you need to find external market data, suppliers, or specifications for a part that are not in the internal BOM.",
            "inputModes": ["text"],
            "outputModes": ["text", "json"],
            "parameters": {
                "type": "object",
                "properties": {
                    "part_name": {"type": "string", "description": "Name of the car part"}
                },
                "required": ["part_name"]
            }
        }
    ]
    register_agent("materials-agent", 8002, skills)

@app.get("/.well-known/agent.json")
def get_agent_card():
    return {
        "name": "materials-agent",
        "description": "Materials Discovery Agent",
        "url": "http://materials-agent:8002",
        "version": "1.0.0",
        "skills": [{"id": "find-material", "name": "Find Material Details"}]
    }


class MaterialRequest(BaseModel):
    part_name: str

class MaterialResponse(BaseModel):
    part_name: str
    oem_status: str
    manufacturer: str
    origin_country: str
    average_price: str
    details: str
    search_query: Optional[str] = None
    search_result: Optional[str] = None

@app.post("/find-material", response_model=MaterialResponse)
def find_material(request: MaterialRequest):
    # Get the configured search tool from Tools Hub
    search_tool = get_search_tool(os.getenv("SEARCH_PROVIDER", "google"))
    
    # 1. Search Web using Tools Hub
    search_query = f"Find details for car part '{request.part_name}': OEM status, manufacturer, country of origin, and average price."
    context = search_tool.search(search_query)
    
    # Extract trace info if available
    search_result = None
    if isinstance(context, dict) and "error" not in context:
        search_result = context.get("result")
    else:
        # Fallback: treat context as raw string
        search_result = context if isinstance(context, str) else None

    # 2. Extract Info using OpenAI with structured output
    prompt = f"""
    You are a Materials Expert. Analyze the following search results for the car part '{request.part_name}':
    
    Search Results:
    {search_result or ''}
    
    Extract the following information:
    - OEM Status (Is it typically OEM or Aftermarket?)
    - Manufacturer (Top manufacturers)
    - Origin Country (Where is it mostly made?)
    - Average Price (Provide a price range or average)
    
    Return JSON format:
    {{
        "oem_status": "...",
        "manufacturer": "...",
        "origin_country": "...",
        "average_price": "...",
        "details": "Short summary..."
    }}
    """
    
    try:
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        content = completion.choices[0].message.content
        data = json.loads(content)
        
        return MaterialResponse(
            part_name=request.part_name,
            oem_status=data.get("oem_status", "Unknown"),
            manufacturer=data.get("manufacturer", "Unknown"),
            origin_country=data.get("origin_country", "Unknown"),
            average_price=data.get("average_price", "Unknown"),
            details=data.get("details", "No details found."),
            search_query=search_query,
            search_result=search_result,
        )
    except Exception as e:
        print(f"LLM Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8002)))

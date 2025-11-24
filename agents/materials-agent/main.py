from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import uvicorn
import os
from pydantic_ai import Agent, RunContext
from langfuse import get_client

from shared.utils import register_agent
from shared.tools.search import get_search_tool, SearchInterface

app = FastAPI(title="Materials Agent")

# --- PydanticAI Agent Setup ---

class MaterialResult(BaseModel):
    part_name: str
    oem_status: str = Field(description="Is it typically OEM or Aftermarket?")
    manufacturer: str = Field(description="Top manufacturers")
    origin_country: str = Field(description="Where is it mostly made?")
    average_price: str = Field(description="Price range or average")
    details: str = Field(description="Short summary of the part")

# Define the agent with correct syntax
material_agent = Agent(
    'openai:gpt-4o',
    deps_type=SearchInterface,
    output_type=MaterialResult,  # Correct parameter name!
    system_prompt="You are a Materials Expert. Use the search_web tool to find details about car parts."
)

@material_agent.tool
async def search_web(ctx: RunContext[SearchInterface], query: str) -> str:
    """Search the web for information."""
    result = ctx.deps.search(query)
    if "error" in result:
        return f"Search Error: {result['error']}"
    return result.get("result", "No results found.")

# --- FastAPI Endpoints ---

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

@app.post("/find-material", response_model=MaterialResult)
async def find_material(request: MaterialRequest):
    # Get the configured search tool (dependency)
    search_tool = get_search_tool(os.getenv("SEARCH_PROVIDER", "google"))
    
    # Initialize Langfuse client
    langfuse = get_client()
    
    # Create trace with context manager
    with langfuse.start_as_current_observation(
        as_type="generation",
        name="find-material",
        input={"part_name": request.part_name},
        metadata={"agent": "materials", "provider": os.getenv("SEARCH_PROVIDER", "google")}
    ) as observation:
        try:
            # Run the PydanticAI agent
            result = await material_agent.run(
                f"Find details for car part '{request.part_name}': OEM status, manufacturer, country of origin, and average price.",
                deps=search_tool
            )
            
            # Update observation with output
            observation.update(output=result.output.dict())
            
            # Flush Langfuse to ensure trace is sent
            langfuse.flush()
            
            return result.output
            
        except Exception as e:
            print(f"Agent Error: {e}")
            observation.update(level="ERROR", status_message=str(e))
            langfuse.flush()  # Flush even on error
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8002)))

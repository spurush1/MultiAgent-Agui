from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import StructuredTool
from langchain.pydantic_v1 import BaseModel, Field, create_model
import requests
import json
from typing import List, Any

from .registry import registry

def create_dynamic_tool(agent_url: str, tool_name: str, description: str, parameters: dict):
    """
    Creates a LangChain tool that forwards calls to the remote agent.
    """
    # 1. Create Pydantic model for args dynamically
    fields = {}
    for param_name, param_info in parameters.get("properties", {}).items():
        # Simplified type mapping
        field_type = str
        if param_info.get("type") == "integer":
            field_type = int
        fields[param_name] = (field_type, Field(description=param_info.get("description", "")))
    
    ArgsModel = create_model(f"{tool_name}Args", **fields)

    # 2. Define the function to call
    def func(**kwargs):
        # Construct the endpoint URL. Assuming a standard /tool-execution or specific endpoint convention?
        # For this demo, we'll assume the tool name maps to an endpoint path or a generic /execute
        # Let's assume the agent registered with a specific endpoint for the tool or we map it.
        # Simplified: POST {agent_url}/{tool_name_kebab_case}
        
        # Convert tool_name (e.g. "get_supplier_risk") to url path (e.g. "analyze-risk")
        # This mapping needs to be robust. For now, we'll rely on the tool name matching the endpoint path logic
        # or pass the endpoint in the registration.
        
        # Let's assume the tool name IS the endpoint path for simplicity in this demo.
        endpoint = f"{agent_url}/{tool_name}"
        
        try:
            response = requests.post(endpoint, json=kwargs)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return f"Error calling {tool_name}: {e}"

    return StructuredTool.from_function(
        func=func,
        name=tool_name,
        description=description,
        args_schema=ArgsModel
    )

def get_react_agent():
    """
    Re-creates the agent executor with the current set of registered tools.
    """
    tools = []
    for agent_name, agent in registry.agents.items():
        for skill in agent.skills:
            tool = create_dynamic_tool(
                agent.url, 
                skill.id, # Use ID as the tool name/endpoint suffix
                skill.description, 
                skill.parameters
            )
            tools.append(tool)
            
    if not tools:
        # Return a dummy agent if no tools yet
        return None

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a smart Orchestrator Agent for a supply chain system. 
        
        Your goal is to answer user questions by routing them to the correct tools.
        
        STRATEGY:
        1. If the user asks about a part's suppliers or composition, FIRST try `get-bom`.
        2. If `get-bom` returns an error (like 404) or empty results (no suppliers found), YOU MUST try `find-material`.
        3. `find-material` is capable of searching the web for real-world suppliers and details.
        4. ONLY use `analyze-risk` if the user explicitly asks for risk analysis, geopolitical data, or supplier evaluation. Do not call it for simple BOM or part lookup queries.
        
        Provide a complete answer based on the user's specific request.
        """),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)
    
    return agent_executor

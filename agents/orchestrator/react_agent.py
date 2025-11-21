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
    tool_instructions = []

    for agent_name, agent in registry.agents.items():
        for skill in agent.skills:
            tool = create_dynamic_tool(
                agent.url, 
                skill.id, # Use ID as the tool name/endpoint suffix
                skill.description, 
                skill.parameters
            )
            tools.append(tool)
            
            # Collect instructions
            instruction = skill.instructions if skill.instructions else skill.description
            tool_instructions.append(f"- **{skill.id}**: {instruction}")
            
    if not tools:
        # Return a dummy agent if no tools yet
        return None

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Dynamic Strategy Construction
    strategy_text = "\n".join(tool_instructions)
    
    system_prompt = f"""You are a smart Orchestrator Agent for a supply chain system. 
    
    Your goal is to answer user questions by routing them to the correct tools.
    
    AVAILABLE TOOLS & STRATEGY:
    {strategy_text}
    
    Provide a complete answer based on the user's specific request.
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)
    
    return agent_executor

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import StructuredTool
from langchain.pydantic_v1 import BaseModel, Field, create_model
from langfuse.langchain import CallbackHandler
import requests
import json
import os
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
    Dynamically constructs a ReAct agent using registered tools and their instructions.
    """
    tools = []
    
    # Build tools from registry
    for agent_name, agent in registry.agents.items():
        for skill in agent.skills:
            tool = create_dynamic_tool(
                agent_url=agent.url,
                tool_name=skill.id,
                description=skill.description,
                parameters=skill.parameters
            )
            tools.append(tool)
    
    # Build dynamic system prompt from instructions
    strategy_instructions = []
    for agent_name, agent in registry.agents.items():
        for skill in agent.skills:
            if hasattr(skill, 'instructions') and skill.instructions:
                strategy_instructions.append(f"- {skill.instructions}")
    
    strategy_section = "\n".join(strategy_instructions) if strategy_instructions else "Use the available tools to answer user questions."
    
    system_message = f"""You are a smart Orchestrator Agent for a supply chain system.

Your goal is to answer user questions by routing them to the correct tools.

STRATEGY:
{strategy_section}

Provide a complete answer based on the user's specific request.
"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    agent = create_openai_tools_agent(llm, tools, prompt)
    
    # Initialize Langfuse callback handler
    langfuse_handler = None
    if os.getenv("LANGFUSE_ENABLED", "true").lower() == "true":
        try:
            langfuse_handler = CallbackHandler(
                secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
                public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
                host=os.getenv("LANGFUSE_HOST"),
            )
        except Exception as e:
            print(f"Warning: Could not initialize Langfuse: {e}")
    
    # Create executor with Langfuse callback
    callbacks = [langfuse_handler] if langfuse_handler else []
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, callbacks=callbacks)
    
    return agent_executor

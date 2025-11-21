from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
from enum import Enum

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
from enum import Enum

# --- Google Agent-to-Agent (A2A) Protocol ---

class AgentSkill(BaseModel):
    id: str
    name: str
    description: str
    inputModes: List[str] = ["text"] # e.g. "text", "audio"
    outputModes: List[str] = ["text"]
    parameters: Dict[str, Any] # JSON Schema for the skill input
    instructions: Optional[str] = None # Instructions for the orchestrator on when/how to use this skill

class AgentCapabilities(BaseModel):
    streaming: bool = False
    pushNotifications: bool = False
    stateTransitionHistory: bool = False

class AgentCard(BaseModel):
    name: str
    description: str
    url: str
    version: str = "1.0.0"
    provider: str = "Google Antigravity"
    capabilities: AgentCapabilities = Field(default_factory=AgentCapabilities)
    skills: List[AgentSkill]
    
    # Optional: Authentication scheme (simplified for this demo)
    authentication: Dict[str, Any] = Field(default_factory=lambda: {"type": "none"})

# Legacy aliases for compatibility if needed, but better to switch fully
AgentRegistration = AgentCard

class AgentQuery(BaseModel):
    task: str
    context: Dict[str, Any] = Field(default_factory=dict)

class AgentResponse(BaseModel):
    status: str = "success" # success, error
    data: Any
    metadata: Dict[str, Any] = Field(default_factory=dict)

# --- AGUI Protocol (Frontend) ---

class AGUIComponentType(str, Enum):
    TEXT = "text"
    MARKDOWN = "markdown"
    RISK_CARD = "risk-card"
    BOM_TREE = "bom-tree"
    SUPPLIER_TABLE = "supplier-table"
    JSON = "json"

class AGUIComponent(BaseModel):
    type: AGUIComponentType
    data: Dict[str, Any]
    title: Optional[str] = None
    id: Optional[str] = None

class AGUIMessage(BaseModel):
    message: str
    components: List[AGUIComponent] = Field(default_factory=list)

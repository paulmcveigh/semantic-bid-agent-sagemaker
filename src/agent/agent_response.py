from dataclasses import dataclass, field
from typing import Any, Dict, List
from semantic_kernel.agents import ChatHistoryAgentThread
from agent_message import AgentMessage

@dataclass
class AgentResponse:
    messages: List[AgentMessage]
    thread: ChatHistoryAgentThread
    metrics: Dict[str, Any] = field(default_factory=dict)

import asyncio
import os
import json
import requests
import faiss
import numpy as np
import sqlite3
import streamlit as st
import random
import boto3
from boto3.dynamodb.conditions import Key, Attr

from dataclasses import dataclass, field, asdict
from typing import Annotated, Optional, TypedDict, List, Any, Dict

from sentence_transformers import SentenceTransformer
import pandas as pd
import openai

from semantic_kernel import Kernel
from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread
from semantic_kernel.functions import KernelArguments, kernel_function
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.open_ai_prompt_execution_settings import OpenAIChatPromptExecutionSettings
from semantic_kernel.contents import ChatMessageContent, FunctionCallContent, FunctionResultContent, TextContent
from semantic_kernel.connectors.ai.bedrock.bedrock_prompt_execution_settings import BedrockChatPromptExecutionSettings
from semantic_kernel.connectors.ai.bedrock.services.bedrock_chat_completion import BedrockChatCompletion
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
#from semantic_kernel.connectors.ai.azure_openai import AzureChatCompletion
from semantic_kernel.contents import ChatHistory
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
#from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.open_ai_prompt_execution_settings import OpenAIChatPromptExecutionSettings
from semantic_kernel.agents import BedrockAgent, BedrockAgentThread


from azure_methods import FailureScoreChecker, RiskEvaluator, InsurancePremiumEstimator
from agent_text_processing import VectorMemoryRAGPlugin, StructureClaimData

#api_key = st.secrets["api_keys"]["OPENAI_API_KEY"]
openai_api_type = "azure"
openai_key = st.secrets["openai"]["AZURE_OPENAI_API_KEY"]
openai_endpoint = st.secrets["openai"]["AZURE_OPENAI_ENDPOINT"]
openai_version = st.secrets["openai"]["AZURE_OPENAI_API_VERSION"]
openai_deployment_name = st.secrets["openai"]["AZURE_OPENAI_DEPLOYMENT_NAME"]



endpoint = st.secrets["cosmosdb"]["endpoint"]
key = st.secrets["cosmosdb"]["key"]
database_name = st.secrets["cosmosdb"]["database"]
container_name = st.secrets["cosmosdb"]["container"]


# --- Instructions
AGENT_INSTRUCTIONS = """You are an expert financial risk analyst specialising in small business lending. Your name, if asked, is 'FRA'.

Wait for specific instructions from the user before taking any action. Do not perform tasks unless they are explicitly requested.

You may be asked to:
- Estimate the potential insurance premium
- Recommend whether a loan or overdraft request should be approved, based on a risk threshold
- Reference insights from the database or other risk analytics sources to support your recommendation, and if the user asks for info from the database please provide it

If a large document/email or business application has been pasted into the chat, use StructureClaimData to structure its contents and use the output for any function that takes a `loan_data` parameter.

Keep responses brief—no more than a few paragraphs—and always respond only to what the user has asked, when they ask it.

For example:
- If the user only asks for credit risk, only give the credit risk
- If they only ask whether to approve the loan, give a clear recommendation with minimal justification
- If user asks for insurance premium, only give it
- If they only ask for supporting data or insights, do not provide a risk score or decision

Do not offer next steps, summaries, or additional actions unless explicitly prompted.

"""


@dataclass
class AgentMessage:
    role: str
    content: Optional[str] = None
    name: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None
    function_response: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AgentResponse:
    messages: List[AgentMessage]
    thread: ChatHistoryAgentThread
    metrics: Dict[str, Any] = field(default_factory=dict)

from azure.cosmos import CosmosClient, exceptions
from typing import Annotated




# --- Main async entrypoint
async def main(
    user_input: str, 
    thread: Optional[ChatHistoryAgentThread] = None, 
    claim_text: Optional[str] = None
) -> AgentResponse:
    kernel = Kernel()
    kernel.add_service(AzureChatCompletion(
        deployment_name="kainosgpt",
        endpoint=openai_endpoint,
        api_key=openai_key
    ))

    messages: List[AgentMessage] = []

    # 👉 Keep RAG setup for policy lookup
    vector_memory_rag = VectorMemoryRAGPlugin()
    if claim_text:
        vector_memory_rag.add_document(claim_text)

    # --- Register plugins
    kernel.add_plugin(FailureScoreChecker(), plugin_name="FailureScoreChecker")
    #kernel.add_plugin(DataCollector(kernel), plugin_name="collector")    
    kernel.add_plugin(vector_memory_rag, plugin_name="VectorMemoryRAG")
    kernel.add_plugin(RiskEvaluator(), plugin_name="RiskModel")
    kernel.add_plugin(InsurancePremiumEstimator(), plugin_name="PremiumEstimator")
    #kernel.add_plugin(ConsumerDutyChecker(kernel), plugin_name="ConsumerDuty")
    kernel.add_plugin(StructureClaimData(kernel), plugin_name="StructureClaimData")

    

    agent = ChatCompletionAgent(
        kernel=kernel,
        name="FRA",
        instructions=AGENT_INSTRUCTIONS,
        arguments=KernelArguments(
            settings=OpenAIChatPromptExecutionSettings(
                temperature=0.5,
                top_p=0.95,
                function_choice_behavior=FunctionChoiceBehavior.Auto()
            )
        )
    )

    intermediate_steps: List[AgentMessage] = []
    metrics = {
        "total_tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "steps": 0
    }

    messages.append(AgentMessage(role="user", content=user_input))

    async def handle_intermediate(message: ChatMessageContent):
        for item in message.items:
            if isinstance(item, FunctionCallContent):
                intermediate_steps.append(AgentMessage(
                    role="function_call",
                    name=item.name,
                    function_call={"name": item.name, "arguments": item.arguments},
                ))
            elif isinstance(item, FunctionResultContent):
                intermediate_steps.append(AgentMessage(
                    role="function_response",
                    name=item.name,
                    function_response=item.result,
                ))
            else:
                intermediate_steps.append(AgentMessage(
                    role=message.role.value,
                    content=message.content
                ))

    async for response in agent.invoke(
        messages=user_input,
        thread=thread,
        on_intermediate_message=handle_intermediate
    ):
        message = AgentMessage(
            role=response.role,
            content=response.content,
            name=getattr(response, "name", None)
        )

        if hasattr(response, "function_call") and response.function_call:
            message.function_call = {
                "name": response.function_call.name,
                "arguments": response.function_call.arguments
            }

        if hasattr(response, "function_response") and response.function_response:
            message.function_response = response.function_response

        if hasattr(response, "metadata") and response.metadata:
            message.metadata = response.metadata
            if "usage" in message.metadata:
                usage = message.metadata["usage"]
                metrics["prompt_tokens"] += usage.prompt_tokens
                metrics["completion_tokens"] += usage.completion_tokens
                metrics["total_tokens"] = metrics["prompt_tokens"] + metrics["completion_tokens"]

        messages.extend(intermediate_steps)
        metrics["steps"] += len(intermediate_steps)
        intermediate_steps.clear()
        messages.append(message)
        metrics["steps"] += 1
        thread = response.thread

    return AgentResponse(
        messages=messages,
        thread=thread,
        metrics=metrics
    )

# --- Debug runner
if __name__ == "__main__":
    async def test():
        response = await main("produce a bid to sell a new AI note taking product to Microsoft")
        for msg in response.messages:
            print(f"\n[{msg.role}]")
            if msg.content:
                print(msg.content.strip())
            if msg.function_call:
                print(f"Function: {msg.function_call['name']}")
            if msg.function_response:
                print(f"Result: {str(msg.function_response)[:100]}...")
        print(f"\nMetrics: {response.metrics}")

    asyncio.run(test())

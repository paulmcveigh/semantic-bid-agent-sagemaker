from semantic_kernel import Kernel
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.functions import KernelArguments
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
import streamlit as st

from kernel_functions.failure_score_checker import FailureScoreChecker
from kernel_functions.risk_evaluator import RiskEvaluator
from kernel_functions.insurance_premium_estimator import InsurancePremiumEstimator
from kernel_functions.structure_claim_data import StructureClaimData
from kernel_functions.vector_memory import VectorMemoryRAGPlugin

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


openai_api_type = "azure"
openai_key = st.secrets["openai"]["AZURE_OPENAI_API_KEY"]
openai_endpoint = st.secrets["openai"]["AZURE_OPENAI_ENDPOINT"]
openai_version = st.secrets["openai"]["AZURE_OPENAI_API_VERSION"]
openai_deployment_name = st.secrets["openai"]["AZURE_OPENAI_DEPLOYMENT_NAME"]


AGENT_INSTRUCTIONS = """You are an expert insurance underwriting consultant. Your name, if asked, is 'IUA'.

Wait for specific instructions from the user before taking any action. Do not perform tasks unless they are explicitly requested.

You may be asked to:
- Assess the risk profile of an organisation based on model outputs. Please check the database first then run this
- Estimate the likely insurance premium using our model. Please check the database first then run this
- Reference insights from a database to assist underwriting decisions

If a large document has been pasted into the chat, use StructureClaimData to structure its contents and use the output for any function that takes a `claim_data` parameter.

Keep responses brief—no more than a few paragraphs—and always respond only to what the user has asked, when they ask it. 
For example 
- If the user only asks for risk rating only give the risk rating 
- If they only ask for insurance premium only give the insurance premium, do not run both models unless you are asked to in the prompt
- If they only ask for insights from the database do not give risk or insurance premium scores.
"""

def build_agent(claim_text):
    kernel = Kernel()
    kernel.add_service(AzureChatCompletion(
        deployment_name="kainosgpt",
        endpoint=openai_endpoint,
        api_key=openai_key
    ))



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

    return agent

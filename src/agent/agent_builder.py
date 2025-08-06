import os
import json
import streamlit as st
from typing import Annotated
# Plugin classes
from kernel_functions.failure_score_checker import FailureScoreChecker
from kernel_functions.risk_evaluator import RiskEvaluator
from kernel_functions.survivability_estimator import SurvivabilityEstimator
from kernel_functions.counterfactual import Counterfactual
from kernel_functions.structure_claim_data import StructureClaimData
from kernel_functions.vector_memory import VectorMemoryRAGPlugin
# Semantic Kernel core
from semantic_kernel import Kernel
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.functions import KernelArguments
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.open_ai_prompt_execution_settings import OpenAIChatPromptExecutionSettings


openai_api_type = "azure"
openai_key = st.secrets["openai"]["AZURE_OPENAI_API_KEY"]
openai_endpoint = st.secrets["openai"]["AZURE_OPENAI_ENDPOINT"]
openai_version = st.secrets["openai"]["AZURE_OPENAI_API_VERSION"]
openai_deployment_name = st.secrets["openai"]["AZURE_OPENAI_DEPLOYMENT_NAME"]


AGENT_INSTRUCTIONS = """You are an assistant for people approving loans to small businesses. Your name, if asked, is 'FRA'.

Wait for specific instructions from the user before taking any action. Do not perform tasks unless they are explicitly requested.

You may be asked to:
- Assess the credit risk profile of an organisation based on model outputs, we are trying to predict the chance of the business of defaulting on the loan. Please check the database first then run this
- Check the survivability of a small business using our model, i.e. how long they are expected to survive
- Use our counterfactuals model to figure out what it would take for a company to be approved

If a large document has been pasted into the chat, use StructureClaimData to structure its contents and use the output for any function that takes a `claim_data` parameter.

Keep responses brief—no more than a few paragraphs—and always respond only to what the user has asked, when they ask it. 
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
    kernel.add_plugin(vector_memory_rag, plugin_name="VectorMemoryRAG")
    kernel.add_plugin(RiskEvaluator(), plugin_name="RiskModel")
    kernel.add_plugin(Counterfactual(), plugin_name="Counterfactual")
    kernel.add_plugin(SurvivabilityEstimator(), plugin_name="SurvivabilityEstimator")
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

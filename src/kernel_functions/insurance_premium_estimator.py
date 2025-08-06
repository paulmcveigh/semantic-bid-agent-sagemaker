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


import boto3
import json
import numpy as np
from typing import Annotated

#self.runtime = boto3.client("sagemaker-runtime")
#self.endpoint_name = "claim-amount-linear-v2-endpoint"

class InsurancePremiumEstimator:
    def __init__(self):
        self.runtime = "testruntime"
        self.endpoint_name = "testendpoint"

    @kernel_function(description="Calculate the survivability of a business, i.e. the length of time they are expected to survive")
    async def estimate_size(
        self,
        claim_data: Annotated[dict, "Structured company data."]
    ) -> dict:

        return {
            "years": 2.5,
            "service_used": self.runtime,
            "model_used": self.endpoint_name 
        }

'''coverage_amount = claim_data.get("coverage_amount", "")
region_of_operation = claim_data.get("region_of_operation", "").lower()
#coverage_amount_str = claim_data.get("coverage_amount", "").lower()
#region_of_operation = claim_data.get("region_of_operation", "").lower()
#cleaned = coverage_amount_str.lower().replace(",", "")
# Extract digits only
#digits = ''.join(filter(str.isdigit, cleaned))
# Convert to integer and scale down
coverage_amount = int(coverage_amount) // 1000 if coverage_amount else 0

if region_of_operation == "gb":
    region_value = 0
elif region_of_operation == "usa":
    region_value = 1
elif region_of_operation == "eu":
    region_value = 2
elif region_of_operation == "asia":
    region_value = 3
elif region_of_operation == "africa":
    region_value = 4
else:
    region_value = 5

payload = f"{coverage_amount},{region_value}"

#try:
response = self.runtime.invoke_endpoint(
    EndpointName=self.endpoint_name,
    ContentType="text/csv",
    Body=payload
)
result = json.loads(response["Body"].read().decode())
prediction = result["predictions"][0]["score"]

return {
    "estimated_insurance_premium": round(prediction, 2),
    "currency": "GBP",
    "service_used": self.runtime,
    "model_used": self.endpoint_name 
}'''

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

class VectorMemoryRAGPlugin:
    def __init__(self):
        self.text_chunks = []
        self.index = None
        self.embeddings = SentenceTransformer("all-MiniLM-L6-v2")

    def add_document(self, doc_text: str, chunk_size: int = 500):
        self.text_chunks = [
            doc_text[i:i + chunk_size]
            for i in range(0, len(doc_text), chunk_size)
        ]
        vectors = self.embeddings.encode(self.text_chunks, convert_to_numpy=True)
        dim = vectors.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(vectors)

    @kernel_function(description="retrieve relevant chunks from uploaded claim documents.")
    async def retrieve_chunks(self, query: Annotated[str, "Query to summmarise / retrieve relevant claim information"]) -> str:
        if not self.index:
            return "No documents indexed yet."
        query_vec = self.embeddings.encode([query], convert_to_numpy=True)
        D, I = self.index.search(query_vec, k=3)
        relevant_chunks = [self.text_chunks[i] for i in I[0] if i < len(self.text_chunks)]
        return "\n---\n".join(relevant_chunks)

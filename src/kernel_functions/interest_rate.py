import streamlit as st
from typing import Annotated
from semantic_kernel.functions import kernel_function

class InterestRate:
    def __init__(self):
        self.runtime = "testruntime"
        self.endpoint_name = "fraud-detection-xgb-v1-endpoint"

    @kernel_function(description="Use a model to predict the chance small business has of defaulting on the loan")
    async def interest_rate(
        self,
        claim_data: Annotated[dict, "Structured claim data with fields like coverage_amount and region_of_operation."],
        risk_score = 100,
        survival_prob = 0.6
    ) -> dict:
        base_rate = 0.05  # 5% base
        risk_adj = min(risk_score * 0.1, 0.10)  # up to +10%
        survival_adj = max((1 - survival_prob) * 0.05, 0)  # up to +5%
        return {
            "interest_rate": round(base_rate + risk_adj + survival_adj, 4),
            "service_used": self.runtime,
            "model_used": self.endpoint_name
        }

import streamlit as st
from typing import Annotated
from semantic_kernel.functions import kernel_function

#self.runtime = boto3.client("sagemaker-runtime")
#self.endpoint_name = "claim-amount-linear-v2-endpoint"

class Counterfactual:
    def __init__(self):
        self.runtime = "testruntime"
        self.endpoint_name = "testendpoint"

    @kernel_function(description="Figure out what it would take for a company to get approved")
    async def estimate_size(
        self,
        claim_data: Annotated[dict, "Structured company data."]
    ) -> dict:

        return {
            "verdict": "They need to reduce the amount they are asking for",
            "service_used": self.runtime,
            "model_used": self.endpoint_name 
        }

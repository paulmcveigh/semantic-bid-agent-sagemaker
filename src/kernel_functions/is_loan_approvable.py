import streamlit as st
from typing import Annotated
from semantic_kernel.functions import kernel_function

class IsLoanApprovable:
    def __init__(self):
        self.runtime = "testruntime"
        self.endpoint_name = "fraud-detection-xgb-v1-endpoint"

    @kernel_function(description="Should the loan be approved")
    async def is_loan_approvable(
        self,
        claim_data: Annotated[dict, "Structured company data."],
        risk_score = 60,
        survival_prob = 0.6,
        loan_amount = 1000000
    ) -> dict:
        if risk_score > 0.5 or survival_prob < 0.5:
            return {
                verdict: False
            }
        if loan_amount > 100_000 and risk_score > 0.3:
            return {
                verdict: False
            }
        return {
            verdict: True
        }


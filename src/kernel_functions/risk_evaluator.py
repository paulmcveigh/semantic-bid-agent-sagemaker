
##################################################################

##################################################################

#self.runtime = boto3.client("sagemaker-runtime")
#self.endpoint_name = "fraud-detection-xgb-v1-endpoint"

#TODO: Adapt to db data
class RiskEvaluator:
    def __init__(self):
        self.runtime = "testruntime"
        self.endpoint_name = "fraud-detection-xgb-v1-endpoint"

    @kernel_function(description="Recommend whether a loan or overdraft request should be approved, based on a risk threshold")
    async def assess_risk(
        self,
        claim_data: Annotated[dict, "Structured claim data with fields like coverage_amount and region_of_operation."]
    ) -> dict:
        
        return {
            "risk_score": 0.48,
            "model_used": "fraud-detection-xgb-v1-endpoint"
        }

'''return {
    "risk_score": 0.48,
    "service_used": self.runtime,
    "model_used": self.endpoint_name 
}'''

'''coverage_amount = claim_data.get("coverage_amount", "")

region_of_operation = claim_data.get("region_of_operation", "").lower()
organisation_name =  claim_data.get("organisation_name", "").lower()

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
    region_value = 5'''
'''
#payload = f"{coverage_amount},{region_value}"
payload = f"1,150000000"


response = self.runtime.invoke_endpoint(
    EndpointName=self.endpoint_name,
    ContentType="text/csv",
    Body=payload
)
prediction = json.loads(response["Body"].read().decode())

return {
    "risk_score": prediction
}

rationale = f"Risk for organisation: {organisation_name} assessed as {prediction} based on our model."

return {
    "risk_score": prediction,
    "explanation": rationale
}
'''



######################################################

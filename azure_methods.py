

#################################################################
#TODO: Adapt to use db after chat with Tania
class FailureScoreChecker:
    @kernel_function(description="Retrieve the failure score and failure score commentary for an organisation from the database.")
    async def retrieve_failure_rating(
        self,
        claim_data: Annotated[dict, "Structured claim object containing organisation_name."]
    ) -> dict:

        client = CosmosClient(endpoint, key)
        database = client.get_database_client(database_name)
        container = database.get_container_client(container_name)

        items = list(container.read_all_items())
        print("Items:")
        print(items)
        return items

        '''
        dynamodb = boto3.resource('dynamodb', region_name="eu-north-1")
        dnb_table = dynamodb.Table("dnb_data")
        organisation_name = claim_data.get("organisation_name", "N/A")
        '''
        '''
        name = organisation_name
        df = pd.read_csv("data\\dnb.csv")
        risk_score = df.loc[df.organization_name==name, 'climate_risk_score'].values[0]
        return {
            "organisation_name": name,
            "climate_risk_score": risk_score
        }
        '''
        '''
        response = dnb_table.scan()
        items = response.get('Items', [])

        return items
        '''
        '''
        response = dnb_table.scan(
            FilterExpression='organization_name = :org',
            ExpressionAttributeValues={':org': {'S': organisation_name}},
            ProjectionExpression='organization_name, climate_risk_score'
        )

        item = response['Items'][0] if response['Items'] else None

        return {
            "organisation_name": organisation_name,
            "climate_risk_score": item['climate_risk_score']['S'] if item else None
        }'''

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

    @kernel_function(description="Estimate the likely insurance premium range using model in GBP.")
    async def estimate_size(
        self,
        claim_data: Annotated[dict, "Structured company data."]
    ) -> dict:

        return {
            "estimated_insurance_premium": 100000,
            "currency": "GBP",
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

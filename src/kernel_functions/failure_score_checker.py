import streamlit as st
from typing import Annotated
from semantic_kernel.functions import kernel_function  
from azure.cosmos import CosmosClient 

endpoint = st.secrets["cosmosdb"]["endpoint"]
key = st.secrets["cosmosdb"]["key"]
database_name = st.secrets["cosmosdb"]["database"]
container_name = st.secrets["cosmosdb"]["container"]


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


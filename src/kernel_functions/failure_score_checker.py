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
    @kernel_function(description="Retrieve data from the database")
    async def retrieve_failure_rating(
        self,
        claim_data: Annotated[dict, "Structured claim object containing organisation_name."]
    ) -> dict:

        sectors = ['Consumer Goods', 'Technology & Communications', 'Financials', 'Health Care', 'Transportation', 'Services', 'Infrastructure', 'Resource Transformation']
        purposes = ['Working Capital', 'Equipment', 'Expansion', 'Product Development']
        locations = ["London", "Manchester", "Birmingham", "Leeds", "Glasgow"]
        
         df = pd.DataFrame({
            "sector": np.random.choice(sectors, n),
            "location": np.random.choice(locations, n),
            "years_operating": np.random.randint(0, 20, size=n),
            "num_employees": np.random.poisson(15, size=n).clip(min=1),
            "annual_revenue": np.random.lognormal(mean=12, sigma=0.5, size=n),
            "profit_margin": np.round(np.random.normal(loc=5, scale=10, size=n), 1),
            "late_payments": np.random.poisson(1.5, size=n).clip(0, 10),
            "credit_score": np.clip(np.random.normal(loc=600, scale=100, size=n), 300, 1000).astype(int),
            "existing_debt": np.clip(np.random.normal(loc=80000, scale=50000, size=n), 0, None),
            "purpose": np.random.choice(purposes, size=n),
            "productivity_gain": np.clip(np.round(np.random.normal(loc=10, scale=10, size=n), 1), 0, None),
            "dnb_risk_score": np.random.randint(1, 100, size=n),
            "loan_amount": np.clip(np.random.normal(loc=50000, scale=20000, size=n), 5000, None),
            "term_months": np.random.choice([12, 18, 24, 36, 48, 60], size=n, p=[0.1, 0.2, 0.3, 0.2, 0.1, 0.1]),
            "current_ratio": np.random.uniform(0.5, 3.0, size=n),
            "debt_to_equity": np.random.uniform(0.1, 3.0, size=n)
        })

        return {
            data: df
        }

        
        '''client = CosmosClient(endpoint, key)
        database = client.get_database_client(database_name)
        container = database.get_container_client(container_name)

        items = list(container.read_all_items())
        print("Items:")
        print(items)
        return items'''




        
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


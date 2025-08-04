import requests
import pandas as pd
import json

df = pd.DataFrame({"user_id": ["userA", "userB", "userZ"]})

# Convert dataframe to json records
data_json = json.dumps({"dataframe_records": df.to_dict(orient="records")})

# POST request
response = requests.post(
    "http://localhost:1234/invocations",
    headers={"Content-Type": "application/json"},
    data=data_json
)

print(response.json())

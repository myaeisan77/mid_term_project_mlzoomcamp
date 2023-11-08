
import requests


url = "http://localhost:9697/predict"

client = {"fixed_acidity": 7.4, "residual_sugar": 1.9, "density": 0.9978}


response = requests.post(url, json=client).json()
print(response)
 


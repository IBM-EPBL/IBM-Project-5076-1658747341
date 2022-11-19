from flask import Flask, render_template, request, session, url_for, redirect
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os, types
from botocore.client import Config
import ibm_boto3
import requests

app = Flask(__name__)

@app.route('/')

def index():
	return render_template("home.html")

@app.route('/analyze', methods = ["POST", "GET"])

def analyze():
	try:
		cylinders = int(request.form.get("cylinders"))
		displacement = int(request.form.get("displacement"))
		hp = int(request.form.get("hp"))
		weight = int(request.form.get("weight"))
		acceleration = int(request.form.get("acceleration"))
		year = int(request.form.get("year"))
		origin = int(request.form.get("origin")) % 100
		test = pd.DataFrame()

		cos_client = ibm_boto3.client(service_name='s3',
		    ibm_api_key_id='bmTZ1tDihoDNN3IgTrgFVwo7l1Lgc9kntdKX4gUvHR1x',
		    ibm_auth_endpoint="https://iam.cloud.ibm.com/oidc/token",
		    config=Config(signature_version='oauth'),
		    endpoint_url='https://s3.us.cloud-object-storage.appdomain.cloud')
		bucket = 'vehicleperformanceanalyzer-donotdelete-pr-vbny2haofpbejk'
		object_key = 'Car_Performance.csv'
		body = cos_client.get_object(Bucket=bucket,Key=object_key)['Body']
		if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )
		dataset = pd.read_csv(body)

		API_KEY = "hS4CGi9PpGEJH4LVNkKH9oys9lA0dughL6i0Sxm2bmh6"
		token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey":
		 API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
		mltoken = token_response.json()["access_token"]
		header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}
		payload_scoring = {"input_data": [{"values": [[cylinders, displacement, hp, weight, acceleration, year, origin]]}]}
		response_scoring = requests.post('https://us-south.ml.cloud.ibm.com/ml/v4/deployments/52402c81-c0dd-4cb0-aaf4-c4c11c60ab3f/predictions?version=2022-11-19', json=payload_scoring,
		 headers={'Authorization': 'Bearer ' + mltoken})
		performance = response_scoring.json()['predictions'][0]['values'][0]


		sd = StandardScaler()
		x = dataset[['cylinders','displacement','horsepower','weight','acceleration','model year','origin']].values
		y = dataset[['mpg']].values
		x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=0)
		x_train = sd.fit_transform(x_train)
		x_test = sd.fit_transform(x_test)
		y_train = sd.fit_transform(y_train)
		y_test = sd.fit_transform(y_test)
		performance = sd.inverse_transform([performance])[0][0]
		if performance < 10:
			desc = 1
		elif 10 <= performance < 15:
			desc = 2 
		elif 15 <= performance < 25:
			desc = 3 
		elif 25 <= performance < 40:
			desc = 4
		elif performance >= 40:
			desc = 5
		return render_template("home.html", performance = "{:.2f}".format(performance), desc = desc)

	except Exception as e:
		return render_template("home.html", error = e)
import torch
from torch import nn
import numpy as np
import requests
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import joblib
import time
from flask import Flask, jsonify, request



# Define the LSTM model class with an additional hidden layer
device = 'cpu'

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # Forward pass through LSTM layer
        out, _ = self.lstm(x, (h0, c0))
        out = self.relu(out)
        # Get output from the last time step
        out = self.fc(out[:, -1, :])
        return out

start = time.time()

model = LSTMModel(5, 128, 1)
model.load_state_dict(torch.load('trained_lstm_model.pt', map_location=torch.device(device)))
model.eval()

print(f"Model loaded in {time.time() - start}s")



#Windspeed method
def get_predictions(lat, lon, height):
    h = 2
    if height >= 10:
        h = 10
    if height >= 100:
        h = 100


    now = datetime.utcnow()  # Use utcnow() because Open-Meteo uses UTC by default
    start_datetime = now - timedelta(hours=24)
    start_date = start_datetime.strftime('%Y-%m-%d')
    end_date = now.strftime('%Y-%m-%d')
        
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": f"temperature_2m,windspeed_{h}m,relative_humidity_2m,precipitation",
        "start_date" : start_date,
        "end_date" : end_date #96 15-minute intervals interpolated from hourly readings
    }
    response = requests.get(url, params=params)
    data = response.json()['hourly']
    df = pd.DataFrame(data)
    df['time'] = pd.to_datetime(df['time'])
    df = df[(df['time'] >= start_datetime) & (df['time'] <= now)] #Only for the last day, predicting today
    
    
    #Input format is hour_seq, windspeed_seq, temp_seq, humidity_seq, precip_seq
    temp_scaler = joblib.load('minmax_scaler.joblib')
    df['temperature_2m'] = temp_scaler.transform(
        [[i] for i in df['temperature_2m']]
    )

    input_data = []
    hours_passed = 0
    for _, row in df.iterrows():
        for _ in range(4):
            input_data.append([
                hours_passed, row[f'windspeed_{h}m'], row[f'temperature_2m'],
                row[f'relative_humidity_2m'], row['precipitation']
            ])
            hours_passed += 0.15

    input_data = torch.from_numpy(np.array([input_data])).to(torch.float32)
    start = time.time()
    output_data = model(input_data)
    print(f"Model inferred in {time.time() - start}s")


    #Using the formula with Area swept = 5, p = 1.225, C = 0.3
    wind_speed = output_data[0].item()
    power = 0.5 * (wind_speed * wind_speed * wind_speed) * 1.225 * 5 * 0.3
    return (wind_speed, power)



#API
app = Flask(__name__)

@app.route('/', methods = ['GET'])
def home():
    return "Hello from api"

@app.route('/power', methods = ['POST'])
def power():
    body = request.get_json()
    wind_speed, power = get_predictions(
        body['lat'], body['lon'], body['height']
    )
    return jsonify({
        "wind_speed" : wind_speed,
        "power" : power
    })


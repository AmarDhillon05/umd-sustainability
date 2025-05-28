import torch
from torch import nn
import numpy as np
import requests
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import joblib
from flask import Flask, jsonify, request

# ===== Device setup =====
device = torch.device("cpu")  # You could allow switching if you deploy on GPU

# ===== Model definition =====
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.relu(out)
        out = self.fc(out[:, -1, :])
        return out

# ===== Load model and scaler once at startup =====
model = LSTMModel(5, 128, 1).to(device)
model.load_state_dict(torch.load('trained_lstm_model.pt', map_location=device))
model.eval()

temp_scaler = joblib.load('minmax_scaler.joblib')

# ===== Inference logic =====
def get_predictions(lat, lon, height):
    try:
        height = max(min(height, 100), 2)  # Clamp between 2m and 100m
        wind_heights = [2, 10, 100]
        if height not in wind_heights:
            h = min(wind_heights, key=lambda x: abs(x - height))
        else:
            h = height

        now = datetime.utcnow()
        start_datetime = now - timedelta(hours=24)

        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "temperature_2m,windspeed_2m,windspeed_10m,windspeed_100m,relative_humidity_2m,precipitation",
            "start_date": start_datetime.strftime('%Y-%m-%d'),
            "end_date": now.strftime('%Y-%m-%d')
        }

        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json().get("hourly", {})
        df = pd.DataFrame(data)
        df['time'] = pd.to_datetime(df['time'])
        df = df[(df['time'] >= start_datetime) & (df['time'] <= now)]

        # Transform temp
        df['temperature_2m'] = temp_scaler.transform(df[['temperature_2m']])

        # Choose closest valid windspeed column
        ws_key = f'windspeed_{h}m'
        for fallback in ['windspeed_10m', 'windspeed_2m', 'windspeed_100m']:
            if ws_key in df.columns and df[ws_key].notna().all():
                break
            ws_key = fallback

        # Build input tensor
        input_data = []
        hours_passed = 0.0
        for _, row in df.iterrows():
            for _ in range(4):  # Interpolate into 15-min intervals
                input_data.append([
                    hours_passed,
                    row.get(ws_key, 0),
                    row['temperature_2m'],
                    row['relative_humidity_2m'],
                    row['precipitation']
                ])
                hours_passed += 0.15

        input_tensor = torch.tensor([input_data], dtype=torch.float32, device=device)

        with torch.no_grad():
            output = model(input_tensor)

        wind_speed = output[0].item()
        power = 0.5 * wind_speed**3 * 1.225 * 5 * 0.3  # Area=5, p=1.225, Cp=0.3

        del input_tensor, output  # Explicit cleanup
        return wind_speed, power

    except Exception as e:
        return str(e), None

# ===== Flask App =====
app = Flask(__name__)

@app.route('/')
def home():
    return "Wind Power Prediction API"

@app.route('/power', methods=['POST'])
def power():
    try:
        body = request.get_json()
        lat = body['lat']
        lon = body['lon']
        height = body.get('height', 10)

        wind_speed, power = get_predictions(lat, lon, height)
        if power is None:
            return jsonify({"error": wind_speed}), 500

        return jsonify({
            "wind_speed": wind_speed,
            "power": power
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Optional for deployment: don't run app.run() in production setups
if __name__ == "__main__":
    app.run()

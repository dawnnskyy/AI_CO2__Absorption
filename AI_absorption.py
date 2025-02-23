#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      pranj
#
# Created:     23-02-2025
# Copyright:   (c) pranj 2025
# Licence:     <your licence>
#---------------------------------
# AI-Powered CO2 Monitoring & Absorption Simulation
# Google Colab Notebook for AI-based CO2 monitoring with Dashboard and Real Sensor Integration

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from flask import Flask, render_template, jsonify
import threading
import time
import serial  # For reading from real sensors
import firebase_admin
from firebase_admin import credentials, db
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

# Initialize Serial Communication for Sensors (Modify COM Port as needed)
SERIAL_PORT = "COM3"  # Change to your port (e.g., /dev/ttyUSB0 on Linux)
BAUD_RATE = 9600
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
except serial.SerialException:
    ser = None  # Handle case where serial device is not available

# Firebase Setup
try:
    cred = credentials.Certificate("firebase_credentials.json")  # Replace with your Firebase credentials file
    db_app = firebase_admin.initialize_app(cred, {'databaseURL': "https://your-database-url.firebaseio.com/"})
    firebase_enabled = True
except Exception as e:
    print("Firebase initialization failed:", e)
    firebase_enabled = False

# Train AI Model (Linear Regression for Prediction)
data = {
    'Temperature_C': np.random.uniform(20, 40, 100),  # Temperature in degrees Celsius
    'Humidity_%': np.random.uniform(40, 80, 100),  # Humidity percentage
    'CO2_ppm': np.random.uniform(300, 800, 100)  # CO2 concentration in ppm
}
df = pd.DataFrame(data)
df['Absorption_Efficiency'] = 100 - (df['CO2_ppm'] / 8)  # Simulated efficiency based on CO2 levels

X = df[['Temperature_C', 'Humidity_%', 'CO2_ppm']]
y = df['Absorption_Efficiency']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

# Visualization of Predictions
plt.figure(figsize=(10,5))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='dashed', color='red')
plt.xlabel("Actual Absorption Efficiency")
plt.ylabel("Predicted Absorption Efficiency")
plt.title("AI Model Prediction vs Actual Values")
plt.show()

print(f"Mean Absolute Error: {mae:.2f}")
print("Model Training Complete. Predicts CO2 absorption efficiency based on environmental factors.")

# Flask Web Dashboard with Real Sensor Integration
app = Flask(__name__)

def get_sensor_data():
    """Reads real-time sensor data from connected sensors via Serial or generates simulated data."""
    if ser:
        ser.write(b'R')  # Request data from the sensor
        line = ser.readline().decode('utf-8').strip()
        try:
            temp, humidity, co2 = map(float, line.split(','))
        except ValueError:
            temp, humidity, co2 = np.random.uniform(20, 40), np.random.uniform(40, 80), np.random.uniform(300, 800)
    else:
        temp, humidity, co2 = np.random.uniform(20, 40), np.random.uniform(40, 80), np.random.uniform(300, 800)

    sensor_data = {
        'Temperature_C': round(temp, 2),
        'Humidity_%': round(humidity, 2),
        'CO2_ppm': round(co2, 2)
    }

    # Upload to Firebase if enabled
    if firebase_enabled:
        db.reference("sensor_data").push(sensor_data)

    return sensor_data

@app.route('/')
def home():
    return render_template('dashboard.html')

@app.route('/data')
def get_data():
    sensor_data = get_sensor_data()
    sensor_data['Absorption_Efficiency'] = model.predict(
        np.array([[sensor_data['Temperature_C'], sensor_data['Humidity_%'], sensor_data['CO2_ppm']]]).reshape(1, -1)
    )[0]
    return jsonify(sensor_data)

# Dash Web Application for Visualization
dash_app = dash.Dash(__name__, server=app, routes_pathname_prefix='/dashboard/')

dash_app.layout = html.Div([
    html.H1("Real-Time CO2 Absorption Monitoring"),
    dcc.Graph(id='live-graph'),
    dcc.Interval(
        id='interval-component',
        interval=5000,
        n_intervals=0
    )
])

@dash_app.callback(
    Output('live-graph', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_graph(n):
    sensor_data = get_sensor_data()
    fig = {
        'data': [
            {'x': ['Temperature', 'Humidity', 'CO2 Level'], 'y': [sensor_data['Temperature_C'], sensor_data['Humidity_%'], sensor_data['CO2_ppm']], 'type': 'bar', 'name': 'Sensor Data'}
        ],
        'layout': {
            'title': 'Real-Time CO2 Absorption Data'
        }
    }
    return fig

# Run Flask server in a separate thread
def run_flask():
    app.run(debug=True, use_reloader=False)

flask_thread = threading.Thread(target=run_flask)
flask_thread.start()

print("Flask server running. Access the dashboard in a web browser at /dashboard/")
import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import math
import sys
import os
import numpy as np
import uuid
import time

# --- FIX IMPORT PATH ---
# Get current directory: .../VDAWS/src/map
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get project root: .../VDAWS
project_root = os.path.dirname(os.path.dirname(current_dir))
# Add to path
sys.path.append(project_root)

# Local Imports
from src.map.object_manager import ObjectManager
from src.map.flying_object import FlyingObject
from src.scanning.remoteid_sniffer import run_sniffer_thread

# --- CONFIGURATION ---
WIFI_INTERFACE = 'Wi-Fi'  # Change to 'wlan0' or 'en0' if on Linux/Mac

# --- 1. LIVE TRACKING SETUP ---
manager = ObjectManager(timeout_seconds=5)
run_sniffer_thread(manager, interface=WIFI_INTERFACE)

# --- 2. SIMULATION SETUP (Restored) ---
# Center Map (San Francisco for simulation)
CENTER_LAT = 37.76
CENTER_LON = -122.43

TOTAL_FRAMES = 100
PATH_MOVEMENT_SCALE = 1.5 

def generate_path_coordinates(steps):
    paths = [[], [], []]
    radius = 0.01 * PATH_MOVEMENT_SCALE # Reduced radius slightly for better zoom
    for i in range(steps):
        t = i / steps 
        angle = t * 2 * np.pi
        
        # Drone 1: Circle
        paths[0].append((CENTER_LAT + radius * np.sin(angle), CENTER_LON + radius * np.cos(angle), 100))
        
        # Drone 2: Figure 8
        scale = radius * 1.5
        d2_lat = CENTER_LAT + (scale * np.sin(angle) * np.cos(angle)) / (1 + np.sin(angle)**2)
        d2_lon = CENTER_LON + (scale * np.cos(angle)) / (1 + np.sin(angle)**2)
        paths[1].append((d2_lat, d2_lon, 500))
        
        # Drone 3: Line
        paths[2].append((CENTER_LAT + (radius * 1.5 * np.sin(angle)), CENTER_LON - 0.02, 800))
    return paths

path_data = generate_path_coordinates(TOTAL_FRAMES)

# Initialize Simulated Drones
current_time = int(time.time())
simulated_drones = [
    FlyingObject.create_with_id(101, CENTER_LAT, CENTER_LON, 100.0, 0.0, 0.0, 0.0, current_time),
    FlyingObject.create_with_id(102, CENTER_LAT, CENTER_LON, 500.0, 0.0, 0.0, 0.0, current_time),
    FlyingObject.create_with_id(103, CENTER_LAT, CENTER_LON, 800.0, 0.0, 0.0, 0.0, current_time)
]

# --- DASH APP ---
app = dash.Dash(__name__)

app.layout = html.Div([
    html.Div([
        html.H3("VDAWS - Drone Tracking", style={'margin': '0', 'color': 'white'}),
        html.Div(id='status-indicator', style={'color': 'lime', 'fontWeight': 'bold'})
    ], style={'position': 'absolute', 'z-index': '10', 'left': '20px', 'top': '20px', 'backgroundColor': 'rgba(0,0,0,0.5)', 'padding': '10px', 'borderRadius': '5px'}),

    dcc.Graph(id='map-graph', style={'height': '100vh'}),
    dcc.Interval(id='interval-component', interval=200, n_intervals=0), # 200ms = 5fps
])

@app.callback(
    [Output('map-graph', 'figure'),
     Output('status-indicator', 'children'),
     Output('status-indicator', 'style')],
    Input('interval-component', 'n_intervals')
)
def update_map(n):
    # 1. Check for LIVE data
    active_objects = manager.get_active_objects()
    
    mode_text = "MODE: LIVE TRACKING"
    mode_style = {'color': '#00ff00', 'fontWeight': 'bold'} # Green

    # 2. If no live data, use SIMULATION
    if not active_objects:
        mode_text = "MODE: SIMULATION (No Live Signal)"
        mode_style = {'color': '#ffaa00', 'fontWeight': 'bold'} # Orange
        
        frame_idx = int(n) % TOTAL_FRAMES
        next_frame_idx = (int(n) + 1) % TOTAL_FRAMES
        
        # Update Simulated Positions
        for i, drone in enumerate(simulated_drones):
            target_pos = path_data[i][frame_idx]
            future_pos = path_data[i][next_frame_idx]
            
            # Simple velocity approx
            vx = (future_pos[1] - target_pos[1]) * 10000 # Scale up for visibility
            vy = (future_pos[0] - target_pos[0]) * 10000
            
            drone.set_position(target_pos[0], target_pos[1], target_pos[2])
            drone.set_velocity(vx, vy, 0.0)
            
        active_objects = simulated_drones

    # 3. Prepare Data for Plotly
    node_data = []
    arrow_data = []
    ARROW_OFFSET = 0.00025 

    for obj in active_objects:
        d_lat = float(obj.x)
        d_lon = float(obj.y)
        d_alt = float(obj.altitude)
        vx = float(obj.velocity[0])
        vy = float(obj.velocity[1])
        
        node_data.append({
            'lat': d_lat, 'lon': d_lon, 'alt': d_alt,
            'id': f"ID:{obj.id}", 'size': 15
        })

        speed_mag = math.sqrt(vx**2 + vy**2)
        # Lower threshold for simulation visibility
        if speed_mag > 0.00001: 
            norm_vx = vx / speed_mag
            norm_vy = vy / speed_mag
            LON_SCALE = 1.0 / math.cos(math.radians(d_lat))
            
            head_lat = d_lat + (norm_vy * ARROW_OFFSET)
            head_lon = d_lon + (norm_vx * ARROW_OFFSET * LON_SCALE)
            
            arrow_data.append({
                'lat': head_lat, 'lon': head_lon,
                'speed': speed_mag, 'size': 8
            })

    df_nodes = pd.DataFrame(node_data)
    df_arrows = pd.DataFrame(arrow_data)

    # 4. Build Map
    if df_nodes.empty:
        fig = px.scatter_mapbox(lat=[CENTER_LAT], lon=[CENTER_LON], zoom=12)
    else:
        # Auto-center on the first drone
        center_lat = df_nodes.iloc[0]['lat']
        center_lon = df_nodes.iloc[0]['lon']

        fig = px.scatter_mapbox(
            df_nodes, lat="lat", lon="lon", color="alt", size="size",
            size_max=15, zoom=13, hover_name="id",
            color_continuous_scale="Jet", range_color=[0, 500],
            center={"lat": center_lat, "lon": center_lon}
        )

    if not df_arrows.empty:
        fig.add_trace(go.Scattermapbox(
            lat=df_arrows['lat'], lon=df_arrows['lon'],
            mode='markers',
            marker=go.scattermapbox.Marker(size=10, color='red'),
            hoverinfo='skip'
        ))

    fig.update_layout(
        mapbox_style="open-street-map",
        margin={"r":0, "t":0, "l":0, "b":0},
        showlegend=False,
        uirevision='constant_loop' 
    )
    
    return fig, mode_text, mode_style

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
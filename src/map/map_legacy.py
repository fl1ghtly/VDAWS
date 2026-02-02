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

# --- IMPORT PATH FIX ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

# Local Imports
from src.map.object_manager import ObjectManager
from src.map.flying_object import FlyingObject
from src.scanning.remoteid_sniffer import run_sniffer_thread
# --- NEW IMPORT ---
from src.map.collision_detector import CollisionDetector

# --- CONFIGURATION ---
WIFI_INTERFACE = 'Wi-Fi' 
CENTER_LAT = 37.76
CENTER_LON = -122.43
TOTAL_FRAMES = 100
PATH_MOVEMENT_SCALE = 1.5 

# --- 1. SETUP ---
manager = ObjectManager(timeout_seconds=5)
# run_sniffer_thread(manager, interface=WIFI_INTERFACE) # Uncomment for live

# Initialize Collision Detector (50m radius, look 10s into future)
collision_detector = CollisionDetector(warning_radius_meters=150.0, prediction_horizon_seconds=10.0)

# --- 2. SIMULATION DATA ---
def generate_path_coordinates(steps):
    paths = [[], [], []]
    radius = 0.01 * PATH_MOVEMENT_SCALE 
    for i in range(steps):
        t = i / steps 
        angle = t * 2 * np.pi
        paths[0].append((CENTER_LAT + radius * np.sin(angle), CENTER_LON + radius * np.cos(angle), 100))
        scale = radius * 1.5
        d2_lat = CENTER_LAT + (scale * np.sin(angle) * np.cos(angle)) / (1 + np.sin(angle)**2)
        d2_lon = CENTER_LON + (scale * np.cos(angle)) / (1 + np.sin(angle)**2)
        paths[1].append((d2_lat, d2_lon, 100)) # Changed alt to 100 to force collision for demo
        paths[2].append((CENTER_LAT + (radius * 1.5 * np.sin(angle)), CENTER_LON - 0.02, 800))
    return paths

path_data = generate_path_coordinates(TOTAL_FRAMES)
current_time = int(time.time())
simulated_drones = [
    FlyingObject.create_with_id(101, CENTER_LAT, CENTER_LON, 100.0, 0.0, 0.0, 0.0, current_time),
    FlyingObject.create_with_id(102, CENTER_LAT, CENTER_LON, 100.0, 0.0, 0.0, 0.0, current_time),
    FlyingObject.create_with_id(103, CENTER_LAT, CENTER_LON, 800.0, 0.0, 0.0, 0.0, current_time)
]

# --- DASH APP ---
app = dash.Dash(__name__)

app.layout = html.Div([
    html.Div([
        html.H3("VDAWS - Drone Tracking", style={'margin': '0', 'color': 'white'}),
        html.Div(id='status-indicator', style={'fontWeight': 'bold'}),
        html.Div(id='collision-alert', style={'color': 'red', 'fontWeight': 'bold', 'marginTop': '5px'})
    ], style={'position': 'absolute', 'z-index': '10', 'left': '20px', 'top': '20px', 'backgroundColor': 'rgba(0,0,0,0.6)', 'padding': '15px', 'borderRadius': '5px'}),

    dcc.Graph(id='map-graph', style={'height': '100vh'}),
    dcc.Interval(id='interval-component', interval=200, n_intervals=0),
])

@app.callback(
    [Output('map-graph', 'figure'),
     Output('status-indicator', 'children'),
     Output('status-indicator', 'style'),
     Output('collision-alert', 'children')],
    Input('interval-component', 'n_intervals')
)
def update_map(n):
    # 1. Get Objects (Live or Sim)
    active_objects = manager.get_active_objects()
    mode_text = "MODE: LIVE TRACKING"
    mode_style = {'color': '#00ff00', 'fontWeight': 'bold'}

    if not active_objects:
        mode_text = "MODE: SIMULATION"
        mode_style = {'color': '#ffaa00', 'fontWeight': 'bold'}
        
        frame_idx = int(n) % TOTAL_FRAMES
        next_frame_idx = (int(n) + 1) % TOTAL_FRAMES
        
        for i, drone in enumerate(simulated_drones):
            target_pos = path_data[i][frame_idx]
            future_pos = path_data[i][next_frame_idx]
            
            # Calculate simulated velocity (Degrees per frame -> approx Degrees per sec)
            fps = 5.0
            vx = (future_pos[0] - target_pos[0]) * fps 
            vy = (future_pos[1] - target_pos[1]) * fps
            vz = (future_pos[2] - target_pos[2]) * fps
            
            drone.set_position(target_pos[0], target_pos[1], target_pos[2])
            drone.set_velocity(vx, vy, vz)
            
        active_objects = simulated_drones

    # 2. RUN COLLISION DETECTION
    collision_events = collision_detector.detect_collisions(active_objects)
    
    alert_text = ""
    collision_lines_lat = []
    collision_lines_lon = []

    if collision_events:
        alert_text = f"⚠️ COLLISION WARNING: {len(collision_events)} Predicted!"
        
        # Create line segments between colliding drones
        obj_map = {obj.id: obj for obj in active_objects}
        for event in collision_events:
            if event.drone_a_id in obj_map and event.drone_b_id in obj_map:
                d1 = obj_map[event.drone_a_id]
                d2 = obj_map[event.drone_b_id]
                
                # Add line segment (with None to break connection between different pairs)
                collision_lines_lat.extend([d1.x, d2.x, None])
                collision_lines_lon.extend([d1.y, d2.y, None])

    # 3. Prepare Visuals
    node_data = []
    arrow_data = []
    ARROW_OFFSET = 0.00025 

    for obj in active_objects:
        node_data.append({
            'lat': obj.x, 'lon': obj.y, 'alt': obj.altitude,
            'id': f"ID:{obj.id}", 'size': 15
        })

        # Visualization for direction vectors
        speed_mag = math.sqrt(obj.velocity[0]**2 + obj.velocity[1]**2)
        if speed_mag > 0.0:
            norm_vx = obj.velocity[0] / speed_mag
            norm_vy = obj.velocity[1] / speed_mag
            # Correct vector orientation for visual map (lat is Y, lon is X)
            head_lat = obj.x + (norm_vx * ARROW_OFFSET) 
            head_lon = obj.y + (norm_vy * ARROW_OFFSET)
            
            arrow_data.append({'lat': head_lat, 'lon': head_lon, 'size': 8})

    df_nodes = pd.DataFrame(node_data)
    
    # Base Map
    if df_nodes.empty:
        fig = px.scatter_mapbox(lat=[CENTER_LAT], lon=[CENTER_LON], zoom=12)
    else:
        center_lat = df_nodes.iloc[0]['lat']
        center_lon = df_nodes.iloc[0]['lon']
        fig = px.scatter_mapbox(
            df_nodes, lat="lat", lon="lon", color="alt", size="size",
            size_max=15, zoom=13, hover_name="id",
            color_continuous_scale="Jet", range_color=[0, 500],
            center={"lat": center_lat, "lon": center_lon}
        )

    # Layer: Direction Arrows
    if arrow_data:
        df_arrows = pd.DataFrame(arrow_data)
        fig.add_trace(go.Scattermapbox(
            lat=df_arrows['lat'], lon=df_arrows['lon'],
            mode='markers', marker=go.scattermapbox.Marker(size=8, color='white'),
            hoverinfo='skip', name='Heading'
        ))

    # Layer: Collision Lines (Red)
    if collision_lines_lat:
        fig.add_trace(go.Scattermapbox(
            lat=collision_lines_lat, lon=collision_lines_lon,
            mode='lines',
            line=dict(width=4, color='red'),
            hoverinfo='skip', name='Collision Course'
        ))

    fig.update_layout(
        mapbox_style="open-street-map",
        margin={"r":0, "t":0, "l":0, "b":0},
        showlegend=False,
        uirevision='constant_loop'
    )
    
    return fig, mode_text, mode_style, alert_text

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
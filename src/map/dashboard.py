import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State, dash_table
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import math
import sys
import os
import numpy as np
import time
import json
from flask import request, jsonify

# --- SETUP PATHS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir)) 

# --- LOCAL IMPORTS ---
from flying_object import FlyingObject
from object_manager import ObjectManager
from collision_detector import CollisionDetector
from scanning.remoteid_sniffer import run_sniffer_thread

# --- CONFIGURATION ---
config_path = '/app/config.json'

try:
    with open(config_path, 'r') as f:
        config = json.load(f)
        grid_min = config.get('voxel_tracer', {}).get('grid_min', [-117.84, 33.64])
        CENTER_LON = grid_min[0]
        CENTER_LAT = grid_min[1]
except Exception as e:
    print(f"[WARNING] Could not load config.json: {e}")
    CENTER_LAT = 33.64
    CENTER_LON = -117.84

TOTAL_FRAMES = 100
PATH_MOVEMENT_SCALE = 1.5 
WIFI_INTERFACE = 'Wi-Fi' 

GRID_STATE = {
    "min": None,
    "max": None,
    "active": False
}

SYSTEM_STATE = {
    "has_received_live_data": False
}

# --- INITIALIZE MANAGERS ---
manager = ObjectManager(timeout_seconds=5)
collision_detector = CollisionDetector(warning_radius_meters=150.0)

# --- START SNIFFER ---
try:
    run_sniffer_thread(manager, interface=WIFI_INTERFACE)
    print(f"[SUCCESS] Sniffer thread started on {WIFI_INTERFACE}")
except Exception as e:
    print(f"[ERROR] Sniffer failed: {e}")

# --- SIMULATION DATA SETUP ---
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
        paths[1].append((d2_lat, d2_lon, 100)) 
        paths[2].append((CENTER_LAT + (radius * 1.5 * np.sin(angle)), CENTER_LON - 0.02, 800))
    return paths

path_data = generate_path_coordinates(TOTAL_FRAMES)
current_time = int(time.time())
simulated_drones = [
    FlyingObject.create_with_id(101, CENTER_LAT, CENTER_LON, 100.0, 0.0, 0.0, 0.0, current_time),
    FlyingObject.create_with_id(102, CENTER_LAT, CENTER_LON, 100.0, 0.0, 0.0, 0.0, current_time),
    FlyingObject.create_with_id(103, CENTER_LAT, CENTER_LON, 800.0, 0.0, 0.0, 0.0, current_time)
]

# --- DASHBOARD APP ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
server = app.server 

# --- API ENDPOINTS ---
@server.route('/update_parameters', methods=['POST'])
def update_parameters():
    try:
        data = request.get_json()
        GRID_STATE["min"] = data.get("grid_min")
        GRID_STATE["max"] = data.get("grid_max")
        GRID_STATE["active"] = True
        return jsonify({"status": "success"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@server.route('/stream_objects', methods=['POST'])
def stream_objects():
    try:
        data = request.get_json()
        if not data or 'objects' not in data:
            return jsonify({"status": "error"}), 400
            
        for obj in data['objects']:
            manager.update_object(
                id=obj.get('id'),
                lat=obj.get('lat', 0.0),
                lon=obj.get('lon', 0.0),
                alt=obj.get('alt', 0.0),
                vx=obj.get('vx', 0.0),
                vy=obj.get('vy', 0.0),
                vz=obj.get('vz', 0.0)
            )
        return jsonify({"status": "success"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# --- LAYOUT ---
app.layout = dbc.Container([
    dcc.Store(id='filter-store', data=0),
    dbc.Row([dbc.Col(html.H2("VDAWS Live Tracker", className="text-center text-primary mb-4"), width=12)], className="mt-3"),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(["Live Airspace ", html.Span(id='live-badge', className="badge bg-secondary")]),
                dbc.CardBody([dcc.Graph(id='map-graph', style={'height': '70vh'})], style={'padding': '0'}) 
            ])
        ], width=8),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Filter & Status"),
                dbc.CardBody([
                    html.Div(id='status-indicator', className="mb-2"),
                    html.Div(id='collision-alert', className="mb-3"),
                    html.Label("Min Avg Velocity (m/s):"),
                    dbc.InputGroup([
                        dbc.Input(id='velocity-input', type='number', value=0, min=0, step=0.5),
                        dbc.Button("Update", id='velocity-btn', color="primary", n_clicks=0)
                    ], className="mb-3"),
                    # NEW: Reset to Sim Button
                    dbc.Button("Reset to Simulation", id='reset-sim-btn', color="warning", size="sm", className="w-100 mt-2")
                ])
            ], className="mb-3"),
            dbc.Card([
                dbc.CardHeader("Tracked Objects"),
                dbc.CardBody([
                    dash_table.DataTable(
                        id='object-table',
                        columns=[
                            {"name": "ID", "id": "id"},
                            {"name": "Alt (m)", "id": "alt"},
                            {"name": "Avg Spd", "id": "spd"},
                        ],
                        style_cell={'textAlign': 'left', 'backgroundColor': '#303030', 'color': 'white'},
                        style_header={'backgroundColor': '#1a1a1a', 'fontWeight': 'bold'},
                        style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': '#383838'}]
                    )
                ])
            ], style={'height': '35vh', 'overflowY': 'scroll'})
        ], width=4) 
    ]),
    dcc.Interval(id='interval-component', interval=500, n_intervals=0),
], fluid=True)

# --- CALLBACKS ---
@app.callback(
    Output('filter-store', 'data'),
    [Input('velocity-btn', 'n_clicks'),
     Input('reset-sim-btn', 'n_clicks')],
    State('velocity-input', 'value')
)
def handle_inputs(v_clicks, r_clicks, v_value):
    ctx = dash.callback_context
    if not ctx.triggered:
        return 0
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'reset-sim-btn':
        # Clear the manager and reset the persistent flag
        manager.objects.clear()
        SYSTEM_STATE["has_received_live_data"] = False
        return 0
        
    return float(v_value) if v_value is not None else 0

@app.callback(
    [Output('map-graph', 'figure'),
     Output('object-table', 'data'),
     Output('status-indicator', 'children'),
     Output('collision-alert', 'children'),
     Output('live-badge', 'children'),
     Output('live-badge', 'className')],
    [Input('interval-component', 'n_intervals'),
     Input('filter-store', 'data')] 
)
def update_dashboard(n, min_velocity):
    active_objects = manager.get_active_objects()
    
    # Stay in Live Mode if data was ever seen
    if active_objects or SYSTEM_STATE["has_received_live_data"]:
        if active_objects:
            SYSTEM_STATE["has_received_live_data"] = True
            
        status_text, status_color = f"● LIVE TRACKING ({len(active_objects)} detected)", "#00ff00"
        badge_text, badge_class = "LIVE", "badge bg-danger ms-2"
        visible_objects = [obj for obj in active_objects if obj.average_speed >= min_velocity]
        is_live = True
    else:
        # Fallback to simulation
        frame_idx, next_frame_idx = int(n) % TOTAL_FRAMES, (int(n) + 1) % TOTAL_FRAMES
        for i, drone in enumerate(simulated_drones):
            target_pos, future_pos = path_data[i][frame_idx], path_data[i][next_frame_idx]
            dx, dy, dz = (future_pos[0] - target_pos[0]) * 111000, (future_pos[1] - target_pos[1]) * 88000, (future_pos[2] - target_pos[2])
            drone.set_position(target_pos[0], target_pos[1], target_pos[2])
            drone.set_velocity(dx * 5, dy * 5, dz * 5)
        
        active_objects = simulated_drones
        visible_objects = [obj for obj in active_objects if obj.average_speed >= min_velocity]
        status_text, status_color = "● SIMULATION MODE", "#ffaa00" 
        badge_text, badge_class = "SIM", "badge bg-warning text-dark ms-2"
        is_live = False

    collision_events = collision_detector.detect_collisions(active_objects)
    
    node_data, trail_lats, trail_lons = [], [], []
    for obj in visible_objects:
        node_data.append({
            'lat': obj.x, 'lon': obj.y, 'alt': obj.altitude,
            'id': f"ID:{obj.id}", 'size': 15, 'speed': f"{obj.average_speed:.1f} m/s"
        })
        t_lats, t_lons = obj.get_trail_coordinates()
        if t_lats:
            trail_lats.extend(t_lats + [None]) 
            trail_lons.extend(t_lons + [None])
    
    df_nodes = pd.DataFrame(node_data)
    center_lat = df_nodes['lat'].mean() if is_live and not df_nodes.empty else CENTER_LAT
    center_lon = df_nodes['lon'].mean() if is_live and not df_nodes.empty else CENTER_LON
    zoom_level = 15 if is_live and not df_nodes.empty else 13

    if df_nodes.empty:
        fig = px.scatter_map(lat=[CENTER_LAT], lon=[CENTER_LON], zoom=zoom_level)
    else:
        fig = px.scatter_map(
            df_nodes, lat="lat", lon="lon", color="alt", size="size",
            size_max=15, zoom=zoom_level, hover_name="id", hover_data=["speed"],
            color_continuous_scale="Viridis", range_color=[0, 800],
            center={"lat": center_lat, "lon": center_lon}
        )

    if GRID_STATE["active"] and GRID_STATE["min"] and GRID_STATE["max"]:
        min_p, max_p = GRID_STATE["min"], GRID_STATE["max"]
        fig.add_trace(go.Scattermap(
            lat=[min_p[1], max_p[1], max_p[1], min_p[1], min_p[1]],
            lon=[min_p[0], min_p[0], max_p[0], max_p[0], min_p[0]],
            mode='lines', line=dict(width=2, color='#00FF00', dash='dot'), 
            name='Detection Grid'
        ))

    if trail_lats:
        fig.add_trace(go.Scattermap(
            lat=trail_lats, lon=trail_lons, mode='lines',
            line=dict(width=2, color='cyan'), name='Trail (5s)'
        ))

    fig.update_layout(map_style="carto-darkmatter", margin={"r":0, "t":0, "l":0, "b":0}, showlegend=False, uirevision='constant')

    table_data = [{"id": str(obj.id), "alt": f"{obj.altitude:.1f}", "spd": f"{obj.average_speed:.1f}"} for obj in visible_objects]
    status_html = html.Span(status_text, style={'color': status_color, 'fontWeight': 'bold'})
    
    alert_html = ""
    if collision_events:
        alert_html = html.Div(f"WARNING: {len(collision_events)} Potential Collisions!", style={'color': 'red', 'fontWeight': 'bold'})

    return fig, table_data, status_html, alert_html, badge_text, badge_class

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=8050)
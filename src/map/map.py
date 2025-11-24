import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import math
import sys
import os

# --- IMPORT PATH ---
# Get the current file's directory: .../VDAWS/src/map
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level: .../VDAWS/src
src_dir = os.path.dirname(current_dir)
# Go up another level: .../VDAWS (Project Root)
project_root = os.path.dirname(src_dir)

# Add Project Root to path so "import src.scanning..." works
sys.path.append(project_root)

# Local Imports
# Assuming map.py, flying_object.py, and object_manager.py are in src/map/
from object_manager import ObjectManager
# Assuming remoteid_sniffer.py is in src/scanning/
from src.scanning.remoteid_sniffer import run_sniffer_thread

# --- SETUP ---

# 1. CONFIGURATION
# *** CHANGE THIS TO YOUR WIFI ADAPTER NAME ***
# Windows: usually 'Wi-Fi' or 'Ethernet'
# Linux/Mac: usually 'wlan0', 'mon0', or 'en0'
WIFI_INTERFACE = 'Wi-Fi' 

# 2. Initialize the Manager (The "Database")
# Drones disappear from map if no signal for 10 seconds
manager = ObjectManager(timeout_seconds=10)

# 3. Start the Sniffer (The "Producer")
# This runs in the background and feeds data into 'manager'
run_sniffer_thread(manager, interface=WIFI_INTERFACE) 

# --- DASH APP ---
app = dash.Dash(__name__)

# Fixed Map Settings
CENTER_LAT = 34.05 # Default (LA) - will auto-center if drones appear
CENTER_LON = -118.25

app.layout = html.Div([
    html.H3("Live Drone Remote ID Tracker", style={'position': 'absolute', 'z-index': '10', 'left': '20px', 'top': '10px', 'color': 'white'}),
    
    dcc.Graph(id='map-graph', style={'height': '100vh'}),
    
    # Update the map every 500ms (2 times per second)
    dcc.Interval(id='interval-component', interval=500, n_intervals=0),
])

@app.callback(
    Output('map-graph', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_map(n):
    # 1. Get Real Data from Manager
    active_objects = manager.get_active_objects()
    
    # 2. Prepare Data for Plotly
    node_data = []
    arrow_data = []
    
    # Arrow visuals
    ARROW_OFFSET = 0.00025 
    LON_SCALE = 1.0 # Will calc based on lat later

    for obj in active_objects:
        d_lat = float(obj.x)
        d_lon = float(obj.y)
        d_alt = float(obj.altitude)
        vx = float(obj.velocity[0])
        vy = float(obj.velocity[1])
        
        # Node (The Drone Dot)
        node_data.append({
            'lat': d_lat,
            'lon': d_lon,
            'alt': d_alt,
            'id': f"ID: {obj.id}",
            'size': 15
        })

        # Velocity Arrow
        speed_mag = math.sqrt(vx**2 + vy**2)
        if speed_mag > 0.5: # Only draw arrow if moving > 0.5 m/s
            # Normalize vector
            norm_vx = vx / speed_mag
            norm_vy = vy / speed_mag
            
            # Adjust Longitude scale for arrow direction (simple Mercator fix)
            LON_SCALE = 1.0 / math.cos(math.radians(d_lat))
            
            head_lat = d_lat + (norm_vy * ARROW_OFFSET)
            head_lon = d_lon + (norm_vx * ARROW_OFFSET * LON_SCALE)
            
            arrow_data.append({
                'lat': head_lat,
                'lon': head_lon,
                'speed': speed_mag,
                'size': 8
            })

    df_nodes = pd.DataFrame(node_data)
    df_arrows = pd.DataFrame(arrow_data)

    # 3. Construct the Map
    
    # CASE A: No Drones Found
    if df_nodes.empty:
        fig = px.scatter_mapbox(
            lat=[CENTER_LAT], lon=[CENTER_LON], zoom=10
        )
        fig.update_layout(mapbox_style="open-street-map")
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        return fig

    # CASE B: Drones Detected
    # Create the scatter plot for drones
    fig = px.scatter_mapbox(
        df_nodes, 
        lat="lat", 
        lon="lon", 
        color="alt", 
        size="size",
        size_max=15, 
        zoom=14, 
        hover_name="id",
        color_continuous_scale="Jet", 
        range_color=[0, 120] # 0-120 meters (approx 400ft)
    )

    # Overlay Velocity Arrows
    if not df_arrows.empty:
        fig.add_trace(go.Scattermapbox(
            lat=df_arrows['lat'],
            lon=df_arrows['lon'],
            mode='markers',
            marker=go.scattermapbox.Marker(
                size=10,
                color='red', # Red tip for direction
                allowoverlap=True
            ),
            hoverinfo='skip'
        ))

    # Layout Updates
    fig.update_layout(
        mapbox_style="open-street-map",
        margin={"r":0, "t":0, "l":0, "b":0},
        showlegend=False,
        # 'uirevision' is CRITICAL. 
        # It stops the map from resetting zoom/pan every 0.5 seconds.
        uirevision='constant_loop' 
    )
    
    return fig

if __name__ == '__main__':
    # Turn off reloader because it messes with the background Sniffer thread
    app.run(debug=True, use_reloader=False)
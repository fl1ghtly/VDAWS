import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import math
import time
import uuid
import traceback
import pandas as pd  # px works best with pandas or dicts

# --- IMPORT FROM YOUR FILE ---
from flying_object import FlyingObject

# --- HELPER FUNCTIONS ---
def serialize_drone(drone_obj):
    return {
        "id": int(drone_obj.id),
        "position": [float(x) for x in drone_obj.position],
        "velocity": [float(x) for x in drone_obj.velocity],
        "lastHeartbeat": int(drone_obj.lastHeartbeat)
    }

def deserialize_drone(data_dict):
    return FlyingObject(
        id=int(data_dict["id"]),
        position=tuple(data_dict["position"]),
        velocity=tuple(data_dict["velocity"]),
        lastHeartbeat=int(data_dict["lastHeartbeat"])
    )

# --- SIMULATION SETUP ---
boundary_coords = [
    [37.70, -122.52], [37.70, -122.35], [37.83, -122.35],
    [37.83, -122.52], [37.70, -122.52]
]
lats = [p[0] for p in boundary_coords]
lons = [p[1] for p in boundary_coords]
CENTER_LAT = (min(lats) + max(lats)) / 2
CENTER_LON = (min(lons) + max(lons)) / 2

# Define coordinates for the boundary line to plot separately
boundary_df = pd.DataFrame({
    'lat': [p[0] for p in boundary_coords],
    'lon': [p[1] for p in boundary_coords]
})

TOTAL_FRAMES = 100
PATH_MOVEMENT_SCALE = 1.5 

def generate_path_coordinates(steps):
    paths = [[], [], []]
    radius = 0.03 * PATH_MOVEMENT_SCALE
    for i in range(steps):
        t = i / steps 
        angle = t * 2 * np.pi
        paths[0].append((CENTER_LAT + radius * np.sin(angle), CENTER_LON + radius * np.cos(angle), 100))
        scale = radius * 1.5
        d2_lat = CENTER_LAT + (scale * np.sin(angle) * np.cos(angle)) / (1 + np.sin(angle)**2)
        d2_lon = CENTER_LON + (scale * np.cos(angle)) / (1 + np.sin(angle)**2)
        paths[1].append((d2_lat, d2_lon, 500))
        paths[2].append((CENTER_LAT + (radius * 1.5 * np.sin(angle)), CENTER_LON - 0.04, 800))
    return paths

path_data = generate_path_coordinates(TOTAL_FRAMES)

# --- INITIALIZATION ---
current_time = int(time.time())
initial_drones_objs = [
    FlyingObject.create_with_id(uuid.uuid4().int & (1<<32)-1, CENTER_LAT, CENTER_LON, 100.0, 0.0, 0.0, 0.0, current_time),
    FlyingObject.create_with_id(uuid.uuid4().int & (1<<32)-1, CENTER_LAT, CENTER_LON, 500.0, 0.0, 0.0, 0.0, current_time),
    FlyingObject.create_with_id(uuid.uuid4().int & (1<<32)-1, CENTER_LAT, CENTER_LON, 800.0, 0.0, 0.0, 0.0, current_time)
]
initial_drone_dicts = [serialize_drone(d) for d in initial_drones_objs]

# --- DASH APP ---
app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Graph(id='map-graph', style={'height': '100vh'}),
    dcc.Interval(id='interval-component', interval=200, n_intervals=0),
    dcc.Store(id='drone-store', data=initial_drone_dicts)
])

@app.callback(
    [Output('map-graph', 'figure'),
     Output('drone-store', 'data')],
    Input('interval-component', 'n_intervals'),
    State('drone-store', 'data')
)
def update_simulation_and_map(n, current_drone_dicts):
    try:
        if not current_drone_dicts:
            current_drone_dicts = initial_drone_dicts

        active_drones = [deserialize_drone(d) for d in current_drone_dicts]
        frame_idx = int(n) % TOTAL_FRAMES
        next_frame_idx = (int(n) + 1) % TOTAL_FRAMES
        
        print(f"Processing Frame: {frame_idx}")

        # --- Physics Update ---
        for i, drone in enumerate(active_drones):
            target_pos = path_data[i][frame_idx]
            future_pos = path_data[i][next_frame_idx]
            vx = future_pos[1] - target_pos[1] 
            vy = future_pos[0] - target_pos[0]
            drone.set_velocity(vx, vy, 0.0)
            drone.set_position(target_pos[0], target_pos[1], target_pos[2])

        # --- Data Prep for Plotly Express ---
        # We use lists of dictionaries for easy DataFrame creation
        node_data = []
        arrow_data = []
        
        ARROW_START_OFFSET_DEG = 0.00025 
        LON_SCALE = 1.0 / math.cos(math.radians(CENTER_LAT))

        for drone in active_drones:
            d_lat = float(drone.x)
            d_lon = float(drone.y)
            d_alt = float(drone.altitude)
            vx = float(drone.velocity[0])
            vy = float(drone.velocity[1])
            
            # 1. Node Info
            node_data.append({
                'lat': d_lat,
                'lon': d_lon,
                'alt': d_alt,
                'id': str(drone.id),
                'size': 15
            })

            # 2. Arrow Info
            speed_mag = math.sqrt(vx**2 + vy**2) * 100000 
            
            if speed_mag > 0:
                norm_vx = vx / math.sqrt(vx**2 + vy**2)
                norm_vy = vy / math.sqrt(vx**2 + vy**2)
                head_lat = d_lat + (norm_vy * ARROW_START_OFFSET_DEG)
                head_lon = d_lon + (norm_vx * ARROW_START_OFFSET_DEG * LON_SCALE)
            else:
                head_lat = d_lat
                head_lon = d_lon
                
            arrow_data.append({
                'lat': head_lat,
                'lon': head_lon,
                'speed': float(speed_mag),
                'size': 8
            })

        # Create DataFrames
        df_nodes = pd.DataFrame(node_data)
        df_arrows = pd.DataFrame(arrow_data)

        # --- Build Figure with PX ---
        # 1. Create Base Map with Nodes
        fig = px.scatter_mapbox(
            df_nodes, 
            lat="lat", 
            lon="lon", 
            color="alt",
            size="size",
            size_max=15,
            color_continuous_scale="Jet",
            range_color=[0, 1000],
            hover_data=["id", "alt"],
            zoom=11,
            center={"lat": CENTER_LAT, "lon": CENTER_LON}
        )

        # 2. Add Arrows manually (as PX is best for single traces)
        # We use go.Scattermapbox to overlay the velocity heads
        fig.add_trace(go.Scattermapbox(
            lat=df_arrows['lat'],
            lon=df_arrows['lon'],
            mode='markers',
            marker=go.scattermapbox.Marker(
                size=10,
                color=df_arrows['speed'],
                colorscale='Viridis',
                cmin=0, cmax=600,
                showscale=False
            ),
            hoverinfo='skip' # Skip hover for arrows to reduce clutter
        ))

        # 3. Add Boundary Line
        fig.add_trace(go.Scattermapbox(
            lat=boundary_df['lat'],
            lon=boundary_df['lon'],
            mode='lines',
            line=dict(width=4, color='red'),
            hoverinfo='skip'
        ))

        # 4. Update Layout settings
        fig.update_layout(
            mapbox_style="open-street-map",
            margin={"r":0, "t":0, "l":0, "b":0},
            showlegend=False,
            # CRITICAL: Keeps the map from resetting zoom every frame
            uirevision='constant_loop' 
        )
        
        updated_drone_dicts = [serialize_drone(d) for d in active_drones]
        return fig, updated_drone_dicts

    except Exception as e:
        print("CRASH DETECTED:")
        traceback.print_exc()
        return dash.no_update, dash.no_update

if __name__ == '__main__':
    app.run(debug=True)
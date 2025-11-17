import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import numpy as np
import math
import time
import uuid

# --- IMPORT FROM YOUR FILE ---
from flying_object import FlyingObject

# --- HELPER FUNCTIONS ---
def serialize_drone(drone_obj):
    # FlyingObject now handles type safety internally, so this is safe
    return {
        "id": drone_obj.id,
        "position": drone_obj.position,
        "velocity": drone_obj.velocity,
        "lastHeartbeat": drone_obj.lastHeartbeat
    }

def deserialize_drone(data_dict):
    return FlyingObject(
        id=data_dict["id"],
        position=tuple(data_dict["position"]),
        velocity=tuple(data_dict["velocity"]),
        lastHeartbeat=data_dict["lastHeartbeat"]
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

geojson_boundary = {
    "type": "Feature",
    "geometry": {"type": "LineString", "coordinates": [[lon, lat] for lat, lon in boundary_coords]}
}

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
    print(f"Updating Frame: {n % TOTAL_FRAMES}")

    if not current_drone_dicts:
        current_drone_dicts = initial_drone_dicts

    active_drones = [deserialize_drone(d) for d in current_drone_dicts]
    frame_idx = n % TOTAL_FRAMES
    next_frame_idx = (n + 1) % TOTAL_FRAMES
    
    # Update Physics
    for i, drone in enumerate(active_drones):
        target_pos = path_data[i][frame_idx]
        future_pos = path_data[i][next_frame_idx]
        
        vx = future_pos[1] - target_pos[1] 
        vy = future_pos[0] - target_pos[0]
        
        # Now safe because FlyingObject converts them to float internally
        drone.set_velocity(vx, vy, 0.0)
        drone.set_position(target_pos[0], target_pos[1], target_pos[2])

    node_lats, node_lons = [], []
    altitudes, texts = [], []
    
    for drone in active_drones:
        node_lats.append(drone.x)
        node_lons.append(drone.y)
        altitudes.append(drone.altitude)
        texts.append(f"ID: {drone.id}<br>Lat: {drone.x:.4f}<br>Lon: {drone.y:.4f}")

    fig = go.Figure()

    # NODES
    fig.add_trace(go.Scattermapbox(
        lat=node_lats, 
        lon=node_lons, 
        mode='markers', 
        text=texts,
        marker=go.scattermapbox.Marker(
            size=20,
            color=altitudes, 
            colorscale='Jet', 
            cmin=0, cmax=1000
        ),
        hoverinfo='text'
    ))

    fig.update_layout(
        # Force Redraw using datarevision
        datarevision=n,
        mapbox=dict(
            style='open-street-map',
            center=dict(lat=CENTER_LAT, lon=CENTER_LON),
            zoom=11,
            layers=[{
                "source": geojson_boundary,
                "type": "line", 
                "color": "red",       
                "line": {"width": 5}
            }]
        ),
        margin={"r":0, "t":0, "l":0, "b":0},
        showlegend=False,
        uirevision='constant_loop' 
    )
    
    updated_drone_dicts = [serialize_drone(d) for d in active_drones]
    return fig, updated_drone_dicts

if __name__ == '__main__':
    app.run(debug=True)
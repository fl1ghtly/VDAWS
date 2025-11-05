import dash
from dash import dcc
from dash import html
import plotly.graph_objects as go
import numpy as np

class FlyingObject:
    def __init__(self, name, lat, lon, altitude):
        self.name = name
        self.lat = lat
        self.lon = lon
        self.altitude = altitude

#Data Preparation
flying_objects = [
    FlyingObject("Drone1", 37.7749, -122.4194, 100),
    FlyingObject("Drone2", 37.7849, -122.4094, 300),
    FlyingObject("Drone3", 37.7649, -122.4294, 600),
]
boundary_coords = [
    [37.70, -122.52], [37.70, -122.35], [37.83, -122.35],
    [37.83, -122.52], [37.70, -122.52]
]
altitudes = [obj.altitude for obj in flying_objects]
lats = [obj.lat for obj in flying_objects]
lons = [obj.lon for obj in flying_objects]
texts = [f"{obj.name}: {obj.altitude}m" for obj in flying_objects]
min_alt, max_alt = min(altitudes), max(altitudes)
center_lat = (boundary_coords[0][0] + boundary_coords[2][0]) / 2
center_lon = (boundary_coords[0][1] + boundary_coords[1][1]) / 2

# Convert Boundary to GeoJSON Format
boundary_geojson_coords = [[lon, lat] for lat, lon in boundary_coords]
geojson_boundary = {
    "type": "Feature",
    "geometry": {"type": "Polygon", "coordinates": [boundary_geojson_coords]}
}

#Create the Plotly Figure
fig = go.Figure()

#Add Flying Object Scattermap
fig.add_trace(go.Scattermap(
    lat=lats,
    lon=lons,
    mode='markers',
    marker=go.scattermap.Marker(
        size=12,
        color=altitudes,
        colorscale='Jet',
        cmin=min_alt,
        cmax=max_alt,
        showscale=True,
        colorbar=dict(title='Altitude (m)')
    ),
    text=texts,
    hoverinfo='text'
))

#Update Map Layout
fig.update_layout(
    title='Flying Object Tracker',
    
    
    mapbox=dict(
        
        style='open-street-map', 
        
        center=go.layout.mapbox.Center(lat=center_lat, lon=center_lon), 
        
        zoom=12,
        layers=[
            {"source": geojson_boundary, "type": "fill", "color": "blue", "opacity": 0.1},
            {"source": geojson_boundary, "type": "line", "color": "blue", "line": {"width": 2}}
        ]
    ),
    margin={"r":0, "t":40, "l":0, "b":0}
)

#Initialize the Dash App
app = dash.Dash(__name__)

#Define the App Layout
app.layout = html.Div(children=[
    html.H1(children='Dash Flying Object Map'),
    html.Div(children='A map showing flying objects, using Plotly Dash.'),
    dcc.Graph(
        id='map-graph',
        figure=fig,
        style={'height': '80vh'}
    )
])

#Run the App
if __name__ == '__main__':
    app.run(debug=True)
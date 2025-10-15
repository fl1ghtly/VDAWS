import folium
from folium import plugins
from flying_object import FlyingObject
import matplotlib.cm as cm
import matplotlib.colors as colors
import numpy as np

# Example list of flying objects
flying_objects = [
    FlyingObject("Drone1", 37.7749, -122.4194, 100),
    FlyingObject("Drone2", 37.7849, -122.4094, 300),
    FlyingObject("Drone3", 37.7649, -122.4294, 600),
]

# Predetermined boundary (San Francisco bounding box example)
boundary_coords = [
    [37.70, -122.52],
    [37.70, -122.35],
    [37.83, -122.35],
    [37.83, -122.52],
    [37.70, -122.52]
]

# Get min and max altitude for normalization
altitudes = [obj.altitude for obj in flying_objects]
min_alt, max_alt = min(altitudes), max(altitudes)

# Function to get color from a gradient (colormap)
def altitude_color_gradient(altitude):
    norm = colors.Normalize(vmin=min_alt, vmax=max_alt)
    cmap = cm.get_cmap('jet')  # You can use 'viridis', 'plasma', etc.
    rgba = cmap(norm(altitude))
    # Convert RGBA to hex
    return colors.rgb2hex(rgba)

# Center map on the boundary
center_lat = (boundary_coords[0][0] + boundary_coords[2][0]) / 2
center_lon = (boundary_coords[0][1] + boundary_coords[1][1]) / 2
m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

# Draw boundary
folium.PolyLine(boundary_coords, color='blue', weight=2, fill=True, fill_opacity=0.1).add_to(m)

# Add flying objects with gradient color
for obj in flying_objects:
    color = altitude_color_gradient(obj.altitude)
    folium.CircleMarker(
        location=[obj.lat, obj.lon],
        radius=8,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.8,
        popup=f"{obj.name}: {obj.altitude}m"
    ).add_to(m)

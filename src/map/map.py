import folium
import time
from flying_object import FlyingObject
import matplotlib.cm as cm
import matplotlib.colors as colors

# ---- CONFIGURATION ----
flying_objects = [
    FlyingObject("Drone1", 37.7749, -122.4194, 100),
    FlyingObject("Drone2", 37.7849, -122.4094, 300),
    FlyingObject("Drone3", 37.7649, -122.4294, 600)
]

boundary_coords = [
    [37.70, -122.52],
    [37.70, -122.35],
    [37.83, -122.35],
    [37.83, -122.52],
    [37.70, -122.52]
]

update_interval = 3  # seconds

# ---- FUNCTIONS ----
def altitude_color_gradient(altitude, min_alt, max_alt):
    norm = colors.Normalize(vmin=min_alt, vmax=max_alt)
    cmap = cm.get_cmap('jet')
    rgba = cmap(norm(altitude))
    return colors.rgb2hex(rgba)

def get_new_position_for(obj):
    # Replace this with a real update from sensors or API
    # Here we just "move" each drone slightly for demo purposes
    return obj.x + 0.0001, obj.y + 0.0001, obj.altitude + 10

# ---- MAIN LOOP ----
while True:
    altitudes = [obj.altitude for obj in flying_objects]
    min_alt, max_alt = min(altitudes), max(altitudes)
    center_lat = (boundary_coords[0][0] + boundary_coords[2][0]) / 2
    center_lon = (boundary_coords[0][1] + boundary_coords[1][1]) / 2

    m = folium.Map(location=[center_lat, center_lon], zoom_start=12) #plan to swap off folium for something more lighweight that does not need to regenerate the map every loop.
    folium.PolyLine(boundary_coords, color='blue', weight=2, fill=True, fill_opacity=0.1).add_to(m)

    for obj in flying_objects:
        # Update object's position
        new_x, new_y, new_altitude = get_new_position_for(obj) # to add remoteid parser to obtain new_x, new_y, new_altitude.
        obj.set_position(new_x, new_y, new_altitude)
        # Compute marker color based on altitude
        color = altitude_color_gradient(obj.altitude, min_alt, max_alt)
        folium.CircleMarker(
            location=[obj.x, obj.y],
            radius=8,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            popup=f"{obj.id}: {obj.altitude}m"
        ).add_to(m)

    # Save or display map after update
    m.save('realtime_map.html')
    print("Map updated and saved as 'realtime_map.html'.")
    time.sleep(update_interval)

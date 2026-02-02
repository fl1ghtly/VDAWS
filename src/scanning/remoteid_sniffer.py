import sys
import math
import time
import threading
import asyncio
import pyshark

# --- VELOCITY MATH HELPER ---
def calculate_velocity_vector(speed_h, direction_deg, speed_v):
    """
    Converts Speed (magnitude) and Direction (degrees) into a 3D vector (vx, vy, vz).
    Navigation: 0° is North (Positive Y), 90° is East (Positive X).
    """
    rads = math.radians(direction_deg)
    vx = speed_h * math.sin(rads) # East
    vy = speed_h * math.cos(rads) # North
    vz = speed_v                  # Vertical
    return vx, vy, vz

# --- THREAD RUNNER ---
def run_sniffer_thread(object_manager, interface='Wi-Fi'):
    """
    Starts the sniffer loop in a non-blocking daemon thread.
    """
    t = threading.Thread(target=sniff_loop, args=(object_manager, interface), daemon=True)
    t.start()
    return t

# --- MAIN LOOP ---
def sniff_loop(object_manager, interface):
    # --- CRITICAL FIX: INITIALIZE EVENT LOOP ---
    # Pyshark uses asyncio. When running in a separate thread, 
    # we must explicitly create a new event loop for that thread.
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    # -------------------------------------------

    print(f"[Sniffer] Starting background capture on interface: {interface}...")
    
    # Cache to store data for MAC addresses (since ID and Location often come in different packets)
    drone_cache = {}

    try:
        # Filter for OpendroneID. 
        capture = pyshark.LiveCapture(interface=interface, display_filter='opendroneid')

        for packet in capture.sniff_continuously():
            try:
                if not hasattr(packet, 'opendroneid'):
                    continue

                layer = packet.opendroneid
                
                # Get the source MAC address to link messages together
                source_mac = getattr(packet, 'wlan', None).sa if hasattr(packet, 'wlan') else "unknown_mac"
                
                if source_mac not in drone_cache:
                    drone_cache[source_mac] = {'id': None, 'last_update': 0}

                # --- MESSAGE TYPE 1: BASIC ID (contains Serial Number) ---
                if hasattr(layer, 'basic_id_id'):
                    raw_id = layer.basic_id_id
                    drone_cache[source_mac]['id'] = str(raw_id)

                # --- MESSAGE TYPE 2: LOCATION (contains Lat/Lon/Alt/Speed) ---
                if hasattr(layer, 'location_latitude'):
                    lat = float(layer.location_latitude)
                    lon = float(layer.location_longitude)
                    alt = float(layer.location_geodetic_altitude)
                    
                    speed_h = float(getattr(layer, 'location_speed_horizontal', 0.0))
                    direction = float(getattr(layer, 'location_direction', 0.0))
                    speed_v = float(getattr(layer, 'location_speed_vertical', 0.0))
                    
                    vx, vy, vz = calculate_velocity_vector(speed_h, direction, speed_v)
                    
                    # Update cache
                    drone_cache[source_mac].update({
                        'lat': lat, 'lon': lon, 'alt': alt,
                        'vx': vx, 'vy': vy, 'vz': vz,
                        'last_update': time.time()
                    })

                # --- UPDATE MANAGER ---
                # Only push to object manager if we have BOTH an ID and a recent Location
                cache_entry = drone_cache[source_mac]
                
                if cache_entry['id'] and 'lat' in cache_entry:
                    # Send to main tracking system
                    object_manager.update_object(
                        id=cache_entry['id'],
                        lat=cache_entry['lat'],
                        lon=cache_entry['lon'],
                        alt=cache_entry['alt'],
                        vx=cache_entry.get('vx', 0.0),
                        vy=cache_entry.get('vy', 0.0),
                        vz=cache_entry.get('vz', 0.0)
                    )

            except Exception:
                # Ignore partial packet errors
                continue

    except Exception as e:
        print(f"[Sniffer] CRITICAL ERROR: {e}")
        print(f"Ensure '{interface}' is correct. (Windows: 'Wi-Fi', Linux: 'wlan0' or 'mon0')")
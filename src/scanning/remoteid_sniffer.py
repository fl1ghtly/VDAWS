import sys
import math
import time
import threading
import pyshark 

# --- VELOCITY MATH HELPER ---
def calculate_velocity_vector(speed_h, direction_deg, speed_v):
    """
    Converts Speed (magnitude) and Direction (degrees) into a 3D vector (vx, vy, vz).
    Assumes Direction 0° is North, 90° is East.
    """
    # Convert degrees to radians
    # Standard math: 0 is East. Navigation: 0 is North.
    # We treat 0 deg as North (positive Y), 90 deg as East (positive X).
    rads = math.radians(direction_deg)
    
    vx = speed_h * math.sin(rads) # East component
    vy = speed_h * math.cos(rads) # North component
    vz = speed_v                  # Vertical component
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
    print(f"[Sniffer] Starting background capture on interface: {interface}...")
    
    try:
        # 'display_filter' ensures we only process Drone ID packets (saving CPU)
        capture = pyshark.LiveCapture(interface=interface, display_filter='opendroneid')

        for packet in capture.sniff_continuously():
            try:
                # Check if the layer exists
                if not hasattr(packet, 'opendroneid'):
                    continue

                layer = packet.opendroneid

                # 1. Extract ID (Required)
                # 'basic_id_id' is the field name for the Serial Number / ID
                raw_id = getattr(layer, 'basic_id_id', None)
                
                # If we didn't get a Basic ID, we can't track it. Skip.
                if not raw_id:
                    continue 

                # 2. Extract Position
                # Use getattr with defaults to prevent crashing on partial packets
                lat = float(getattr(layer, 'location_latitude', 0.0))
                lon = float(getattr(layer, 'location_longitude', 0.0))
                alt = float(getattr(layer, 'location_geodetic_altitude', 0.0))

                # 3. Extract Velocity Data
                speed_h = float(getattr(layer, 'location_speed_horizontal', 0.0))
                direction = float(getattr(layer, 'location_direction', 0.0))
                speed_v = float(getattr(layer, 'location_speed_vertical', 0.0))

                # Calculate Vector Components
                vx, vy, vz = calculate_velocity_vector(speed_h, direction, speed_v)

                # 4. Update the Manager
                # This sends the data safely to the map's storage
                object_manager.update_object(
                    id=str(raw_id),
                    lat=lat, lon=lon, alt=alt,
                    vx=vx, vy=vy, vz=vz
                )
                
            except Exception as parse_err:
                # UDP packets can sometimes be malformed or partial, ignore single packet errors
                # print(f"[Sniffer] Parse Error: {parse_err}")
                continue

    except Exception as e:
        print(f"[Sniffer] CRITICAL ERROR: {e}")
        print(f"Ensure '{interface}' is in Monitor Mode if on Linux, or supported on Windows.")
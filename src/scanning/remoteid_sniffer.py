import sys
import pyshark 

def main():
    print("Starting Pyshark sniffer...")
    print("This requires 'tshark' (from Wireshark) to be installed and in your PATH.")
    print("Sniffing live traffic indefinitely. Press Ctrl+C to stop.")

    # --- IMPORTANT ---
    # Find your interface name.
    # Linux: 'eth0', 'wlan0' (check with 'ifconfig')
    # macOS: 'en0' (check with 'ifconfig')
    # Windows: 'Wi-Fi', 'Ethernet' (check with 'tshark -D')
    INTERFACE = 'Wi-Fi'  # <-- *** CHANGE THIS TO YOUR INTERFACE ***

    try:
        # 'pyshark.LiveCapture' is the main entry point
        # It's a wrapper for tshark, which comes with Wireshark
        capture = pyshark.LiveCapture(interface=INTERFACE)

        print(f"\nListening on interface: {INTERFACE}...")
        
        # 'sniff_continuously' is a generator that yields packets as they arrive.
        # This is the pyshark equivalent of scapy's sniff(count=0)
        for packet in capture.sniff_continuously():
            # str(packet) provides a good one-line summary
            print(str(packet))

    except FileNotFoundError:
        print("\n[Error] 'tshark' not found.")
        print("Pyshark requires tshark to be installed.")
        print("Please install Wireshark (which includes tshark) and ensure it's in your system's PATH.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nSniffing stopped by user.")
    except Exception as e:
        # This will catch permission errors (run as sudo)
        # or if the interface (INTERFACE) is wrong.
        print(f"\n[Error] An error occurred.")
        print(f"Details: {e}")
        print(f"\nCommon Fixes:")
        print(f"1. Did you run this script with sudo/Administrator privileges?")
        print(f"2. Is the INTERFACE variable '{INTERFACE}' correct for your system?")
        print("   (Try running 'tshark -D' in your terminal to list interfaces)")
        sys.exit(1)
    
    print("\nSniffing stopped.")

if __name__ == "__main__":
    main()
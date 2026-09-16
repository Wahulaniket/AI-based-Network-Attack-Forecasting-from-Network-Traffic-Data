import os
import time
import socket
import threading
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
import psutil

import scapy.all as scapy
from scapy.arch.windows import get_windows_if_list

class NpcapCapturer:
    """Native Windows Npcap packet capture engine using Scapy AsyncSniffer.
    
    Provides:
    - Dynamic active interface discovery (default route / IPv4)
    - Manual interface override support (LIVE_INTERFACE env or explicit API argument)
    - Network change monitoring & clean capture restarts
    - Real-time packet, byte, flow, and diagnostic metrics
    """
    def __init__(self, packet_callback: Optional[Callable[[Any], None]] = None, on_network_changed: Optional[Callable[[], None]] = None):
        self.packet_callback = packet_callback
        self.on_network_changed = on_network_changed
        
        # Capture metrics
        self.capture_active = False
        self.packets_captured = 0
        self.bytes_captured = 0
        self.flows_created = 0
        self.last_packet_time: Optional[datetime] = None
        self.start_time: Optional[float] = None
        self.error: Optional[str] = None
        
        # Interface binding state
        self.selected_interface: str = "AUTO"
        self.interface_name: str = ""
        self.interface_description: str = ""
        self.interface_index: int = 0
        self.current_ipv4: str = ""
        self.scapy_iface: Any = None
        
        # Async Sniffer & Threads
        self.sniffer: Optional[scapy.AsyncSniffer] = None
        self.lock = threading.Lock()
        self.monitor_thread: Optional[threading.Thread] = None
        self.stop_monitor_event = threading.Event()

    def discover_active_interface(self, target_iface: str = "AUTO") -> Dict[str, Any]:
        """Discovers active Windows network interface and associated IPv4.
        
        Respects `target_iface` override ('AUTO' or explicit interface name/index).
        Avoids selecting VMnet1/VMnet8 merely because their status is Up.
        """
        discovered_ip = "127.0.0.1"
        scapy_iface_obj = None
        iface_name = "Unknown"
        iface_desc = "Unknown Adapter"
        iface_index = 0

        # Step 1: Detect active default route IPv4
        try:
            route_info = scapy.conf.route.route("8.8.8.8")
            npf_dev, default_ip, gw_ip = route_info
            if default_ip and default_ip != "0.0.0.0" and default_ip != "127.0.0.1":
                discovered_ip = default_ip
        except Exception:
            # Fallback via UDP socket connect
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80))
                discovered_ip = s.getsockname()[0]
                s.close()
            except Exception:
                discovered_ip = "127.0.0.1"

        # Step 2: Resolve target interface against scapy.conf.ifaces
        if target_iface and target_iface.upper() != "AUTO":
            # Manual interface override specified
            target_str = str(target_iface).strip()
            
            # Check by index
            if target_str.isdigit():
                idx = int(target_str)
                for k, obj in scapy.conf.ifaces.items():
                    if getattr(obj, 'index', None) == idx:
                        scapy_iface_obj = obj
                        break

            # Check by name or pcap_name or description
            if not scapy_iface_obj:
                for k, obj in scapy.conf.ifaces.items():
                    name = str(getattr(obj, 'name', ''))
                    pcap_name = str(getattr(obj, 'pcap_name', ''))
                    description = str(getattr(obj, 'description', ''))
                    if target_str.lower() in name.lower() or target_str.lower() in pcap_name.lower() or target_str.lower() in description.lower():
                        scapy_iface_obj = obj
                        break
                        
        if not scapy_iface_obj:
            # AUTO mode: find interface matching discovered default IP
            for k, obj in scapy.conf.ifaces.items():
                obj_ip = getattr(obj, 'ip', None)
                if obj_ip == discovered_ip:
                    scapy_iface_obj = obj
                    break

        if not scapy_iface_obj:
            # Fallback: pick default route NPF device
            try:
                route_info = scapy.conf.route.route("8.8.8.8")
                scapy_iface_obj = route_info[0]
            except Exception:
                scapy_iface_obj = scapy.conf.iface

        # Extract details from resolved scapy_iface_obj
        if hasattr(scapy_iface_obj, 'name'):
            iface_name = getattr(scapy_iface_obj, 'name', 'Unknown')
            iface_desc = getattr(scapy_iface_obj, 'description', iface_name)
            iface_index = getattr(scapy_iface_obj, 'index', 0)
            if hasattr(scapy_iface_obj, 'ip') and getattr(scapy_iface_obj, 'ip'):
                discovered_ip = getattr(scapy_iface_obj, 'ip')
        else:
            iface_name = str(scapy_iface_obj)
            iface_desc = str(scapy_iface_obj)

        return {
            "scapy_iface": scapy_iface_obj,
            "interface_name": iface_name,
            "interface_description": iface_desc,
            "interface_index": iface_index,
            "current_ipv4": discovered_ip
        }

    def start(self, interface_override: Optional[str] = None):
        """Starts live packet capture using Scapy AsyncSniffer."""
        with self.lock:
            if self.capture_active:
                return

            env_iface = os.getenv("CYBERCAST_LIVE_INTERFACE", "AUTO")
            target_iface = interface_override or env_iface
            self.selected_interface = target_iface

            discovery = self.discover_active_interface(target_iface)
            self.scapy_iface = discovery["scapy_iface"]
            self.interface_name = discovery["interface_name"]
            self.interface_description = discovery["interface_description"]
            self.interface_index = discovery["interface_index"]
            self.current_ipv4 = discovery["current_ipv4"]

            self.packets_captured = 0
            self.bytes_captured = 0
            self.flows_created = 0
            self.last_packet_time = None
            self.start_time = time.time()
            self.error = None
            self.capture_active = True

            try:
                self.sniffer = scapy.AsyncSniffer(
                    iface=self.scapy_iface,
                    prn=self._on_packet,
                    store=False
                )
                self.sniffer.start()
                print(f"[NpcapCapturer] Sniffer started on {self.interface_name} ({self.current_ipv4}) via {self.scapy_iface}")
            except Exception as e:
                self.capture_active = False
                self.error = f"Failed to start Npcap sniffer: {str(e)}"
                print(f"[NpcapCapturer ERROR] {self.error}")
                raise RuntimeError(self.error)

            # Start network change monitor thread
            self.stop_monitor_event.clear()
            self.monitor_thread = threading.Thread(target=self._monitor_network_loop, daemon=True)
            self.monitor_thread.start()

    def stop(self):
        """Stops live packet capture cleanly."""
        with self.lock:
            self.capture_active = False
            self.stop_monitor_event.set()
            if self.sniffer:
                try:
                    self.sniffer.stop()
                except Exception as e:
                    print(f"[NpcapCapturer Warning] Stopping sniffer: {e}")
                self.sniffer = None
            print("[NpcapCapturer] Packet capture stopped.")

    def _on_packet(self, pkt):
        """Callback invoked for each captured packet."""
        if not self.capture_active:
            return

        with self.lock:
            self.packets_captured += 1
            self.bytes_captured += len(pkt)
            self.last_packet_time = datetime.now()

        if self.packet_callback:
            try:
                self.packet_callback(pkt)
            except Exception as e:
                print(f"[NpcapCapturer Callback Error] {e}")

    def _monitor_network_loop(self):
        """Periodically checks for network interface / IPv4 changes."""
        while not self.stop_monitor_event.is_set():
            time.sleep(3)
            if not self.capture_active:
                continue

            try:
                discovery = self.discover_active_interface(self.selected_interface)
                new_ip = discovery["current_ipv4"]
                new_iface_name = discovery["interface_name"]

                if (new_ip != self.current_ipv4 or new_iface_name != self.interface_name) and new_ip != "127.0.0.1":
                    print(f"[NpcapCapturer] NETWORK CHANGE DETECTED: {self.interface_name}({self.current_ipv4}) -> {new_iface_name}({new_ip})")
                    if self.on_network_changed:
                        self.on_network_changed()
                    
                    # Restart capture on new network
                    self.stop()
                    self.start(self.selected_interface)
                    break
            except Exception as e:
                print(f"[NpcapCapturer Monitor Warning] {e}")

    def get_capture_status(self) -> Dict[str, Any]:
        return self.get_status_dict()


    def get_status_dict(self) -> Dict[str, Any]:
        """Returns diagnostic dictionary for GET /api/live/capture-status."""
        with self.lock:
            elapsed = time.time() - self.start_time if (self.start_time and self.capture_active) else 0.0
            pps = (self.packets_captured / elapsed) if elapsed > 0 else 0.0
            
            # Check if sniffer thread is actually alive
            is_alive = bool(self.capture_active and self.sniffer and self.sniffer.running)

            return {
                "capture_backend": "npcap",
                "interface": self.interface_name,
                "interface_description": self.interface_description,
                "interface_index": self.interface_index,
                "current_ipv4": self.current_ipv4,
                "capture_active": is_alive,
                "packets_captured": self.packets_captured,
                "bytes_captured": self.bytes_captured,
                "flows_created": self.flows_created,
                "packets_per_sec": round(pps, 2),
                "last_packet_time": self.last_packet_time.isoformat() if self.last_packet_time else None,
                "error": self.error
            }

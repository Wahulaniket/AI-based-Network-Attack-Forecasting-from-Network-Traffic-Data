import scapy.all as scapy
from scapy.arch.windows import get_windows_if_list

print("=== WINDOWS IF LIST ===")
for iface in get_windows_if_list():
    print(f"Name: {iface.get('name')}")
    print(f"  Description: {iface.get('description')}")
    print(f"  GUID: {iface.get('guid')}")
    print(f"  IPs: {iface.get('ips')}")
    print(f"  Network Name: {iface.get('network_name')}")
    print(f"  NPF Name: {iface.get('win_index')}")
    print()

print("=== SCAPY CONF.IFACES ===")
print(scapy.conf.ifaces)

print("\n=== DEFAULT ROUTE ===")
try:
    print("Default route iface:", scapy.conf.route.route("8.8.8.8"))
except Exception as e:
    print("Route error:", e)

import scapy.all as scapy
import time

print("Finding active interface...")
# Find default route interface
route_info = scapy.conf.route.route("8.8.8.8")
dev_npf = route_info[0]
my_ip = route_info[1]
print("NPF Dev:", dev_npf)
print("My IP:", my_ip)

# Find matching NetworkInterface in scapy.conf.ifaces
selected_iface = None
for iface_key, iface_obj in scapy.conf.ifaces.items():
    if getattr(iface_obj, 'ip', None) == my_ip or getattr(iface_obj, 'pcap_name', None) == dev_npf:
        selected_iface = iface_obj
        break

if not selected_iface:
    selected_iface = dev_npf

print("Selected Scapy Interface:", selected_iface)

packet_count = 0
def packet_callback(pkt):
    global packet_count
    packet_count += 1
    if packet_count <= 5:
        print(f"Captured packet #{packet_count}: {pkt.summary()}")

print("Starting 5-second capture...")
t = scapy.AsyncSniffer(iface=selected_iface, prn=packet_callback, store=False)
t.start()
time.sleep(5)
t.stop()

print(f"Total packets captured in 5s: {packet_count}")

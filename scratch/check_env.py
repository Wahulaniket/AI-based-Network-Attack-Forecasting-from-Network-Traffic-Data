import sys
import os
import scapy
from scapy.all import get_if_list, conf

print("Python version:", sys.version)
print("Executable:", sys.executable)
print("Scapy version:", scapy.__version__)
print("Interfaces:", get_if_list())

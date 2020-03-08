#!/usr/bin/python3.6
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host = "time-c.nist.gov"
port = 13
s.connect((host, port))
data = s.recv(1024)
print(data)
s.close()
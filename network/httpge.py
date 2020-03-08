#!/usr/bin/python3.6
import socket
import sys

if len(sys.argv) != 2 :
    print("Usage: ", sys.argv[0], "<host>")
    exit(1)

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

host = sys.argv[1]
request = "GET / HTTP/1.1\r\n"  \
    "Host: " + host + "\r\n"    \
    "Connection: close\r\n\r\n"

port = 13
s.connect((host, port))
data = s.recv(1024)
print(data)
s.close()
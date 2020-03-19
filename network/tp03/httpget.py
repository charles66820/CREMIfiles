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

port = 80
try :
    s.connect((host, port))

    s.sendall(request.encode("utf-8"))

    while True :
        data = s.recv(1024)
        if data.decode("utf-8") == "" : break
        print (data.decode("utf-8"), end ="")

except Exception as e :
    print("Erreur avec la connexion : ", e)

s.close()
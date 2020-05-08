#!/usr/bin/python3
import socket
import ssl

HOST="127.0.0.1"
PORT=7777
BUFSIZE= 1024

# init ssl support
context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
context.load_verify_locations('client/ca.crt')
conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sslconn = context.wrap_socket(conn, server_hostname=HOST, server_side=False)
sslconn.connect((HOST, PORT))
request = b"Hello World!"
sslconn.sendall(request)
answer = sslconn.recv(BUFSIZE)
print(answer.decode())
sslconn.close()
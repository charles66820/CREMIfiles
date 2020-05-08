#!/usr/bin/python3
import socket
import ssl

HOST= '127.0.0.1'
PORT= 7777
BUFSIZE= 1024

# echo server
def echo(conn):
    while True:
        data = conn.recv(BUFSIZE)
        if data == b'' or data == b'\n' : break
        print(data.decode())
        conn.sendall(data)

# main program
srvsocket = socket.socket()
srvsocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srvsocket.bind((HOST, PORT))
srvsocket.listen()

# init ssl support
context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
context.load_cert_chain('serveur/server.crt', 'serveur/server.key')

srvsockets = context.wrap_socket(srvsocket, server_side=True)

while True:
    conn, fromaddr = srvsockets.accept()
    echo(conn)
    conn.close()

srvsockets.close()
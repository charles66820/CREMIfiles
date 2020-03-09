#!/usr/bin/python3.6
import os
import socket
import sys
import threading

def handle (sclient):
    while True :
        data = sclient.recv(1500)
        if data == b"" : break
        sclient.send(data)

    print ("Requet from : ", a[0])

    sclient.close()

try:
    host = ""
    port = 7777

    s = socket.socket(socket.AF_INET6, socket.SOCK_STREAM, 0)
    try :
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
        s.listen(1)

        try :
            while True :
                # Récupère le socket de la connexion du client et sont adresse
                sclient, a = s.accept()
                t = threading.Thread(None, handle, None, (sclient,))
                t.start()

        except Exception as e :
            print("Error with accept : ", e)

        s.close()
    except Exception as e :
        print("Error on bind the socket : ", e)

except KeyboardInterrupt:
    print('Interrupted')
    s.close()
    try:
        sys.exit(0)
    except SystemExit:
        os._exit(0)
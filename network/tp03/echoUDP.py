#!/usr/bin/python3.6
import os
import socket
import sys

try:
    host = ""
    port = 7777

    s = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM, 0)
    try :
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))

        try :
            while True :
                data, a = s.recvfrom(1500)
                if data == b"" : break
                s.sendto(data, a)

            print ("Requet from : ", a[0])

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
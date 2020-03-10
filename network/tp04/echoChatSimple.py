#!/usr/bin/python3.6
import os
import select
import socket
import sys

try:
    host = ""
    port = 7777

    s = socket.socket(socket.AF_INET6, socket.SOCK_STREAM, 0)
    try :

        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
        s.listen(1)

        try :
            l = []
            while True :
                (rl, wl, xl) = select.select(l+[s], [], [])
                for soc in rl :
                    if soc == s :
                        # Récupère le socket de la connexion du client et sont adresse
                        sclient, a = s.accept()
                        l.append(sclient)
                        print(a[0], "is connect!")
                    else :
                        data = soc.recv(1500)
                        if data == b"" :
                            soc.close()
                            l.remove(soc)
                        else :
                            # Envois le message a tous le monde sauf au serveur et au l'emeteur
                            for sc in l :
                                if soc != sc and s != sc:
                                    sc.sendall(data)

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
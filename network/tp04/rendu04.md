# TP 04

## 1 Gestion de multiples clients avec un serveur TCP

### 1.1 Version thread

- Le code :

```py
#!/usr/bin/python3.6
import os
import socket
import sys
import threading

def handle (sclient) :
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
```

- Pendant l'exécution, j'effectue la commande `netstat -tuap | grep 7777` :

```txt
tcp   0  0 localhost:47282  localhost:7777   ESTABLISHED 14275/nc
tcp   0  0 localhost:47292  localhost:7777   ESTABLISHED 14372/nc
tcp   0  0 localhost:47286  localhost:7777   ESTABLISHED 14323/nc
tcp6  0  0 [::]:7777        [::]:*           LISTEN      14207/python3.6
tcp6  0  0 localhost:7777   localhost:47282  ESTABLISHED 14207/python3.6
tcp6  0  0 localhost:7777   localhost:47286  ESTABLISHED 14207/python3.6
tcp6  0  0 localhost:7777   localhost:47292  ESTABLISHED 14207/python3.6
```

### 1.2 Version select

- Le code :

```py
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

        l = []

        try :
            while True :
                (rl, wl, xl) = select.select(l+[s], [], [])
                for soc in rl :
                    if soc == s :
                        # Récupère le socket de la connexion du client et sont adresse
                        sclient, a = s.accept()
                        l.append(sclient)
                        print(a[0], "is connect!")
                    else:
                        data = soc.recv(1500)
                        if data == b"" :
                            soc.close()
                            l.remove(soc)
                        else:
                            sclient.sendall(data)

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
```

- Pendant l'exécution, j'effectue la commande `netstat -tuap | grep 7777` :

```txt
tcp   0  0 localhost:47646  localhost:7777   ESTABLISHED 17983/nc
tcp   0  0 localhost:47644  localhost:7777   ESTABLISHED 17962/nc
tcp   0  0 localhost:47648  localhost:7777   ESTABLISHED 17993/nc
tcp6  0  0 [::]:7777        [::]:*           LISTEN      17946/python3.6
tcp6  0  0 localhost:7777   localhost:47646  ESTABLISHED 17946/python3.6
tcp6  0  0 localhost:7777   localhost:47648  ESTABLISHED 17946/python3.6
tcp6  0  0 localhost:7777   localhost:47644  ESTABLISHED 17946/python3.6
```

## 2 Serveur de chat

### 2.1 Une première version simple

```py
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
```

### 2.2 A propos d’une version thread

- Si on utilise la version thread il peut se passer que 2 messages soit traiter en meme temps sur 2 thread differant. IL peut y avoire aussi des elements qui ne sont pas a jour dans la liste des client connecter.

### 2.3 Extensions du serveur de chat

- MSG fait !
- Invalid command fait !
- Notif fait !
- NICK fait !
- Modifs fait !
- WHO fait !
- QUIT fait !
- KILL fait !

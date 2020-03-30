# TP 05

## 1 Protocole HTTP

- Le port du protocol `HTTP` est le port `80` (notez que pour le https le port est le 443).

### 1.1 Méthode `HEAD`

- Pour chaque domains vous troverez si dessous sur quelle machine est implanté le serveur, quel est le type de ce serveur et quelle est la classe de réponse :

  - Pour www.emi.u-bordeaux.fr :
    - Avec la commande `telnet www.emi.u-bordeaux.fr 80` et la requéte suivant :

        ```http
        HEAD / HTTP/1.0
        User-Agent: telnet
        Host: www.emi.u-bordeaux.fr


        ```

        ```http
        HTTP/1.1 200 OK
        Date: Mon, 30 Mar 2020 13:50:36 GMT
        Server: Apache/2.2.22 (Debian)
        Last-Modified: Tue, 18 Jun 2019 14:49:52 GMT
        ETag: "bf8b8-2467-58b9a39b56818"
        Accept-Ranges: bytes
        Content-Length: 9319
        Vary: Accept-Encoding
        Connection: close
        Content-Type: text/html

        Connection closed by foreign host.
        ```

    - Machine où est implanté le serveur : `Linux` avec la distrbution `Debian`.
    - Type du serveur : `Apache 2.2.22`.
    - Classe de réponse : `2xx successful` car le code http est 200.
  - Pour www.labri.fr :
    - Avec la commande `telnet www.labri.fr 80` et la requéte suivant :

        ```http
        HEAD / HTTP/1.0
        User-Agent: telnet
        Host: www.labri.fr


        ```

        ```http
        HTTP/1.1 301 Moved Permanently
        Date: Mon, 30 Mar 2020 13:56:20 GMT
        Server: Apache
        Location: https:///error/HTTP_BAD_REQUEST.html.var
        Content-Length: 248
        Connection: close
        Content-Type: text/html; charset=iso-8859-1

        Connection closed by foreign host.
        ```

    - Machine où est implanté le serveur : `Linux`.
    - Type du serveur : `Apache`.
    - Classe de réponse : `3xx redirection` car le code http est 301.
  - Pour www.archlinux.org :
    - Avec la commande `telnet www.archlinux.org 80` et la requéte suivant :

        ```http
        HEAD / HTTP/1.0
        User-Agent: telnet
        Host: www.archlinux.org


        ```

        ```http
        HTTP/1.1 301 Moved Permanently
        Server: nginx/1.16.1
        Date: Mon, 30 Mar 2020 14:12:19 GMT
        Content-Type: text/html
        Content-Length: 169
        Connection: close
        Location: https://www.archlinux.org/

        Connection closed by foreign host.
        ```

    - Machine où est implanté le serveur : `Linux` certainement la distribution `Archlinux`.
    - Type du serveur : `nginx 1.16.1`.
    - Classe de réponse : `3xx redirection` car le code http est 301.
  - Pour www.perdu.com :
    - Avec la commande `telnet www.perdu.com 80` et la requéte suivant :

        ```http
        HEAD / HTTP/1.0
        User-Agent: telnet
        Host: www.perdu.com


        ```

        ```http
        HTTP/1.1 200 OK
        Date: Mon, 30 Mar 2020 14:13:13 GMT
        Server: Apache
        Upgrade: h2
        Connection: Upgrade, close
        Last-Modified: Thu, 02 Jun 2016 06:01:08 GMT
        ETag: "cc-5344555136fe9"
        Accept-Ranges: bytes
        Content-Length: 204
        Cache-Control: max-age=600
        Expires: Mon, 30 Mar 2020 14:23:13 GMT
        Vary: Accept-Encoding,User-Agent
        Content-Type: text/html

        Connection closed by foreign host.
        ```

    - Machine où est implanté le serveur : `Linux`.
    - Type du serveur : `Apache`.
    - Classe de réponse : `2xx successful` car le code http est 200.

### 1.2 Classes de réponse

- Succès :
  - Avec la commande `telnet magicorp.fr 80` et la requéte si dessous j'obtien la classe 2xx :

    ```http
    HEAD / HTTP/1.0
    User-Agent: telnet
    Host: www.magicorp.fr


    ```

    ```http
    HTTP/1.1 200 OK
    X-Powered-By: Express
    Content-Type: text/html; charset=utf-8
    Content-Length: 5794
    ETag: W/"16a2-N5Oz8uDz1wPbVVbe+/Rgn3yCBVQ"
    set-cookie: connect.sid=s%3Al_XQq3Wa4yFwxM_l6kfyo4gSAgbfxbjx.o3k4wEh3JxfJNXUO5YNs9H5ul3Ler3Gn3yZWJ4My2o8; Domain=.magicorp.fr; Path=/; HttpOnly
    Date: Mon, 30 Mar 2020 14:50:31 GMT
    Connection: close

    Connection closed by foreign host.
    ```

- Erreur client :
  - Avec la commande `telnet magicorp.fr 80` et la requéte si dessous j'obtien la classe 4xx :

    ```http
    HEAD /toto HTTP/1.0
    User-Agent: telnet
    Host: www.magicorp.fr


    ```

    ```http
    HTTP/1.1 404 Not Found
    X-Powered-By: Express
    Content-Type: text/html; charset=utf-8
    Content-Length: 10
    ETag: W/"a-/8nFET8AoFHgR39QmkbW0gX342M"
    set-cookie: connect.sid=s%3A0EDklZbF30HAHBRW9tDM4MXA3kso_ewL.13yaI3Drqy26dNrt%2F4S1O7r6KOOLnMwQaIeZVJ%2Bwvs4; Domain=.magicorp.fr; Path=/; HttpOnly
    Date: Mon, 30 Mar 2020 14:56:07 GMT
    Connection: close

    Connection closed by foreign host.
    ```

- Inchangé :
  - Avec la commande `telnet www.perdu.com 80` et la requéte si dessous j'obtien la classe 3xx :

    ```http
    HEAD / HTTP/1.0
    User-Agent: telnet
    Host: www.perdu.com
    If-Modified-Since: Thu, 02 Jun 2016 06:01:08 GMT


    ```

    ```http
    HTTP/1.1 304 Not Modified
    Date: Mon, 30 Mar 2020 15:10:26 GMT
    Server: Apache
    Connection: Upgrade, close
    ETag: "cc-5344555136fe9"
    Expires: Mon, 30 Mar 2020 15:20:26 GMT
    Cache-Control: max-age=600
    Vary: User-Agent,Accept-Encoding

    Connection closed by foreign host.
    ```

- Redirection :
  - Avec la commande `telnet magicorp.fr 80` et la requéte si dessous j'obtien la classe 3xx :

    ```http
    HEAD / HTTP/1.0
    User-Agent: telnet
    Host: magicorp.fr


    ```

    ```http
    HTTP/1.1 307 Temporary Redirect
    X-Powered-By: Express
    Location: http://www.magicorp.fr/
    Vary: Accept
    Content-Type: text/plain; charset=utf-8
    Content-Length: 58
    set-cookie: connect.sid=s%3At2x6B5QMQEj3H5PaFFY0KJbkICXNI_d5.OIeAwZwM3JPvXMxYtLCCRheTPqvPcfi8qt6LCVic7SU; Domain=.magicorp.fr; Path=/; HttpOnly
    Date: Mon, 30 Mar 2020 15:04:40 GMT
    Connection: close

    Connection closed by foreign host.
    ```

    ça nous redirige car la réponse a pour http code "301 Moved Permanently".

### 1.3 Méthode `GET` simple et entêtes

Avec la commande `telnet bruno.pinaud.emi.u-bordeaux.fr 80` et la requéte suivant :

```http
GET /test-redir HTTP/1.0
User-Agent: telnet
Host: bruno.pinaud.emi.u-bordeaux.fr


```

```http
HTTP/1.1 301 Moved Permanently
Date: Mon, 30 Mar 2020 15:34:44 GMT
Server: Apache
Location: http://www.u-bordeaux.fr
Content-Length: 232
Connection: close
Content-Type: text/html; charset=iso-8859-1

<!DOCTYPE HTML PUBLIC "-//IETF//DTD HTML 2.0//EN">
<html><head>
<title>301 Moved Permanently</title>
</head><body>
<h1>Moved Permanently</h1>
<p>The document has moved <a href="http://www.u-bordeaux.fr">here</a>.</p>
</body></html>
Connection closed by foreign host.
```

Avec la commande `wget --server-response http://bruno.pinaud.emi.u-bordeaux.fr/test-redir` :

```http
--2020-03-30 17:32:32--  http://bruno.pinaud.emi.u-bordeaux.fr/test-redir
Resolving bruno.pinaud.emi.u-bordeaux.fr (bruno.pinaud.emi.u-bordeaux.fr)... 147.210.12.220, 2001:660:6101:800:252::9
Connecting to bruno.pinaud.emi.u-bordeaux.fr (bruno.pinaud.emi.u-bordeaux.fr)|147.210.12.220|:80... connected.
HTTP request sent, awaiting response...
  HTTP/1.1 301 Moved Permanently
  Date: Mon, 30 Mar 2020 15:32:32 GMT
  Server: Apache
  Location: http://www.u-bordeaux.fr
  Content-Length: 232
  Keep-Alive: timeout=5, max=100
  Connection: Keep-Alive
  Content-Type: text/html; charset=iso-8859-1
Location: http://www.u-bordeaux.fr [following]
--2020-03-30 17:32:32--  http://www.u-bordeaux.fr/
Resolving www.u-bordeaux.fr (www.u-bordeaux.fr)... 147.210.215.26
Connecting to www.u-bordeaux.fr (www.u-bordeaux.fr)|147.210.215.26|:80... connected.
HTTP request sent, awaiting response...
  HTTP/1.0 302 Found
  Location: https://www.u-bordeaux.fr/
  Content-Type: text/html
  Content-Length: 170
Location: https://www.u-bordeaux.fr/ [following]
--2020-03-30 17:32:32--  https://www.u-bordeaux.fr/
Connecting to www.u-bordeaux.fr (www.u-bordeaux.fr)|147.210.215.26|:443... connected.
HTTP request sent, awaiting response...
  HTTP/1.1 200 OK
  Server: Apache/2.2.22 (Debian)
  X-Powered-By: PHP/5.4.35-0+deb7u2
  Cache-Control: public, s-maxage=60, max-age=40
  vary: X-User-Hash,Accept-Encoding
  x-location-id: 527
  x-content-digest: ezlocation/527/ena4df4c31ff1a754225f5d10fa56a647d4f2da2df
  Expires: Mon, 30 Mar 2020 15:08:51 GMT
  Content-Language: fr
  Content-Type: text/html; charset=UTF-8
  Transfer-Encoding: chunked
  Date: Mon, 30 Mar 2020 15:32:33 GMT
  X-Varnish: 1721732874 1721689436
  Age: 1462
  Via: 1.1 varnish
  Connection: keep-alive
  X-Cache: HIT
  X-Cache-Hits: 258
Length: unspecified [text/html]
Saving to: ‘test-redir’

test-redir       [   <=>                                 ] 166,90K   369KB/s    in 0,5s

2020-03-30 17:32:33 (369 KB/s) - ‘test-redir’ saved [170909]

```

- Dans un premier temps avec le client Chrome ou wget on obtien la même chose qu'avec le client telnet mais le client Chrome ou wget effectue la redireation automaticment et fait les requetes suivant.

### 1.4 Méthode `GET` avec ou sans `Host:`

- Lors ce que je fait cette commande `telnet bruno.pinaud.emi.u-bordeaux.fr 80` avec la requête `GET / HTTP/1.0\n\n` je n'obtien pas la même page que dans le navigateur. La cause de cette differance est du au fait qu'il y a un seul serveur qui s'occupe des requete a destination des hosts `<prenom>.<nom>.emi.u-bordeaux.fr` ce qui a pour effect si on ne presise pas le champ `Host:...` de nous rediriger sur la page par defaut. Dans le navigateur le champ `Host:...` est completter automatiquement ce qui fait que le serveur permet l'accer aux resources de l'host `bruno.pinaud.emi.u-bordeaux.fr`. Pour en arriver a cette conclusion j'ai effectuer la commande `telnet` avec l'host `charles.goedefroit.emi.u-bordeaux.fr` ce qui mas donner la même page.

GET / HTTP/1.0

HTTP/1.1 200 OK
Date: Mon, 30 Mar 2020 15:51:09 GMT
Server: Apache
Last-Modified: Thu, 03 Feb 2011 14:29:55 GMT
ETag: "1f55-49b61996c7ec0"
Accept-Ranges: bytes
Content-Length: 8021
Vary: Accept-Encoding
Connection: close
Content-Type: text/html
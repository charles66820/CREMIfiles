# TD02

## Protocole ARP

- Lorsque je tape la commande `arp -n` j'obtien bien l'adresse MAC du routeur :

  ```bash
  Address                  HWtype  HWaddress           Flags Mask            Iface
  10.0.206.254             ether   58:20:b1:b1:23:00   C                     eth0
  ```

- Lorsque j'effectue un ping vers la machine de mon voisin je constate bien que sont adresse MAC est dans ma table ARP :

    ```bash
    Address                  HWtype  HWaddress           Flags Mask            Iface
    10.0.206.6               ether   d8:9e:f3:10:4a:1c   C                     eth0
    10.0.206.254             ether   58:20:b1:b1:23:00   C                     eth0
    ```

- Je constate bien que la commande `ip neigh ls` affiche la même chose que la commande `arp -n` et affiche en plus les adresses ipv6 :

  ```bash
  10.0.206.6 dev eth0 lladdr d8:9e:f3:10:4a:1c STALE
  10.0.206.254 dev eth0 lladdr 58:20:b1:b1:23:00 REACHABLE
  2001:660:6101:800:206::ffff dev eth0 lladdr 58:20:b1:b1:23:00 router STALE
  fe80::5a20:b1ff:feb1:2300 dev eth0 lladdr 58:20:b1:b1:23:00 router STALE
  ```

- Leurs adresses MAC n'apparaissent pas dans la table ARP car ces 2 adresses ne sont pas dans mon reseau lecal.

## 2 Résolution de noms (DNS)

- Il y a plusieurs adresses IP dans le fichier `/etc/resolv.conf` car il y a un serveur DNS de secoure au cas ou le premier ne fonctionne plus.

- ✓

- Il y a plusieurs adresses IP car il y a plusieur servers. Le résultat est différent car ça dépent dequel serveur répond en premier.

  ```bash
  yahoo.com has address 72.30.35.10
  yahoo.com has address 98.137.246.7
  yahoo.com has address 98.137.246.8
  yahoo.com has address 98.138.219.231
  yahoo.com has address 98.138.219.232
  yahoo.com has address 72.30.35.9
  yahoo.com has IPv6 address 2001:4998:58:1836::11
  yahoo.com has IPv6 address 2001:4998:c:1023::4
  yahoo.com has IPv6 address 2001:4998:c:1023::5
  yahoo.com has IPv6 address 2001:4998:44:41d::3
  yahoo.com has IPv6 address 2001:4998:44:41d::4
  yahoo.com has IPv6 address 2001:4998:58:1836::10
  ```

- ✓

## 3 Services au CREMI : LDAP & NFS

- Il y a plusieurs serveurs pour s'adapter au nombre d'utilisateur et également en cas de probléme il y a des serveur de secoure. Le numéro de port est le 389 (636 pour ldpa en SSL). J'obtien c'est ports avec la command `cat /etc/services | grep ldap` :

  ```bash
  ldap    389/tcp    # Lightweight Directory Access Protocol
  ldap    389/udp
  ldaps   636/tcp    # LDAP over SSL
  ldaps   636/udp
  ```

- Le resultat de la commande `df ~` est :

  ```bash
  Filesystem            1K-blocks   Used Available Use% Mounted on
  unityaccount:/account   4194304 773632   3420672  19% /autofs/unityaccount/cremi
  ```

  `cat /etc/services | grep nfs`
  nfs             2049/tcp                        # Network File System
  nfs             2049/udp                        # Network File System

## 4 Analyse de Trames avec Wireshark

### 4.1 Préambule

- Les informations de la machine utilisée sont :
  - adresse ipv4 : `10.0.2.25`
  - masque du réseau : `255.255.255.0`
  - adresse de la passerelle : `10.0.2.2`
  - adresse du serveur DNS : `10.0.2.3`

### 4.2 Prise en main de Wireshark

- J'ai ouvert le fichier ping.pcap et j'ai regarder les différante trames :
![Capture d'écrant de Wireshark](TP2_4.2.png)

### 4.3 Ping

- L'adresse Ethernet à qui est destinier la requte ARP dans la trame 1 est `FF FF FF FF FF FF`. On doit utiliser un broadcast pour conaitre l'adrese Mac de la machine qui à l'adresse IP : `10.0.2.15`. La machine qui à l'adresse IP répondra avec son adresse MAC.
- Le protocole de transport utiliser pour les trames 3 et 4 est UDP. L'adresse IP de la machine www.google.com retourné par le serveur DNS est : `172.271.19.132`.
- On cherche a trouver l'adresse Ethernet (MAC) de la machine 10.0.2.2 au lieu de l'adresse de la machine cible www.google.com car cette machine n'est pas dans le même réseau donc on cherche à avoir l'adresse MAC de la passerelle (10.0.2.2).
- J'observe que la valeur du champs type dans l'entete ICMP change pour la trame 7 c'est `8` pour `Echo (ping) request` et our la trame 8 c'est `0` pour `Echo (ping) reply`.

### 4.4 Une page Web : je suis perdu !

- Le port source de la trame 7 est `37090`. le port destination est `80`. Le flag SYN veut dire syncronise. Ce flag permet d'initialliser une connexion TCP (synchrone).
- Les tramede la conversation TCP qui corresopnde à la requete HTTP et à la réponce HTTP sont : la trame `10` pour la requete et la trame `12` pour la réponce.
- Le role des champs sont :
  - User-Agent : Est égale au navigateur internet qui est utiliser.
  - Host : C'est le nom de domain ou l'adresse IP du serveur web.
  - Connection : Indique si la connexion doit ce fermer ou pas à la fin du chargement de la page.
- Oui elle est là.
- Dans la réponse HTTP on peut déduire :
  - le logiciel serveur web est `Apache`
  - la longeur du contenu dans la réponse est `204`
  - le type du contenu est `text/html`
- Oui je vois le code HTML de la page.
- Quant on fait "clic droit" sur un des paquets TCP puis "Follow" on vois la requéte est la réponce HTTP :
  ```HTTP
  GET / HTTP/1.1
  User-Agent: Wget/1.20.1 (linux-gnu)
  Accept: */*
  Accept-Encoding: identity
  Host: perdu.com
  Connection: Close

  HTTP/1.1 200 OK
  Date: Wed, 19 Feb 2020 18:44:39 GMT
  Server: Apache
  Upgrade: h2
  Connection: Upgrade, close
  Last-Modified: Thu, 02 Jun 2016 06:01:08 GMT
  ETag: "cc-5344555136fe9"
  Accept-Ranges: bytes
  Content-Length: 204
  Vary: Accept-Encoding
  Content-Type: text/html

  <html><head><title>Vous Etes Perdu ?</title></head><body><h1>Perdu sur l'Internet ?</h1><h2>Pas de panique, on va vous aider</h2><strong><pre>    * <----- vous &ecirc;tes ici</pre></strong></body></html>
  ```

## 5 Capture de Trames avec Wireshark

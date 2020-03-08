# TP 03

## 1 Introduction aux Sockets avec le protocole Daytime

### 1.1 Utilisation du service Daytime avec Telnet

- La commande `telnet time.nist.gov daytime` permet de ce connecter a la machine **time.nist.gov** avec le protocol telnet et ensuite ce connecter au protocole daytime. Cela a pour effect de d'afficher la date et l'heurs en temps univercel (UTC).
- Le protocole daytime est un protocole qui ce trouve sur la couche application du model TCP/IP. Il peut fonctionner avec le protocole TCP ou UDP. Il utilise le port 13.
- Avec la commande `cat /etc/services | grep daytime` je trouve :

```txt
daytime         13/tcp
daytime         13/udp
```

Je test donc de me connecter directement sur le port 13 avec la commande `netcat time.nist.gov 13` et j'obtien le m^eme resultat.

- Avec la commande `cat /etc/services | grep -w 'http\s\|ftp\s\|smtp\s\|telnet\s\|ssh\s\|echo\s'` je trouve les ports par defaut suivant :

```txt
echo            7/tcp
echo            7/udp
ftp             21/tcp
ssh             22/tcp                          # SSH Remote Login Protocol
telnet          23/tcp
smtp            25/tcp          mail
http            80/tcp          www             # WorldWideWeb HTTP
at-echo         204/tcp                         # AppleTalk echo
at-echo         204/udp
echo            4/ddp                   # AppleTalk Echo Protocol
```

- On ne peut-on pas utiliser la commande telnet pour faire ces expériences car le protocol telnet existe seulement en mode connecter (en TCP). La principal differance entre le protocol TCP et le protocol UDP est que TCP demande un accuser de reseption donc est plus fiable que UDP. En TCP le service daytime etablie une connection pour donner la date et l'heurs et est sasure aue l'information est bien recu. En UDP le service daytime se contente juste d'envoyer la date et l'heurs.

### 1.2 Programmation d’un client Daytime avec les Socket

- ✓
- ✓
- Reponse aux question sur le programme
  - La constantes AF_INET definit le type d'adresse utiliser donc le type de socket. Il existe AF_INET6 pour l'ipv6.
  - La constantes SOCK_STREAM definit le protocole de la couche transport utiliser. SOCK_STREAM pour TCP et SOCK_DGRAM pour UDP.
  - La variable s stock un objet de type socket. Cette object permet d'effectuer la connexion sur le r.
  - La ligne 5 déclenchent la connexion.
  - La ligne 8 déclenchent la déconnexion TCP/IP.
  - La method recv (pour receive) prend en paramettre un nombre d'octets. Elle permet lire la reponce de la taille du nombre d'octets.
  - Selon la documentation la method qui sert à envoyer des données est sendall.
- ✓

## 2 Socket Python : requête HTTP à la main

### 2.1 Echauffons-nous d’abord avec Telnet

- ✓

### 2.2 Programmons notre client web !

- 
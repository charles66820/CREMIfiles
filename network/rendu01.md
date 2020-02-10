## 1 Interfaces réseau et Adresse IP

- Résultat de la commandi`ifconfig`:
```
eth0: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 9000
        inet 10.0.206.15  netmask 255.255.255.0  broadcast 10.0.206.255
        inet6 fe80::da9e:f3ff:fe10:a82a  prefixlen 64  scopeid 0x20<link>
        inet6 2001:660:6101:800:206::15  prefixlen 80  scopeid 0x0<global>
        ether d8:9e:f3:10:a8:2a  txqueuelen 1000  (Ethernet)
        RX packets 2621201  bytes 2018019148 (1.8 GiB)
        RX errors 0  dropped 0  overruns 0  frame 0
        TX packets 2983534  bytes 1959199928 (1.8 GiB)
        TX errors 0  dropped 0 overruns 0  carrier 0  collisions 0
        device interrupt 16  memory 0xf7100000-f7120000

lo: flags=73<UP,LOOPBACK,RUNNING>  mtu 65536
        inet 127.0.0.1  netmask 255.0.0.0
        inet6 ::1  prefixlen 128  scopeid 0x10<host>
        loop  txqueuelen 1  (Local Loopback)
        RX packets 267  bytes 37136 (36.2 KiB)
        RX errors 0  dropped 0  overruns 0  frame 0
        TX packets 267  bytes 37136 (36.2 KiB)
        TX errors 0  dropped 0 overruns 0  carrier 0  collisions 0
```

- resultat de la commande `ipcalc 10.0.206.16/24` :
```
Address:   10.0.206.16          00001010.00000000.11001110. 00010000
Netmask:   255.255.255.0 = 24   11111111.11111111.11111111. 00000000
Wildcard:  0.0.0.255            00000000.00000000.00000000. 11111111
=>
Network:   10.0.206.0/24        00001010.00000000.11001110. 00000000
HostMin:   10.0.206.1           00001010.00000000.11001110. 00000001
HostMax:   10.0.206.254         00001010.00000000.11001110. 11111110
Broadcast: 10.0.206.255         00001010.00000000.11001110. 11111111
Hosts/Net: 254                   Class A, Private Internet
```

- L'interface lo c'est l'interface loopback cette interface correspond à notre propre machine.

- Le MTU c'est la tailles maximales des paquets il est limiter à 9000 pour eth0 car 'est la limitation du réseau. Et il est à 65536 pour lo car'est la taille maximum possible pour notre machine local.


- `ping -4 10.0.206.16` et `ping -6 2001:660:6101:800:206::16` fonctionne et me retourne un temps de latance.

## 2 Netcat & Netstat
- résultat de la commande `ss -Ainet -a | grep 12345` :
```
cgoedefroit@dwar:~$ ss -Ainet -a | grep 12345
tcp    LISTEN     0      128     *:12345                 *:*
tcp    LISTEN     0      128    :::12345                 :::*
```


- résultat de la commande `ss -Ainet -ap | grep 12345` :
```
tcp    LISTEN     0      128     *:12345                 *:*                     users:(("nc",pid=21787,fd=4))
tcp    LISTEN     0      128    :::12345                 :::*                     users:(("nc",pid=21787,fd=3))
```
- lors ce que la connexion est établie la command `ss -Ainet -ap | grep 12345` nous permet de voire cette connexion.
```
tcp    LISTEN     0      128     *:12345                 *:*
tcp    ESTAB      0      0      10.0.206.15:35616        10.0.206.16:12345
tcp    LISTEN     0      128    :::12345                 :::*
```

- Quand on tape des lignes de code d'un des deux côté ces lignes son bien transmis de l'autre côté.

- Si on met --send-only du côté de l'émetteur on ne resois plus ce qeu le reseveur envois

## 3 Connexion à une machine distante avec SSH

- **xeyes** s'erxecute sur la machine distante l'option -X permet de faire du X11 forwarding pour que le rendu graphic ce face sur notre machine local. -Y existe égalemnt est est plus récent.

## 4 Configuration d’un réseau local
- je me suis bien connecté en root

- j'utilise la command `ifconfig -a -s` pour voir les interfaces réseaux :
```
Iface      MTU    RX-OK RX-ERR RX-DRP RX-OVR    TX-OK TX-ERR TX-DRP TX-OVR Flg
eth0      1500        0      0      0 0             0      0      0      0 BM
lo       65536      896      0      0 0           896      0      0      0 LRU
```
- ce réseau à
  - pour adresse : 192.168.0.0
  - pour masque : 255.255.255.0 (/24)
  - pour plage d'adresse de 192.168.0.1 à 192.168.0.254
  - pour adresse de broadcast 192.168.0.255

- J'ai configurée immortal avec la command suivant `ifconfig eth0 192.168.0.1/24` puis les 3 autre machine avec les ip suivant :
  - immortal : 192.168.0.1/24
  - grave : 192.168.0.2/24
  - syl : 192.168.0.3/24
  - opeth : 192.168.0.4/24

- J'ai les machine sont bien configurée et dans le même réseau, elles peuve toute ce ping entre elles. La commande ping utilise le protocole **icmp**.

- Lors ce que sur la machine opeth le lanche tcpdump sur l'interface eth0 (`tcpdump -i eth0`) et que je ping avec la machine immortal (`ping -c 4 192.168.0.4`) on vois le fonctionnement de la command ping notamant qu'il utilise le protocole ICMP. On vois si c'est une requête ou une réponce, la taille du pacquet, l'adresse source et l'adresse destination...
```
[ 1995.801415] device eth0 entered promiscuous mode
16:45:07.868220 IP 192.168.0.1 > 192.168.0.4: ICMP echo request, id 1363, seq 1, length 64
16:45:07.868233 IP 192.168.0.4 > 192.168.0.1: ICMP echo reply, id 1363, seq 1, length 64
16:45:08.867707 IP 192.168.0.1 > 192.168.0.4: ICMP echo request, id 1363, seq 2, length 64
16:45:08.867737 IP 192.168.0.4 > 192.168.0.1: ICMP echo reply, id 1363, seq 2, length 64
16:45:09.869076 IP 192.168.0.1 > 192.168.0.4: ICMP echo request, id 1363, seq 3, length 64
16:45:09.869105 IP 192.168.0.4 > 192.168.0.1: ICMP echo reply, id 1363, seq 3, length 64
16:45:10.870566 IP 192.168.0.1 > 192.168.0.4: ICMP echo request, id 1363, seq 4, length 64
16:45:10.870597 IP 192.168.0.4 > 192.168.0.1: ICMP echo reply, id 1363, seq 4, length 64
16:45:12.869542 ARP, Request who-has 192.168.0.1 tell 192.168.0.4, length 28
16:45:12.872061 ARP, Reply 192.168.0.1 is-at aa:aa:aa:aa:00:00 (oui Unknown), length 46
```

- La machine où est exécuter la commande `ping -c 4 -b 192.168.0.255` //TODO: je suis ici
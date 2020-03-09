# TP 01

## 1 Interfaces réseau et Adresse IP

- Résultat de la commande `ifconfig` :

```bash
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

- Résultat de la commande `ipcalc 10.0.206.16/24` :

```bash
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

- L'interface `lo` c'est l'interface loopback cette interface correspond à notre propre machine.

- Le MTU c'est la taille maximale des paquets, il est limité à 9000 pour eth0 car c'est la limitation du réseau. Et il est à 65536 pour lo car c'est la taille maximum possible pour notre machine locale.

- `ping -4 10.0.206.16` et `ping -6 2001:660:6101:800:206::16` fonctionne et retourne un temps de latence.

## 2 Netcat & Netstat

- Résultat de la commande `ss -Ainet -a | grep 12345` :

```bash
tcp    LISTEN     0      128     *:12345         *:*
tcp    LISTEN     0      128    :::12345         :::*
```

- Résultat de la commande `ss -Ainet -ap | grep 12345` :

```bash
tcp  LISTEN   0   128   *:12345    *:*    users:(("nc",pid=21787,fd=4))
tcp  LISTEN   0   128  :::12345    :::*   users:(("nc",pid=21787,fd=3))
```

- Lorsque la connexion est établie la commande `ss -Ainet -ap | grep 12345` nous permet de voir cette connexion :

```bash
tcp    LISTEN     0      128     *:12345           *:*
tcp    ESTAB      0      0      10.0.206.15:35616  10.0.206.16:12345
tcp    LISTEN     0      128    :::12345           :::*
```

- Quand on tape des lignes de code d'un des deux côté ces lignes sont bien transmis de l'autre côté.

- Si on met `--send-only` du côté de *l'émetteur* on ne reçoit plus ce que *le receveur* envoit.

## 3 Connexion à une machine distante avec SSH

- **xeyes** s'exécute sur la machine distante l'option -X permet de faire du X11 forwarding pour que le rendu graphique se fasse sur notre machine local. -Y existe également et elle est plus récente.

## 4 Configuration d’un réseau local

- Je me suis bien connecté en root.

- J'utilise la commande `ifconfig -a -s` pour voir les interfaces réseaux de ma machine :

```bash
Iface   MTU  RX-OK RX-ERR RX-DRP RX-OVR TX-OK TX-ERR TX-DRP TX-OVR Flg
eth0   1500      0      0      0 0          0      0      0      0 BM
lo    65536    896      0      0 0        896      0      0      0 LRU
```

- Ce réseau a
  - pour adresse : 192.168.0.0
  - pour masque : 255.255.255.0 (/24)
  - pour plage d'adresse de 192.168.0.1 à 192.168.0.254
  - pour adresse de broadcast 192.168.0.255

- J'ai configuré la VM immortal avec la commande `ifconfig eth0 192.168.0.1/24` puis les 3 autres VMs avec les ip suivantes :
  - immortal : 192.168.0.1/24
  - grave : 192.168.0.2/24
  - syl : 192.168.0.3/24
  - opeth : 192.168.0.4/24

- Les VMs sont bien configurées et elles sont dans le même réseau. Elles peuvent toutes communiquer entre elles (avec `ping` par exemple). La commande `ping` utilise le protocole **icmp**.

- Lorsque je lance `tcpdump` sur l'interface eth0 de la VM opeth (`tcpdump -i eth0`) et que je la ping avec la VM immortal (`ping -c 4 192.168.0.4`) on voit le fonctionnement de la commande `ping` notamant qu'elle utilise le protocole ICMP. On voit également si c'est une requête ou une réponse, la taille du paquet, l'adresse source et l'adresse destination...

```bash
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

- La VM où est exécutée la commande `ping -c 4 -b 192.168.0.255` envoit un paquet ICMP à toutes les autres (en broadcast). Les autres VMs reçoivent le paquet par contre elles ne répondent pas.
Lorsque j'exécutais la commande `sysctl net.ipv4.icmp_echo_ignore_broadcasts=0` sur des VMs elles répondent au ping en broadcast. Si plusieurs VMs répondent un `(DUP!)` apparé à la fin des réponses.

- La configuration de l'interface des VMs sont :
  - immortal :

    ```bash
    auto eth0
    iface eth0 inet static
        address 192.168.0.1
        netmask 255.255.255.0
    ```

  - grave :

    ```bash
    auto eth0
    iface eth0 inet static
        address 192.168.0.2
        netmask 255.255.255.0
    ```

  - syl :

    ```bash
    auto eth0
    iface eth0 inet static
        address 192.168.0.3
        netmask 255.255.255.0
    ```

  - opeth :

    ```bash
    auto eth0
    iface eth0 inet static
        address 192.168.0.4
        netmask 255.255.255.0
    ```

- J'ai fermé ma session QemuNet après avoir configuré les VMs en IPv6.

- Configuration des VMs en IPv6 (Bonus) :
  - immortal :

    ```bash
    auto eth0
    iface eth0 inet6 static
        address 2001:db8::1
        netmask 48
    ```

  - grave :

    ```bash
    auto eth0
    iface eth0 inet6 static
        address 2001:db8::2
        netmask 48
    ```

  - syl :

    ```bash
    auto eth0
    iface eth0 inet6 static
        address 2001:db8::3
        netmask 48
    ```

  - opeth :

    ```bash
    auto eth0
    iface eth0 inet6 static
        address 2001:db8::4
        netmask 48
    ```

- Configuration des VMs avec la commande `ip` (Bonus) :
  - immortal : `ip a flush dev eth0 && ip a a 192.168.0.1/24 brd + dev eth0`
  - grave : `ip a flush dev eth0 && ip a a 192.168.0.2/24 brd + dev eth0`
  - syl : `ip a flush dev eth0 && ip a a 192.168.0.3/24 brd + dev eth0`
  - opeth : `ip a flush dev eth0 && ip a a 192.168.0.4/24 brd + dev eth0`

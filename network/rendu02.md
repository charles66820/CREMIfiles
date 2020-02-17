# TD02

## Protocole ARP

- Lors ce que je tape la commande `arp -n` j'obtien bien l'adresse MAC du routeur :

  ```bash
  Address                  HWtype  HWaddress           Flags Mask            Iface
  10.0.206.254             ether   58:20:b1:b1:23:00   C                     eth0
  ```

- Lors ce que j'effectue un ping vers la machine de mon voisin je constate bien que sont adresse MAC est dans ma table ARP :

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

## Services au CREMI : LDAP & NFS

- Il y a plusieurs serveurs pour s'adapter au nombre d'utilisateur et également en cas de probléme il y a des serveur de secoure.
  Le numéro de port est le 389 (636 pour ldpa en SSL). J'obtien c'est ports avec la command `cat /etc/services | grep ldap` :

  ```bash
  ldap    389/tcp    # Lightweight Directory Access Protocol
  ldap    389/udp
  ldaps   636/tcp    # LDAP over SSL
  ldaps   636/udp
  ```

- 
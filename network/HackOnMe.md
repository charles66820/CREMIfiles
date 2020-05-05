# Hack on Me!

## flags 0

- ifconfig eth0 192.168.0.20/24
- nmap 192.168.0.100
- nc 192.168.0.100 2000

> result : WHX4D8

## flag 1

- ifconfig eth0 192.168.0.20/24
- nmap 192.168.0.0/24
- nmap -p 1000-2000 192.168.0.116
- nc 192.168.0.116 1425

> result : 9XG97DX

## flag 2

- ifconfig eth0 192.168.0.2/24
- nmap 192.168.0.0/24
- route add default gw 192.168.0.20
- nc 147.210.0.1 2000

> result : CE0ZGL

## flag 3

- ifconfig eth0 192.168.0.2/24
- ifconfig eth1 10.0.0.2/24
- tcpdump -n -i any
- echo 1 > /proc/sys/net/ipv4/ip_forward
- route add default gw 10.0.0.1
- nc 147.210.0.230 2000

> result : FBIDCW2

## flag 4

- nmap -p 1000-2000 192.168.0.1
- iptables -P FORWARD DROP
- iptables -A FORWARD -s 147.210.0.1 -d 192.168.0.1 -p tcp --dport 1890 -j ACCEPT
- iptables -A FORWARD -m state --state RELATED,ESTABLISHED -j ACCEPT
- nc 147.210.0.1 2000

> result : EOWPPI2

## flag 5

- iptables -t nat -A POSTROUTING -o eth1 -j MASQUERADE
- iptables -t nat -A PREROUTING -i eth1 -p tcp --dport 22 -j DNAT --to-destination 192.168.0.1:22
- nc 147.210.0.1 2000

> result : 3FOTNDG

## flag 6

- nmap 192.168.0.0/24
- ssh -f -N toto@192.168.0.168 -L 2000:147.210.0.1:2000
- nc localhost 2000

> result : 1PCXEJ48
> -N for no cmd and -f for forground
> -L for create localhost:2000 => 147.210.0.1:2000 thraw connexion 192.168.0.168

## flag 7

- nmap 192.168.0.0/24
- ssh -f -N -J toto@192.168.0.173 toto@147.210.0.1 -L 2000:localhost:2000
- nc localhost 2000

> result : AJQA59N2

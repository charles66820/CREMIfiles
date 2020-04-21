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

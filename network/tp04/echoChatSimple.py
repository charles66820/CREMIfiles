#!/usr/bin/python3.6
import os
import re
import select
import socket
import sys

# get host of client
def getScHost(sc) :
    return sc.getpeername()[0] + ':' + str(sc.getpeername()[1])

# get selecte comment and execut it
def commands(scSender, cmd, msg) :
    if cmd == 'MSG' or cmd == 'QUIT':
        if msg == '' :
            if settings['notifs']['invalidCommand'] : scSender.sendall(('[' + nicks[scServer] + '] ' + 'Invalid message !\n').encode("utf-8"))
            if logs : print('invalid command')
        else :
            sendall(scSender, '[' + nicks[scSender] + '] ' + msg)
            if cmd == 'QUIT': disconnect(scSender)
    elif cmd == 'NICK' :
        m = re.match(r"^([^\s]+)$", msg)
        if m :
            if logs : print('client "' + nicks[scSender] + '" => "' + m[1] + '"')
            nicks[scSender] = m[1]
        else :
            if settings['notifs']['invalidCommand'] : scSender.sendall(('[' + nicks[scServer] + '] ' + 'Invalid nickname !\n').encode("utf-8"))
            if logs : print('invalid command')
    elif cmd == 'WHO':
        online = ''
        for scClient in scList :
            if scClient != scServer :
                online += ' ' + nicks[scClient]
        scSender.sendall(('[' + nicks[scServer] + ']' + online + '\n').encode("utf-8"))
        del online
    elif cmd == 'KILL' :
        m = re.match(r"^([^\s]+)\s(.+)", msg)
        if m :
            for sc, nick in nicks.items() :
                if nick == m[1] :
                    sc.sendall(('[' + nicks[scSender] + '] ' + m[2] + '\n').encode("utf-8"))
                    disconnect(sc)
                    break
        else :
            if settings['notifs']['invalidCommand'] : scSelected.sendall(('[' + nicks[scServer] + '] ' + 'Invalid nickname !\n').encode("utf-8"))
            if logs : print('invalid command')
    else :
        if settings['notifs']['invalidCommand'] : scSender.sendall(('[' + nicks[scServer] + '] ' + 'Invalid command !\n').encode("utf-8"))
        if logs : print('invalid command')

# Semd all messate to everybody without sever and sender
def sendall(scSender, msg) :
    for scClient in scList :
        if scClient != scSender and scClient != scServer :
            scClient.sendall((msg + "\n").encode("utf-8"))

def disconnect(scClient) :
    sendall(scClient, '[' + nicks[scServer] + '] ' + nicks[scClient] + " is disconnected")
    if logs : print('client disconnected "' + nicks[scClient] + '"')
    del nicks[scClient]
    scList.remove(scClient)
    scClient.close()

try:
    logs = True
    settings = {
        "notifs": {
            "connection": False,
            "connectionM": True,
            "deconnexion": False,
            "deconnexionM": True,
            "invalidCommand": False,
            "motd": False
        }
    }

    host = ""
    port = 7777

    scServer = socket.socket(socket.AF_INET6, socket.SOCK_STREAM, 0)
    try :

        scServer.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        scServer.bind((host, port))
        scServer.listen(1)
        if logs :
            # Chat server is running in verbose mode
            print("Welcome to Chat Server")
        else :
            print("Chat server is running")

        try :
            scList = []
            nicks = {scServer: 'server'}
            while True :
                (rl, wl, xl) = select.select(scList+[scServer], [], [])
                for scSelected in rl :
                    if scSelected == scServer :
                        # Get client socket and ip address
                        scNewClient, a = scServer.accept()

                        # Connect
                        scList.append(scNewClient)
                        nicks[scNewClient] = getScHost(scNewClient)
                        if settings['notifs']['motd'] : scNewClient.sendall(('[' + nicks[scServer] + '] ' + 'Welcome to Chat Server !\n').encode("utf-8"))
                        if settings['notifs']['connection'] : sendall(scNewClient, '[' + nicks[scServer] + '] ' + nicks[scNewClient] + " is connected")
                        if settings['notifs']['connectionM'] : sendall(scNewClient, 'client connected "' + nicks[scNewClient] + '"')
                        if logs : print('client connected "' + nicks[scNewClient] + '"')
                    else :
                        data = scSelected.recv(2008).decode("utf-8") #8 + 2000 + 10
                        if data == "" :
                            disconnect(scSelected)
                        else :
                            m = re.match(r"^([A-Z]+)\s(.+)|^([A-Z]+)\s?", data) # RegEx au lieu de split
                            if m != None :
                                if m[3] == None and len(m[2]) > 2000 :
                                    scSelected.sendall('The message must be less than 2000 characters !\n'.encode("utf-8"))
                                else :
                                    if m[3] :
                                        commands(scSelected, m[3], '')
                                    else :
                                        commands(scSelected, m[1], m[2])
                            else :
                                if settings['notifs']['invalidCommand'] : scSender.sendall(('[' + nicks[scServer] + '] ' + 'Invalid command !\n').encode("utf-8"))
                                if logs : print('invalid command')

        except Exception as e :
            print("Error with accept : ", e)
            scServer.close()

    except Exception as e :
        print("Error on bind the socket : ", e)

except KeyboardInterrupt:
    print('Interrupted')
    scServer.close()
    try:
        sys.exit(0)
    except SystemExit:
        os._exit(0)

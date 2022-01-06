# Import socket module
import configparser
import socket

from decorators import Singleton


@Singleton
class SocketClient():

    def __init__(self):
        #this is the constructor that takes in host and port. retryAttempts is given
        # a default value but can also be fed in.
        config = configparser.ConfigParser()
        config.read('config.properties')
        self.host = config.get("SocketServer", "host")
        self.port = int(config.get("SocketServer", "port"))
        self.socket = None
        self.is_connected = False
        #self.connect()

    def connect(self):
        if not self.is_connected:
            self.socket = socket.socket()
            self.socket.connect((self.host, self.port))
            self.is_connected = True

    def diconnectSocket(self):
        if self.is_connected:
            self.socket.close()
            self.socket = None

    def send(self, msg):
        if self.is_connected:
            self.socket.send(msg.encode('utf-8'))
        else:
            raise Exception('no connection to server')

    def read(self, size):
        if self.is_connected:
            buff = bytearray()
            while len(buff) < size:
                buff.extend(self.socket.recv(4096))
            return buff
        else:
            raise Exception('no connection to server')
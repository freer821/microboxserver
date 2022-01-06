from socketclient import SocketClient

client = SocketClient.instance()

def image_scan():
    try:
        client.send('DoScan\n')
        print(client.read(5))
        client.send('GetImageInfo\n')
        buff = client.read(10)
        height = int.from_bytes([buff[1], buff[2], buff[3], buff[4]], "little")
        width = int.from_bytes([buff[5], buff[6], buff[7], buff[8]], "little")
        channels = int.from_bytes([buff[9], buff[10], buff[11], buff[12]], "little")
        imagesize = int.from_bytes([buff[21], buff[22], buff[23], buff[24]], "little")
        print('image size: %d, width %d, height: %d, channels: %d' + imagesize, width, height, channels)

        client.send('GetImageData\n')
        return client.read(imagesize), width, height, channels
    except Exception as ex:
        client.is_connected = False
        client.connect()
        raise ex

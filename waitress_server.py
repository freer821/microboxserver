from waitress import serve
import main
from waitress.server import create_server

serve(main.app, host='0.0.0.0', port=8080)


# import your flask app

class WaitressServer:

    def __init__(self, host, port):
        self.server = create_server(main.app, host=host, port=port)

    # call this method from your service to start the Waitress server
    def run(self):
        self.server.run()

    # call this method from the services stop method
    def stop_service(self):
        self.server.close()

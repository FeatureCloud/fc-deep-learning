from bottle import Bottle
from FeatureCloud.app.api.http_ctrl import api_server
from FeatureCloud.app.api.http_web import web_server
from FeatureCloud.app.engine.app import app
from utils.utils import is_native
import states

server = Bottle()


def run_app():
    server.mount('/api', api_server)
    server.mount('/web', web_server)
    server.run(host='localhost', port=5000)


if __name__ == '__main__':
    app.register()
    if is_native():
        app.handle_setup(client_id='1', coordinator=True, clients=['1'])
    else:
        run_app()

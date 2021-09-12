import json
import time

from bottle import Bottle, request

from .Customapp import logic

api_server = Bottle()


# CAREFUL: Do NOT perform any computation-related tasks inside these methods, nor inside functions called from them!
# Otherwise your app does not respond to calls made by the FeatureCloud system quickly enough
# Use the threaded loop in the app_flow function inside the file logic.py instead


@api_server.post('/setup')
def ctrl_setup():
    time.sleep(1)
    print(f'[CTRL] POST /setup', flush=True)
    payload = request.json
    logic.handle_setup(payload['id'], payload['master'], payload['clients'])
    return ''


@api_server.get('/status')
def ctrl_status():
    print(f'[CTRL] GET /status', flush=True)
    return json.dumps({
        'available': logic.status_available,
        'finished': logic.status_finished,
    })


@api_server.route('/data', method='GET')
def ctrl_data_out():
    print(f'[CTRL] GET /data', flush=True)
    return logic.handle_outgoing()


@api_server.route('/data', method='POST')
def ctrl_data_in():
    print(f'[CTRL] POST /data', flush=True)
    logic.handle_incoming(request.body)
    return ''

from bottle import Bottle, HTTPResponse, request, run as bottle_run
import logging
import json
from random import randint
import sys
from time import sleep

# naming convention for uWSGI apps
application = app = Bottle()

# set up logging
logging.basicConfig(
    format='%(asctime)-15s %(levelname)s %(name)s %(funcName)s - %(message)s',
    level=logging.DEBUG)
logger = logging.getLogger('augmentor')

@app.post('/augmentor/test')
def augment():
    #logger.debug('received %s' % json.dumps(
    #        request.json, indent=4, separators=(',', ': ')))
    #logger.info('matched agents %s' % request.json['ext']['agents'])
    
    response = [
        {
            "account":["account_bla","child"],
            "augmentation":{
                "data":None,
                "tags":["augmentor-pass"]
            }
         }
    ]
    if randint(0,1):
        response[0]["augmentation"]["tags"] = ["augmentor-noinfo"]
    #if randint(0,10) < 1:
    #    sleep(0.5)
    raise HTTPResponse(body=json.dumps(response),
                       Content_Type='application/json',
                       x_rtbkit_protocol_version='1.0',
                       x_rtbkit_timestamp=1234,
                       x_rtbkit_auction_id=request.json['id'],
                       x_rtbkit_augmentor_name=sys.argv[1],
                       status=200)


if __name__ == '__main__':
    logger.debug('starting up debug server')
    bottle_run(app, host='0.0.0.0', port=int(sys.argv[2]), reloader=True)


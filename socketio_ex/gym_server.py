from flask import Flask, request
from flask_restful import Resource, Api

import json
import os
from datetime import datetime, date, time
from flask_socketio import SocketIO, Namespace, send, emit
import shortuuid

import numpy as np
import cv2

import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__, static_folder='static', static_url_path='')
api = Api(app)

socketio = SocketIO(app)

#------- for connect and get id------#
@socketio.on('connect', namespace='/')
def connect_begin():
    print('Server in connect_begin()')


@socketio.on('session', namespace='/')
def session():
    print('in Connect')
    new_id = shortuuid.uuid()
    ns = '/' + new_id + '/predict' 

    print('Build server predict socket withs ns: {}'.format(ns))
    socketio.on_namespace(PredictNamespace(ns, new_id) )
    emit('session_response', new_id)


#------ Dynamic Namespce Predict -------#
class PredictNamespace(Namespace):
    def __init__(self,ns, client_id):
        super(PredictNamespace, self).__init__(ns)
        self.client_id = client_id

        print("{} init finish".format(self.client_id))

    def on_connect(self):
        print('{} PredictNamespace Connect'.format(self.client_id))
        self.data_dir = 'data_pool/{}/'.format(self.client_id)
        if not os.path.isdir(self.data_dir):
            os.mkdir(self.data_dir)

        self.state_count = 0

    def on_disconnect(self):
        print('{} PredictNamespace Disconnect'.format(self.client_id))

    
    def save_data(self, data):
        state = data['state']
        state = np.array(state)
        print('train state.shape={}'.format(state.shape) )
        # pic_path =  train_dir + tag_id +'+r'+ str(reward) + '_a'+  str(action) + '_0.'+ self.identity +'.jpg'
        self.state_count += 1
        pic_path = '%s/%06d.jpg' % (self.data_dir, self.state_count)
        # cv2.imwrite(pic_path, state[:,:,0])
        print('pic_path = %s' % self.data_dir)
        cv2.imwrite(pic_path, state)

    
    def on_get_action(self, data):
        predict_data = np.random.randint(4)
        emit('action_response', predict_data)

        self.save_data(data)
        

#------ Dashboard -------#
class Dashboard(Resource):
    def get(self):
        return {'url':'in dashboard'}
# you can try on http://localhost:5000/dashboard
api.add_resource(Dashboard,'/dashboard')


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0')
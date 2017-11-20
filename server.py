#
#   Get raw picture and modify to 84*84 gray picture 
#   Modify from ZMQ example (http://zguide.zeromq.org/py:spqueue)
#   
#   Author:  Kartik, Chen  <kbehouse(at)gmail(dot)com>,
#          

import sys
import zmq
import multiprocessing
import tensorflow as tf

from worker import Worker
from utility import *
from network.ACNet import ACNet
from config import cfg


FRONTEND_ADR = "tcp://*:%d" % cfg['conn']['server_frontend_port']
BACKEND_ADR  = "tcp://*:%d" % cfg['conn']['server_backend_port']

class Server:
    def __init__(self):

        # DL Init
        self.sess = tf.Session()
        method = cfg['RL']['method']
        self.main_net = ACNet(self.sess, cfg[method]['main_net_scope'])  

        self.worker_init()
        self.connect_init()

        # DL Init 2
        COORD = tf.train.Coordinator()
        self.sess.run(tf.global_variables_initializer())



        self.check_output_graph()

    def connect_init(self):
        # Connet Init
        self.context = zmq.Context(1)

        self.frontend = self.context.socket(zmq.ROUTER) # ROUTER
        self.backend  = self.context.socket(zmq.ROUTER) # ROUTER
        self.frontend.bind(FRONTEND_ADR) # For clients
        self.backend.bind(BACKEND_ADR)  # For workers

        self.poll_workers = zmq.Poller()
        self.poll_workers.register(self.backend, zmq.POLLIN)

        self.poll_both = zmq.Poller()
        self.poll_both.register(self.frontend, zmq.POLLIN)
        self.poll_both.register(self.backend, zmq.POLLIN)


    def worker_init(self):
        # 'Can' worker list
        self.workers = []
        # 'All' worker list
        self.worker_list = []

        for i in range(cfg['conn']['server_worker_num']):
            worker_id = u"Worker-{}".format(i).encode("ascii")
            w = Worker(self.sess, worker_id, self.main_net)
            self.worker_list.append(w)


    def check_output_graph(self):
        if 'log' in cfg and cfg['log']['output_tf']:
            import os, shutil
            log_dir = cfg['log']['output_tf_dir']
            if os.path.exists(log_dir):
                shutil.rmtree(log_dir)
            tf.summary.FileWriter(log_dir, self.sess.graph)


    def start(self): 
        for w in self.worker_list:
            w.start()

        while True:
            try:
                if self.workers:
                    self.socks = dict(self.poll_both.poll())
                else:
                    self.socks = dict(self.poll_workers.poll())

            except KeyboardInterrupt:
            #This won't catch KeyboardInterupt
                print('KeyboardInterrupt Capture')
                # stop_all = [w.close_connect() for w in worker_list]
                for w in self.worker_list:
                    w.close_connect()

                self.poll_both.unregister(self.frontend)
                self.poll_both.unregister(self.backend)
                self.poll_workers.unregister(self.backend)
                
                self.frontend.close()
                self.backend.close()
                self.context.destroy(linger=0)
                
                self.context.term()
            
                break
            

            # Handle worker activity on backend
            if self.socks.get(self.backend) == zmq.POLLIN:
                # Use worker address for LRU routing
                msg = self.backend.recv_multipart()
                if not msg:
                    break
                address = msg[0]
                # print('Append address: ' + str(address))
                self.workers.append(address)

                # Everything after the second (delimiter) frame is reply
                reply = msg[2:]

                # Forward message to client if it's not a READY
                if reply[0] != LRU_READY:
                    self.frontend.send_multipart(reply)

            if self.socks.get(self.frontend) == zmq.POLLIN:
                #  Get client request, route to first available worker
                msg = self.frontend.recv_multipart()
                request = [self.workers.pop(0), ''] + msg
                self.backend.send_multipart(request)



if __name__ == '__main__':

    s = Server()
    s.start()

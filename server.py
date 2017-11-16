#
#   Get raw picture and modify to 84*84 gray picture 
#   Modify from ZMQ example (http://zguide.zeromq.org/py:spqueue)
#   
#   Author:  Kartik, Chen  <kbehouse(at)gmail(dot)com>,
#          


import zmq
import multiprocessing

from worker import Worker
from utility import *
import sys
FRONTEND_ADR = "tcp://*:5555"
BACKEND_ADR  = "tcp://*:5556"

NBR_WORKERS = 3



LRU_READY = "\x01"

context = zmq.Context(1)

frontend = context.socket(zmq.ROUTER) # ROUTER
backend = context.socket(zmq.ROUTER) # ROUTER
frontend.bind(FRONTEND_ADR) # For clients
backend.bind(BACKEND_ADR)  # For workers

poll_workers = zmq.Poller()
poll_workers.register(backend, zmq.POLLIN)

poll_both = zmq.Poller()
poll_both.register(frontend, zmq.POLLIN)
poll_both.register(backend, zmq.POLLIN)

workers = []



# Start background work tasks
# def start(task, *args):
#     process = multiprocessing.Process(target=task, args=args)
#     process.daemon = True
#     process.start()
# for i in range(NBR_WORKERS):
#     start(worker_task, i)


worker_list = []
for i in range(NBR_WORKERS):
    w = Worker(i)
    w.start()
    worker_list.append(w)


while True:
    	
    try:
        if workers:
            socks = dict(poll_both.poll())
        else:
            socks = dict(poll_workers.poll())

    except KeyboardInterrupt:
    #This won't catch KeyboardInterupt
        print('KeyboardInterrupt Capture')
        # stop_all = [w.close_connect() for w in worker_list]
        for w in worker_list:
            w.close_connect()

        poll_both.unregister(frontend)
        poll_both.unregister(backend)
        poll_workers.unregister(backend)
        
        frontend.close()
        backend.close()
        context.destroy(linger=0)
        
        context.term()
        
        
        # def __del__(self):
        
        break
    

    # Handle worker activity on backend
    if socks.get(backend) == zmq.POLLIN:
        # Use worker address for LRU routing
        msg = backend.recv_multipart()
        if not msg:
            break
        address = msg[0]
        # print('Append address: ' + str(address))
        workers.append(address)

        # Everything after the second (delimiter) frame is reply
        reply = msg[2:]

        # Forward message to client if it's not a READY
        if reply[0] != LRU_READY:
            frontend.send_multipart(reply)

    if socks.get(frontend) == zmq.POLLIN:
        #  Get client request, route to first available worker
        msg = frontend.recv_multipart()
        request = [workers.pop(0), ''] + msg
        backend.send_multipart(request)




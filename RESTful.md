# Create Session
```
URL: /session
Request: config.yaml
Response: 
      client_id:  [dynamic id]
      predict_url: http://[ip]:[port]/[dynamic_id]/predict
      train_url: http://[ip]:[port]/[dynamic_id]/ train
      EXAMPLE:
        {client_id: 2df12cdtw, 
        predict_url: http://127.0.0.1:5555/2df12cdtw/predict,
        train_url: http://127.0.0.1:5555/2df12cdtw/train}
```
# Predict
```
URL: /[client_id ]/predict
Request: 
      env_id: [id] (Depend on algorithm: Q-Learning, DQN dosen't use it, but A3C use it for logging)
      state: [data]
      EXAMPLE:
        {data: [2,5] , env_id: 3} 
Response:
      action: data
      EXAMPLE:
        {action: 3}
```
# Train
```
URL: /[client_id]/train
Request: 
      state: [data]
      next_state: [data]
      reward: data
      action: data
      done: boolean
      EXAMPLE
        {state : [2,4] , next_state: [4,5], reward: 0.5, action: 3, done: true } 
Response: 
      receive: boolean 
      EXAMPLE
        {receive: true}
```

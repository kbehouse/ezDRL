# Eazy DRL (ezDRL)

## Install

1. Download this package from `git clone git@github.com:kbehouse/ezDRL.git` or download this zip

2. Install python requirements by following command

```
pip install -r requirements.txt
```

## Run

1. Run the Server 

```
python server.py
```

2. Run the two_dof_arm (You could update this to your need) 

```
python two_dof_arm.py
```


## Version Note (v0.02)

After run `server.py` and `two_dof_arm.py`

1. Predict:
* `two_dof_arm.py` request env state (7,) to `server.py`
* `server.py` response predict action to  `two_dof_arm.py`

2. Train:
* `two_dof_arm.py` send 5 steps env data to `server.py`

   5  steps env data: state_buf (5,7) action_buf (5,1) reward_buf (5,1), next_state (7,),  done (value)


* `server.py` workers train the parameter to `Main_Net`

## Acknowledgement

1. [MorvanZhou RL Class](https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow)
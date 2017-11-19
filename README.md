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


## Version Note (v0.01)

After run `server.py` and `main.py`

1. Predict:
* `main.py` request env state (84x84x4) to `server.py`
* `server.py` response predict action ,which is random generated, to  `main.py`

2. Train:
* `main.py` request env state (84x84x4), reward, action   to `server.py`
* `server.py` save the state (84x84x4) to train/

## Acknowledgement

1. [MorvanZhou RL Class](https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow)
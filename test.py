import os

os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import torch
from src.env import create_train_env
from src.model import PPO
import torch.nn.functional as F
import wandb
wandb.init(project='Motor_rl')

import numpy as np


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of model described in the paper: Proximal Policy Optimization Algorithms """)
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--output_path", type=str, default="output")
    parser.add_argument('--saved_episode' ,type=int, default=1400, help='number of saved model')
    args = parser.parse_args()
    return args


def test(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
 
    env = create_train_env()
    model = PPO(2,2)
    print(model)

    if torch.cuda.is_available():
        model.load_state_dict(torch.load("{}/ppo_DCseries_{}".format(opt.saved_path, opt.saved_episode)))
        model.cuda()
    else:
        model.load_state_dict(torch.load("{}/ppo_DCseries_{}".format(opt.saved_path, opt.saved_episode),
                                          map_location=lambda storage, loc: storage))
    model.eval()
    states_,ref = env.reset()
    observation_ = np.array([states_[0],ref[0]])
    observation_ = torch.tensor([observation_],dtype=torch.float)
 
    while True:
        if torch.cuda.is_available():
            observation_ = observation_.cuda()
        logits, value = model(observation_)
       
        policy = F.softmax(logits, dim=1)
        action = torch.argmax(policy).item()
        (states_,ref), reward, done, info = env.step(action)

  
        env.render()
        observation = np.array([states_[0],ref[0]])
        observation = torch.tensor([observation],dtype=torch.float)
        observation_ = observation





if __name__ == "__main__":
    opt = get_args()
    test(opt)


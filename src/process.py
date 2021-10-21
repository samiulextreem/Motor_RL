from os import stat
import torch
from torch import tensor
from torch._C import dtype
from src.env import create_train_env
from src.model import PPO
import torch.nn.functional as F
from collections import deque

import numpy as np
import time



def eval(opt, global_model, num_states, num_actions):
    torch.manual_seed(123)
    env = create_train_env()
    local_model = PPO(num_states, num_actions)
    if torch.cuda.is_available():
        local_model.cuda()
    local_model.eval()
    state,ref = env.reset()
    state = np.array([state[0],ref[0]])
    state = torch.tensor(state,dtype=torch.float)
    state = state.unsqueeze(0)
    if torch.cuda.is_available():
        state = state.cuda()
    done = True
    curr_step = 0
    actions = deque(maxlen=opt.max_actions)
    total_reward = 0
    while True:
        curr_step += 1
        if done:
            local_model.load_state_dict(global_model.state_dict())
        logits, value = local_model(state)
        policy = F.softmax(logits, dim=1)
        action = torch.argmax(policy).item()
        (state,ref), reward, done, info = env.step(action)
        total_reward = total_reward + reward
        env.render()
        state = np.array([state[0],ref[0]])
        state = torch.tensor(state,dtype=torch.float)
        state = state.unsqueeze(0)    
        actions.append(action)
        if curr_step > opt.num_global_steps:
            done = True
        if done:
            print('total reward {} with total step {} average reward {}'.format(total_reward, curr_step,(total_reward/curr_step)))
            curr_step = 0
            actions.clear()
            state,ref = env.reset()
            state = np.array([state[0],ref[0]])
            state = torch.tensor(state,dtype=torch.float)
        state = state.unsqueeze(0)
        if torch.cuda.is_available():
            state = state.cuda()
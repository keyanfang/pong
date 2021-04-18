import pickle

import gym
import imageio
import numpy as np

data = {
    b'data':[],
    b'lable':[]
}


def mycallback(obs_t, action, rew, done):
    print("action = ", action, " reward = ", rew, "done = ", done)
    imageio.imwrite("jeu.jpg", obs_t)
    imageio.imwrite('outfile.png', obs_t[34:194:4,12:148:2,1])
    outfile=imageio.imread('outfile.png')

    if rew>0:
        data=outfile.astype(np.float).ravel()-obs_t
    # np.savetxt("outfileX.txt", delimiter='', X=[data], fmt='%d')
    # np.savetxt("outfileY.txt", delimiter='', X=[action], fmt='%d')
    data[b'data'].append(data)
    data[b'lable'].append(action)

def init():
    env = gym.make('Pong-v4')
    env.reset()
    restraint=0
    while True:
        env.render()
        action=env.action_space.sample()
        obs, rew, d, inf = env.step(action)
        #obs_t, rew, d, inf = env.step(env.action_space.sample())
        data = obs.astype(np.float).ravel() - obs
        if rew>0:
            data[b'data'].append(data)
            data[b'lable'].append(action)
        if d:
            env.reset()
            restraint =restraint+1
        if restraint>20:
            with open('data.txt','ab')as outfileX:
                pickle.dump(data.outfileX)
            outfileX.close()
            break
        env.close()
import gym
from gym.utils.play import play
import imageio
import numpy as np

def mycallback(obs_t, obs_tp1, action, rew, done, info):
    print("action = ", action, " reward = ", rew, "done = ", done)
    imageio.imwrite("jeu.jpg", obs_t)
    jeu = imageio.imread("jeu.jpg")
    # print(jeu.shape)
    imageio.imwrite('outfile.png', obs_t[34:194:4, 12:148:2, 1])
    outfile = imageio.imread("outfile.png")
    # print(outfile.shape)
    np.savetxt("outfileX.txt", delimiter='', X=obs_t[34:194:4, 12:148:2, 1], fmt='%d')

    
if __name__ == "__main__":
    env = gym.make('Pong-v4')
    env.reset()
    # while True:
    #     env.render()
    #     obs, rew, d, inf = env.step(env.action_space.sample())  # take a random action
    #     if rew != 0:
    #         print("reward: ", rew)
    gym.utils.play.play(env, zoom=3, fps=12, callback=mycallback)
    env.close()

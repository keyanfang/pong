import gym
from gym.utils.play import play
import imageio
import _pickle as pickle
import numpy as np

D=80*80
neuron=100
model = {}
model['1'] = np.random.randn(neuron, D) / np.sqrt(D)
model['2'] = np.random.randn(neuron) / np.sqrt(neuron)

def prepro(obs_t):
    obs_t = obs_t[35:195]
    obs_t = obs_t[::2, ::2, 0]
    obs_t[obs_t == 144] = 0
    obs_t[obs_t == 109] = 0
    obs_t[obs_t != 0] = 1  #
    return obs_t.astype(np.float).ravel()

def policy_forward(x):
    h = np.dot(model['1'], x)
    h[h < 0] = 0
    logp = np.dot(model['2'], h)
    p = 1.0 / (1.0 + np.exp(logp))
    return p, h

def policy_backward(eph, epdlogp):

    dW2 = np.dot(eph.T, epdlogp).ravel()
    dh = np.outer(epdlogp, model['2'])
    dh[eph <= 0] = 0
    dW1 = np.dot(dh.T, epx)
    return {'W1': dW1, 'W2': dW2}

def mycallback(obs_t, action, rew, done):
    print("action = ", action, " reward = ", rew, "done = ", done)
    imageio.imwrite("jeu.jpg", obs_t)
    jeu = imageio.imread("jeu.jpg")
    imageio.imwrite('outfile.png', obs_t[34:194:4,12:148:2,1])
    obs_tp1=obs_t[34:194:4, 12:148:2, 1]
    np.savetxt("outfileX.txt", delimiter='', X=obs_t[34:194:4, 12:148:2, 1], fmt='%d')
    np.savetxt("outfileY.txt", delimiter='', X=[action], fmt='%d')

grad_buffer = {k: np.zeros_like(v) for k, v in model.items()}  # update buffers that add up gradients over a batch
rmsprop_cache = {k: np.zeros_like(v) for k, v in model.items()}  # rmsprop memory
    
if __name__ == "__main__":
    env = gym.make('Pong-v4')
    obs=env.reset()
    prev_x = None  # used in computing the difference frame
    xs, hs, dlogps, drs = [], [], [], []
    running_reward = None
    reward_sum = 0
    episode_number = 0
    begin=True
    #gym.utils.play.play(env, zoom=3, fps=12, callback=mycallback)
    # while True:
    #     env.render()
    #     obs, rew, d, inf = env.step(env.action_space.sample())
    #     if rew != 0:
    #         print("reward: ", rew)

    while begin:
        env.render()
        cur_x = prepro(obs)
        x = cur_x - prev_x if prev_x is not None else np.zeros(D)
        prev_x = cur_x

        # forward the policy network and sample an action from the returned probability
        aprob, h = policy_forward(x)
        action = 2 if np.random.uniform() < aprob else 3  # roll the dice!

        # record various intermediates (needed later for backprop)
        xs.append(x)  # observation
        hs.append(h)  # hidden state
        y = 1 if action == 2 else 0  # a "fake label"
        dlogps.append(
            y - aprob)  # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)

        # step the environment and get new measurements
        observation, rew, done, info = env.step(action)

        mycallback(observation,action,rew,done)
        reward_sum += rew

        drs.append(rew)  # record reward (has to be done after we call step() to get reward for previous action)

        if done:  # an episode finished
            episode_number += 1

            # stack together all inputs, hidden states, action gradients, and rewards for this episode
            epx = np.vstack(xs)
            eph = np.vstack(hs)
            epdlogp = np.vstack(dlogps)
            epr = np.vstack(drs)
            xs, hs, dlogps, drs = [], [], [], []  # reset array memory

            # compute the discounted reward backwards through time
            discounted_epr = epr
            # standardize the rewards to be unit normal (helps control the gradient estimator variance)
            discounted_epr -= np.mean(discounted_epr)
            discounted_epr /= np.std(discounted_epr)

            # epdlogp *= discounted_epr  # modulate the gradient with advantage (PG magic happens right here.)
            # grad = policy_backward(eph, epdlogp)
            # for k in model: grad_buffer[k] += grad[k]  # accumulate grad over batch



            # boring book-keeping
            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            print
            'resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward)
            if episode_number % 100 == 0: pickle.dump(model, open('save.p', 'wb'))
            reward_sum = 0
            observation = env.reset()  # reset env
            prev_x = None

        if rew != 0:  # Pong has either +1 or -1 reward exactly when game ends.
            print('ep %d: game finished, reward: %f' % (episode_number, rew))

        if episode_number==20:
            begin=False

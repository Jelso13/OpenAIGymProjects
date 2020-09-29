import gym
import numpy as np
import random

"""
Q-Learning Formula:
Q_new = (1 - LEARNING_RATE) * Q_current + LEARNING_RATE * (reward + DISCOUNT * MAX_Q_FUTURE)
LEARNING_RATE   - determines how much new information writes over old information
                    0 learns nothing new
                    1 only considers newest information
DISCOUNT        - determines how important future rewards are compared to short terms rewards
MAX_Q_FUTURE    - estimate of the best future value
"""

# Environment observations given by
#   (cart position, cart velocity, angle of pole, angular velocity of pole)
# attempt 1 was incredibly slow so best to ignore cart values for speed
# env is declared with .env so that the 200 step limit is false
env = gym.make("CartPole-v0").env
BUCKETS = [10, 10]
NUM_ACTIONS = env.action_space.n # (left, right)
# set the bounds for the pole values
#   pole angle obtained from environment min max values
#   velocity given by radians (+-1 radians per timestep ~ +-60 degrees per timestep)
ENV_TEMPLATE = [[env.observation_space.low[2], env.observation_space.high[2]], [-1, 1]]
EXPLORE_RATE = 1.0
LEARNING_RATE = 1.0
DISCOUNT = 0.95
# create qTable with zeros for each env parameter being used in combination with each possible action
QTable = np.zeros(BUCKETS + [NUM_ACTIONS])

def toDiscreteEnv(observation):
    obs = observation[2:]
    buckets = []
    bucketIndex = 0 # if the value is less than the bottom bound for the buckets
    for i in range(len(obs)):
        if obs[i] >= ENV_TEMPLATE[i][1]: # if more than upper bound assign top bucket
            bucketIndex = BUCKETS[i] - 1
        elif obs[i] > ENV_TEMPLATE[i][0]: # if inside the bounds of the top and bottom bucket
            ratio_position = obs[i]*((BUCKETS[i]-1)/(ENV_TEMPLATE[i][1]-ENV_TEMPLATE[i][0]))
            bucketIndex = int(round(ratio_position - (((BUCKETS[i]-1)*ENV_TEMPLATE[i][0])/(ENV_TEMPLATE[i][1] - ENV_TEMPLATE[i][0]))))
        buckets.append(bucketIndex)
    return tuple(buckets)

def selectAction(state, EXPLORE_RATE):
    if random.random() < EXPLORE_RATE: # choose a random action for every EXPLORE_RATE actions
        return random.randint(0, 1)
    else: # else choose highest q value (the greedy optoin)
        return np.argmax(QTable[state])

debugOn = True
run_number = 0
best_timesteps = 0
while True:
    run_number += 1
    #reset env
    obs= env.reset()
    #initialise the total reward counter for each run
    totalReward = 0
    #get initial state
    discreteState = toDiscreteEnv(obs)
    done = False
    timeSteps = 0
    # decay the rate at which random exploration is made from 1.0 to 0.01
    EXPLORE_RATE = max(1/(1+run_number/10), 0.001)
    # decay the learning rate from 1.0 to 0.1 for each run
    LEARNING_RATE = max(1/(1+run_number/100),0.1)
    # while the pole hasnt fallen or 199 timesteps reached (determined by gym)
    while not done:
        timeSteps+=1
        if run_number % 10 == 0: # show the result every 50 runs
            env.render()
        #select action
        action = selectAction(discreteState, EXPLORE_RATE)
        observation, reward, done, info = env.step(action)
        newDiscreteState = toDiscreteEnv(observation)
        topPredictQ = np.max(QTable[newDiscreteState])
        QTable[discreteState][action] = (1-LEARNING_RATE) * QTable[discreteState][action] + LEARNING_RATE * (reward + DISCOUNT * topPredictQ)
        discreteState = newDiscreteState
        totalReward += reward
        if done:
            # if its not done because it failed but because it passed 199 timesteps reset done
            if 'TimeLimit.truncated' in info:
                done = False
            if timeSteps > best_timesteps:
                best_timesteps = timeSteps
            print("Run("+str(run_number)+"):\t"+str(timeSteps)+" timesteps",end="")
            if debugOn: # show more information about the run
                print("\texplore rate = ", round(EXPLORE_RATE,3),end="")
                print("\tlearning rate = ", round(LEARNING_RATE,3), end = "")
                print("\tbest timestep = ", best_timesteps, end = "")
            # if the run would be classified as good enough to finish by openai print *
            if timeSteps > 199:
                print("\t*")
            else:
                print()


import gym
from numpy import random
import numpy as np
from IPython.display import clear_output
from time import sleep

random.seed(1234)

streets = gym.make("Taxi-v3").env
streets.render()
"""
+---------+
|R: | : :G|
| : | : : |
| : : : : |
| | : | : |
|Y| : |B: |
+---------+
Let's break down what we're seeing here:

R, G, B, and Y are pickup or dropoff locations.
The BLUE letter indicates where we need to pick someone up from.
The MAGENTA letter indicates where that passenger wants to go to.
The solid lines represent walls that the taxi cannot cross.
The filled rectangle represents the taxi itself - it's yellow when empty, and green when carrying a passenger.

Our little world here, which we've called "streets", is a 5x5 grid. The state of this world at any time can be defined by:

Where the taxi is (one of 5x5 = 25 locations)
What the current destination is (4 possibilities)
Where the passenger is (5 possibilities: at one of the destinations, or inside the taxi)
So there are a total of 25 x 4 x 5 = 500 possible states that describe our world.

For each state, there are six possible actions:

Move South, East, North, or West
Pickup a passenger
Drop off a passenger
Q-Learning will take place using the following rewards and penalties at each state:

A successfull drop-off yields +20 points
Every time step taken while driving a passenger yields a -1 point penalty
Picking up or dropping off at an illegal location yields a -10 point penalty
"""
#Lets consider initial state wherein the taxi os at 2,3 and passenger at location 2 and destination at 0
initialState=streets.encode(2,3,2,0)#position of taxi (2,3) location of pass. at 2 and desti. at 0
streets.s=initialState#streets.s is street state s is for state
streets.render()

print(streets.P[initialState])#So what happenes here is that it gives us dets about all the movements possible meaning if we take astep in north or south or west or east and then it will tell us how good it is or how bad it is
#eg:So for example, moving North from this state would put us into state number 368, incur a penalty of -1 for taking up time, and does not result in a successful dropoff


#Training the model now

#these are the required attributes of the taxi game for q-learning
qTables=np.zeros([streets.observation_space.n,streets.action_space.n])#assigning the values initially to zero north,south,west,east,drop,pick
learningRate=0.1#these are hyper parameters just like the ones in XGBoost
discountFactor=0.6#this factor is for the q-learning algo
exploration=0.1#this param gives us 10% chance to explore a new random route for learning
epochs=10000

for taxi_run in range(epochs):
    state=streets.reset()
    done=False

    while not done:
        random_value=random.uniform(0,1)#creating a uniform distribution
        if(random_value<exploration):
            # Explore a random action,only 10% chance of this event to happen
            action=streets.action_space.sample()
        else:
            # Use the action with the highest q-value

            action=np.argmax(qTables[state])#return max argument at the given instance simply put it return some max value

            #as we know we have q values given for each action so in the table the highest q value action will be repeated

        nextState,reward,done,info=streets.step(action) #this basically gives us the analysis at each step eg:0: [(1.0, 368, -1, False)]

        prev_q=qTables[state,action]
        next_max_q=np.max(qTables[nextState])
        new_q=(1-learningRate)*prev_q+learningRate*(reward+discountFactor*next_max_q)
        qTables[state,action]=new_q

        state=nextState

# print(qTables[initialState])

#now we will give ago for our model
for tripnum in range(1,11):
    state=streets.reset()

    done=False

    while not done:
        action=np.argmax(qTables[state])
        nextState,reward,done,info=streets.step(action)
        clear_output(wait=True)
        print("Trip number"+str(tripnum))
        print(streets.render(mode='ansi'))
        sleep(.5)
        state=nextState
    sleep(2)


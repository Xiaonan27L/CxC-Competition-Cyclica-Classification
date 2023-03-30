# import numpy as np

# np.random.seed(486)

# from GridWorld import GridWorld
# from Agent import Agent
# from utils import *


# try to commit
# test
# test 2

import pandas as pd
import matplotlib as plt
# np.set_printoptions(suppress=True)

# if __name__ == "__main__":
#     np.random.seed(486)

#     rewards = np.array([
#         [-0.04, -0.04, -1.00],
#         [-0.04, -0.04, 1.000]
#     ])

#     start_state = (0, 0)
#     goal_states = [(0, 2), (1, 2)]
#     walls = [(1, 0)]

#     env = GridWorld(world_height=2, world_width=3,
#                     prob_direct=0.8, prob_lateral=0.1,
#                     rewards=rewards,
#                     start_state=start_state,
#                     goal_states=goal_states,
#                     walls=walls)

#     agent = Agent(env)
#     # cur_state, callback_fn = env.simulate()
#     # # print(cur_state)
#     # # print(callback_fn)
#     # print(env.make_move((0, 0), Action.DOWN))

#     # print(agent.env.state_dim)

#     # print(env.T)
#     env.fill_T()
#     # print(env.T)
#     # print(env.T[1,1,0,1,1])
#     # print(agent.value_iteration(0.99))

# #     V = np.array([[ 0.57605613,  0.63413643, -1.        ],
# #  [ 0.,          0.90428108,  1.        ]])
#     # print(V)
#     # print(agent.find_policy(V))
#     print("aa")
#     pi = np.array([[1, 2, 2], [1, 1, 2]])
#     print(pi)
#     print(agent.passive_adp(pi, 0.99)[0])
#     # print(env.T)
#     # print(np.zeros((env.state_dim + env.state_dim)))
#     # print((env.action_dim))
#     # print(t + env.state_dim + tuple_action)

#     # print(t + env.state_dim + tuple(env.action_dim) + env.state_dim)

import numpy as np

from GridWorld import GridWorld
from Agent import Agent

np.set_printoptions(suppress=True)

if __name__ == "__main__":
    np.random.seed(486)

    rewards = np.array([
        [-0.04, -0.04, -0.04, -0.04],
        [-0.04, -0.04, -0.04, -1],
        [-0.04, -0.04, -0.04, 1]
    ])

    start_state = (0, 0)
    goal_states = [(2, 3), (1, 3)]
    walls = [(1, 1), (0, 2)]

    env = GridWorld(world_height=3, world_width=4,
                    prob_direct=0.8, prob_lateral=0.1,
                    rewards=rewards,
                    start_state=start_state,
                    goal_states=goal_states,
                    walls=walls)

    agent = Agent(env)
    # env.fill_T()
    
    V, i = agent.value_iteration(0.99)
    print(agent.value_iteration(0.99))
    pi = agent.find_policy(V)
    print(pi)
    print(agent.view_policy(pi))
    # path = agent.get_path(pi, (0, 0), [(2, 3)])
    # print(path)
    # print(agent.get_path(pi, (0, 0), [(1, 3)]))
    # print(V)
    # total = 0
    # V_LIST =  [0.65066308,0.71663212,0.77618555,0.84393511, 0.9050959,1]
    # for i in range(len(V_LIST)):
    #     total += 0.99 * V_LIST[i]
    # print(total)
    # import matplotlib.pyplot as plt
    print(agent.passive_adp(pi, 0.99)[0])
    total = 0
    # lst_of_states = agent.passive_adp(pi, 0.99, adp_iters=2000)[1]
    # state_00 = list()
    # state_03 = list()
    # state_22 = list()
    # state_12 = list()
    # for i in range(2000):
    #     array_i = lst_of_states[i]
    #     state_00.append(array_i[(0,0)])
    #     state_03.append(array_i[(0,3)])
    #     state_22.append(array_i[(2,2)])
    #     state_12.append(array_i[(1,2)])
    # fig, ax = plt.subplots()
    # x = list(range(1, 2000+1))
    # # Plot the data on the axis object
    # ax.plot(x, state_00, label='(0, 0)')
    # ax.plot(x, state_03, label='(0, 3)')
    # ax.plot(x, state_22, label='(2, 2)')
    # ax.plot(x, state_12, label='(1, 2)')

    # # Set the axis labels and legend
    # ax.set_xlabel('Iteration')
    # ax.set_ylabel('State Value')
    # ax.legend()

    # # Show the plot
    # plt.show()




    

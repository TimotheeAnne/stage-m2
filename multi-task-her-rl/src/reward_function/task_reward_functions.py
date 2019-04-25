import numpy as np

instructions = ['Move the gripper to the left', #0
                'Move the gripper to the right',  #1
                'Move the gripper further', #2
                'Move the gripper closer', #3
                'Take the object 1', #4
]


# for i in range(len(instructions)):
#     print('Instruction', str(i), ':', instructions[i])

epsilon = 1

def eucl_dist(pos1, pos2):
    return np.linalg.norm(pos1 - pos2, ord=2)


def r0(obs):
    # Move the gripper to left
    if obs[5] >  2 * epsilon:
        return 0
    else:
        return -1


def r1(obs):
    # Move the gripper to right
    if obs[5] < - 2 * epsilon:
        return 0
    else:
        return -1


def r2(obs):
    # Move the gripper away
    if obs[6] > 2 * epsilon:
        return 0
    else:
        return -1


def r3(obs):
    # Move the gripper closer
    if obs[6] < - 2 * epsilon:
        return 0
    else:
        return -1


def r4(obs):
    # Take the object 1
    if obs[4]:
        return 0
    else:
        return -1

task_instructions = [r0, r1, r2, r3, r4]

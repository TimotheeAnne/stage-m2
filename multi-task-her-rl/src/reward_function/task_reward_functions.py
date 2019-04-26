import numpy as np

"""
observation = 
0 arm_angles[0]
1 arm_angles[1]
2 arm_angles[2]
3 hand_pos[0], 
4 hand_pos[1],
5 gripper,
6 stick1_handle_pos[0],
7 stick1_handle_pos[1],
8 stick1_end_pos[0],
9 stick1_end_pos[1],
10 stick2_handle_pos[0],
11 stick2_handle_pos[1],
12 stick2_end_pos[0],
13 stick2_end_pos[1],
14 magnet1_pos[0],
15 magnet1_pos[1],
16 scratch1_pos[0],
17 scratch1_pos[1],
"""          

instructions = ['Move the Hand to the left', #0
                'Move the Hand to the right',  #1
                'Move the Hand further', #2
                'Move the Hand closer',  #3
                
                'Grasp Stick1', #4
                'Grasp Stick2', #5
                
                'Move the Stick1 to the left', #6
                'Move the Stick1 to the right',  #7
                'Move the Stick1 further', #8
                'Move the Stick1 closer', #9

                'Move the Stick2 to the left', #10
                'Move the Stick2 to the right', #11
                'Move the Stick2 further', #12
                'Move the Stick2 closer', #13

                'Move the Magnet1 to the left', #14
                'Move the Magnet1 to the right', #15
                'Move the Magnet1 further', #16
                'Move the Magnet1 closer', #17

                'Move the Scratch1 to the left', #18
                'Move the Scratch1 to the right', #19
                'Move the Scratch1 further', #20
                'Move the Scratch1 closer', #21

]


# for i in range(len(instructions)):
#     print('Instruction', str(i), ':', instructions[i])

epsilon = 0.05


def eucl_dist(pos1, pos2):
    return np.linalg.norm(pos1 - pos2, ord=2)

""" Hand Pos 0-1"""
def r0(obs,d_obs):
    # Move the Hand to right
    if d_obs[3] >  2 * epsilon:
        return 0
    else:
        return -1


def r1(obs,d_obs):
    # Move the Hand to left
    if d_obs[3] < - 2 * epsilon:
        return 0
    else:
        return -1

def r2(obs,d_obs):
    # Move the Hand further
    if d_obs[4] > 2 * epsilon:
        return 0
    else:
        return -1

def r3(obs,d_obs):
    # Move the Hand closer
    if d_obs[4] < - 2 * epsilon:
        return 0
    else:
        return -1
        
""" Grasped stick1 """
def r4(obs,d_obs):
    if np.linalg.norm(d_obs[6:8]) > 0 :
        return 0
    else:
        return -1

""" Grasped stick2 """
def r5(obs,d_obs):
    if np.linalg.norm(d_obs[10:12]) > 0 :
        return 0
    else:
        return -1

""" Stick1 end pos 6-9"""
def r6(obs,d_obs):
    # Move stick1 to the right
    if d_obs[8] >  2 * epsilon:
        return 0
    else:
        return -1


def r7(obs,d_obs):
    # Move stick1 to the left
    if d_obs[8] < - 2 * epsilon:
        return 0
    else:
        return -1


def r8(obs,d_obs):
    # Move the stick1 away
    if d_obs[9] > 2 * epsilon:
        return 0
    else:
        return -1


def r9(obs,d_obs):
    # Move the stick1 closer
    if d_obs[9] < - 2 * epsilon:
        return 0
    else:
        return -1
        
""" Stick2 end pos 10-13"""
def r10(obs,d_obs):
    # Move stick2 to right
    if d_obs[12] >  2 * epsilon:
        return 0
    else:
        return -1


def r11(obs,d_obs):
    # Move stick2 to left
    if d_obs[12] < - 2 * epsilon:
        return 0
    else:
        return -1


def r12(obs,d_obs):
    # Move stick2 away
    if d_obs[13] > 2 * epsilon:
        return 0
    else:
        return -1


def r13(obs,d_obs):
    # Move stick2 closer
    if d_obs[13] < - 2 * epsilon:
        return 0
    else:
        return -1

        
""" Magnet1 pos 14-17"""
def r14(obs,d_obs):
    # Move the gripper to right
    if d_obs[14] >  2 * epsilon:
        return 0
    else:
        return -1


def r15(obs,d_obs):
    # Move the gripper to left
    if d_obs[14] < - 2 * epsilon:
        return 0
    else:
        return -1


def r16(obs,d_obs):
    # Move the gripper away
    if d_obs[15] > 2 * epsilon:
        return 0
    else:
        return -1


def r17(obs,d_obs):
    # Move the gripper closer
    if d_obs[15] < - 2 * epsilon:
        return 0
    else:
        return -1

""" Magnet2 pos 18-21"""
def r18(obs,d_obs):
    # Move the gripper to right
    if d_obs[16] >  2 * epsilon:
        return 0
    else:
        return -1


def r19(obs,d_obs):
    # Move the gripper to the left
    if d_obs[16] < - 2 * epsilon:
        return 0
    else:
        return -1


def r20(obs,d_obs):
    # Move the gripper away
    if d_obs[17] > 2 * epsilon:
        return 0
    else:
        return -1


def r21(obs,d_obs):
    # Move the gripper closer
    if d_obs[17] < - 2 * epsilon:
        return 0
    else:
        return -1

task_instructions = [r0, r1, r2, r3, r4, r5, r6, r7, r8, r9,
                     r10, r11, r12, r13, r14, r15, r16, r17, r18, r19,
                     r20, r21,
                     ]

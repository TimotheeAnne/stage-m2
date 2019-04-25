import numpy as np

"""
0 hand_pos[0], 
1 hand_pos[1],
2 gripper,
3 stick1_end_pos[0],
4 stick1_end_pos[1],
5 stick2_end_pos[0],
6 stick2_end_pos[1],
7 magnet1_pos[0],
8 magnet1_pos[1],
9 magnet2_pos[0],
10 magnet2_pos[1],
11 magnet3_pos[0],
12 magnet3_pos[1],
13 scratch1_pos[0],
14 scratch1_pos[1],
15 scratch2_pos[0],
16 scratch2_pos[1],
17 scratch3_pos[0],
18 scratch3_pos[1],
19 cat_pos[0],
20 cat_pos[1],
21 dog_pos[0],
22 dog_pos[1],
23 static_objects_rest_state[0][0],
24 static_objects_rest_state[0][1],
25 static_objects_rest_state[1][0],
26 static_objects_rest_state[1][1],
27 static_objects_rest_state[2][0],
28 static_objects_rest_state[2][1],
29 static_objects_rest_state[3][0],
30 static_objects_rest_state[3][1]
"""          

instructions = ['Move the Hand to the left', #0
                'Move the Hand to the right',  #1
                'Grasp Stick1', #2
                'Grasp Stick2', #3
                
                'Move the Stick1 to the left', #4
                'Move the Stick1 to the right',  #5
                'Move the Stick1 further', #6
                'Move the Stick1 closer', #7

                'Move the Stick2 to the left', #8
                'Move the Stick2 to the right',  #9
                'Move the Stick2 further', #10
                'Move the Stick2 closer', #11

                'Move the Magnet1 to the left', #12
                'Move the Magnet1 to the right',  #13
                'Move the Magnet1 further', #14
                'Move the Magnet1 closer', #15

                'Move the Magnet2 to the left', #16
                'Move the Magnet2 to the right',  #17
                'Move the Magnet2 further', #18
                'Move the Magnet2 closer', #19

                'Move the Magnet3 to the left', #20
                'Move the Magnet3 to the right',  #21
                'Move the Magnet3 further', #22
                'Move the Magnet3 closer', #23

                'Move the Scratch1 to the left', #24
                'Move the Scratch1 to the right',  #25
                'Move the Scratch1 further', #26
                'Move the Scratch1 closer', #27

                'Move the Scratch2 to the left', #28
                'Move the Scratch2 to the right',  #29
                'Move the Scratch2 further', #30
                'Move the Scratch2 closer', #31

                'Move the Scratch3 to the left', #32
                'Move the Scratch3 to the right',  #33
                'Move the Scratch3 further', #34
                'Move the Scratch3 closer', #35

]


# for i in range(len(instructions)):
#     print('Instruction', str(i), ':', instructions[i])

epsilon = 0.05


def eucl_dist(pos1, pos2):
    return np.linalg.norm(pos1 - pos2, ord=2)

""" Hand Pos 0-1"""
def r0(obs,d_obs):
    # Move the gripper to left
    if d_obs[0] >  2 * epsilon:
        return 0
    else:
        return -1


def r1(obs,d_obs):
    # Move the gripper to right
    if d_obs[0] < - 2 * epsilon:
        return 0
    else:
        return -1

""" Grasp stick1 """
def r2(obs,d_obs):
    if np.linalg.norm(d_obs[3:5]) > 0 :
        return 0
    else:
        return -1

""" Grasp stick2 """
def r3(obs,d_obs):
    if np.linalg.norm(d_obs[5:7]) > 0 :
        return 0
    else:
        return -1

""" Stick1 end pos 4-7"""
def r4(obs,d_obs):
    if d_obs[3] >  2 * epsilon:
        return 0
    else:
        return -1


def r5(obs,d_obs):
    # Move the gripper to right
    if d_obs[3] < - 2 * epsilon:
        return 0
    else:
        return -1


def r6(obs,d_obs):
    # Move the gripper away
    if d_obs[4] > 2 * epsilon:
        return 0
    else:
        return -1


def r7(obs,d_obs):
    # Move the gripper closer
    if d_obs[4] < - 2 * epsilon:
        return 0
    else:
        return -1
        
""" Stick2 end pos 8-11"""
def r8(obs,d_obs):
    # Move the gripper to left
    if d_obs[5] >  2 * epsilon:
        return 0
    else:
        return -1


def r9(obs,d_obs):
    # Move the gripper to right
    if d_obs[5] < - 2 * epsilon:
        return 0
    else:
        return -1


def r10(obs,d_obs):
    # Move the gripper away
    if d_obs[6] > 2 * epsilon:
        return 0
    else:
        return -1


def r11(obs,d_obs):
    # Move the gripper closer
    if d_obs[6] < - 2 * epsilon:
        return 0
    else:
        return -1

        
""" Magnet1 pos 12-15"""
def r12(obs,d_obs):
    # Move the gripper to left
    if d_obs[7] >  2 * epsilon:
        return 0
    else:
        return -1


def r13(obs,d_obs):
    # Move the gripper to right
    if d_obs[7] < - 2 * epsilon:
        return 0
    else:
        return -1


def r14(obs,d_obs):
    # Move the gripper away
    if d_obs[8] > 2 * epsilon:
        return 0
    else:
        return -1


def r15(obs,d_obs):
    # Move the gripper closer
    if d_obs[8] < - 2 * epsilon:
        return 0
    else:
        return -1

""" Magnet2 pos 16-19"""
def r16(obs,d_obs):
    # Move the gripper to left
    if d_obs[9] >  2 * epsilon:
        return 0
    else:
        return -1


def r17(obs,d_obs):
    # Move the gripper to right
    if d_obs[9] < - 2 * epsilon:
        return 0
    else:
        return -1


def r18(obs,d_obs):
    # Move the gripper away
    if d_obs[10] > 2 * epsilon:
        return 0
    else:
        return -1


def r19(obs,d_obs):
    # Move the gripper closer
    if d_obs[10] < - 2 * epsilon:
        return 0
    else:
        return -1

""" Magnet3 pos 20-23"""
def r20(obs,d_obs):
    # Move the gripper to left
    if d_obs[11] >  2 * epsilon:
        return 0
    else:
        return -1


def r21(obs,d_obs):
    # Move the gripper to right
    if d_obs[11] < - 2 * epsilon:
        return 0
    else:
        return -1


def r22(obs,d_obs):
    # Move the gripper away
    if d_obs[12] > 2 * epsilon:
        return 0
    else:
        return -1


def r23(obs,d_obs):
    # Move the gripper closer
    if d_obs[12] < - 2 * epsilon:
        return 0
    else:
        return -1

""" Scratch 1 pos 24-27"""
def r24(obs,d_obs):
    # Move the gripper to left
    if d_obs[13] >  2 * epsilon:
        return 0
    else:
        return -1


def r25(obs,d_obs):
    # Move the gripper to right
    if d_obs[13] < - 2 * epsilon:
        return 0
    else:
        return -1


def r26(obs,d_obs):
    # Move the gripper away
    if d_obs[14] > 2 * epsilon:
        return 0
    else:
        return -1


def r27(obs,d_obs):
    # Move the gripper closer
    if d_obs[14] < - 2 * epsilon:
        return 0
    else:
        return -1
        
""" Scratch 2 pos 28-31"""
def r28(obs,d_obs):
    # Move the gripper to left
    if d_obs[15] >  2 * epsilon:
        return 0
    else:
        return -1


def r29(obs,d_obs):
    # Move the gripper to right
    if d_obs[15] < - 2 * epsilon:
        return 0
    else:
        return -1


def r30(obs,d_obs):
    # Move the gripper away
    if d_obs[16] > 2 * epsilon:
        return 0
    else:
        return -1


def r31(obs,d_obs):
    # Move the gripper closer
    if d_obs[16] < - 2 * epsilon:
        return 0
    else:
        return -1
        
""" Scratch 3 pos 32-35"""
def r32(obs,d_obs):
    # Move the gripper to left
    if d_obs[17] >  2 * epsilon:
        return 0
    else:
        return -1


def r33(obs,d_obs):
    # Move the gripper to right
    if d_obs[17] < - 2 * epsilon:
        return 0
    else:
        return -1


def r34(obs,d_obs):
    # Move the gripper away
    if d_obs[18] > 2 * epsilon:
        return 0
    else:
        return -1


def r35(obs,d_obs):
    # Move the gripper closer
    if d_obs[18] < - 2 * epsilon:
        return 0
    else:
        return -1
task_instructions = [r0, r1, r2, r3, r4, r5, r6, r7, r8, r9,
                     r10, r11, r12, r13, r14, r15, r16, r17, r18, r19,
                     r20, r21, r22, r23, r24, r25, r26, r27, r28, r29,
                     r30, r31, r32, r33, r34, r35]

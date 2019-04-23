import numpy as np

instructions = ['Move the gripper to the left',
                'Move the gripper to the right', 
                'Move the gripper further', 
                'Move the gripper closer',
                'Move the gripper higher',
                'Move the gripper lower',
                'Move the gripper above the yellow object',
                'Touch the yellow object from above', 
                'Throw the yellow object on the floor',
                'Move the yellow object to the left',
                'Move the yellow object to the right',
                'Move the yellow object away',
                'Move the yellow object closer',
                'Lift the yellow object',
                'Lift the yellow object higher',
                'Lift the yellow object and put it on the left', 
                'Lift the yellow object and put it on the right',
                'Lift the yellow object and place it further', 'Lift the yellow object and place it closer']


# for i in range(len(instructions)):
#     print('Instruction', str(i), ':', instructions[i])

epsilon = 0.05

def eucl_dist(pos1, pos2):
    return np.linalg.norm(pos1 - pos2, ord=2)


def r0(gripper_pos, yellow_cube_pos,  gripper_pos_0, yellow_cube_pos_0):
    # Move the gripper to left
    if gripper_pos[1] > gripper_pos_0[1] + 2 * epsilon:
        return 0
    else:
        return -1


def r1(gripper_pos, yellow_cube_pos,  gripper_pos_0, yellow_cube_pos_0):
    # Move the gripper to right
    if gripper_pos[1] < gripper_pos_0[1] - 2 * epsilon:
        return 0
    else:
        return -1


def r2(gripper_pos, yellow_cube_pos,  gripper_pos_0, yellow_cube_pos_0):
    # Move the gripper away
    if gripper_pos[0] > gripper_pos_0[0] + 2 * epsilon:
        return 0
    else:
        return -1


def r3(gripper_pos, yellow_cube_pos,  gripper_pos_0, yellow_cube_pos_0):
    # Move the gripper closer
    if gripper_pos[0] < gripper_pos_0[0] - 2 * epsilon:
        return 0
    else:
        return -1


def r4(gripper_pos, yellow_cube_pos,  gripper_pos_0, yellow_cube_pos_0):
    # Move the gripper higher
    if gripper_pos[2] > gripper_pos_0[2] + 2 * epsilon:
        return 0
    else:
        return -1


def r5(gripper_pos, yellow_cube_pos,  gripper_pos_0, yellow_cube_pos_0):
    # Move the gripper lower
    if gripper_pos[2] < gripper_pos_0[2] - 2 * epsilon:
        return 0
    else:
        return -1


def r6(gripper_pos, yellow_cube_pos,  gripper_pos_0, yellow_cube_pos_0):
    # Move the gripper above the yellow object
    if eucl_dist(gripper_pos[:2], yellow_cube_pos[:2]) < epsilon:
        return 0
    else:
        return -1

def r7(gripper_pos, yellow_cube_pos,  gripper_pos_0, yellow_cube_pos_0):
    # Touch the yellow object from above
    if eucl_dist(gripper_pos[:2], yellow_cube_pos[:2]) < epsilon and gripper_pos[2] - yellow_cube_pos[2] < epsilon:
        return 0
    else:
        return -1


def r8(gripper_pos, yellow_cube_pos,  gripper_pos_0, yellow_cube_pos_0):
    # Throw the yellow object on the floor
    if np.any(np.abs(yellow_cube_pos - yellow_cube_pos_0) > 2 * epsilon) and yellow_cube_pos[2] < 0.41:
        return 0
    else:
        return -1


def r9(gripper_pos, yellow_cube_pos,  gripper_pos_0, yellow_cube_pos_0):
    # Move the yellow object to left
    if yellow_cube_pos[1] > yellow_cube_pos_0[1] + 2 * epsilon and yellow_cube_pos[2] > 0.41:
        return 0
    else:
        return -1


def r10(gripper_pos, yellow_cube_pos,  gripper_pos_0, yellow_cube_pos_0):
    # Move the yellow object to right
    if yellow_cube_pos[1] < yellow_cube_pos_0[1] - 2 * epsilon and yellow_cube_pos[2] > 0.41:
        return 0
    else:
        return -1


def r11(gripper_pos, yellow_cube_pos,  gripper_pos_0, yellow_cube_pos_0):
    # Move the yellow object away
    if yellow_cube_pos[0] > yellow_cube_pos_0[0] + 2 * epsilon and yellow_cube_pos[2] > 0.41:
        return 0
    else:
        return -1


def r12(gripper_pos, yellow_cube_pos,  gripper_pos_0, yellow_cube_pos_0):
    # Move the yellow object closer
    if yellow_cube_pos[0] <= yellow_cube_pos_0[0] - 2 * epsilon and yellow_cube_pos[2] > 0.41:
        return 0
    else:
        return -1



def r13(gripper_pos, yellow_cube_pos,  gripper_pos_0, yellow_cube_pos_0):
    # Lift the yellow cube
    if yellow_cube_pos[2] > yellow_cube_pos_0[2] + 0.25 * epsilon and eucl_dist(gripper_pos, yellow_cube_pos) < 1.2 * epsilon:
        return 0
    else:
        return -1

def r14(gripper_pos, yellow_cube_pos,  gripper_pos_0, yellow_cube_pos_0):
    # Lift the yellow cube higher
    if yellow_cube_pos[2] > yellow_cube_pos_0[2] + 2 * epsilon and eucl_dist(gripper_pos, yellow_cube_pos) < 1.2 * epsilon:
        return 0
    else:
        return -1

def r15(gripper_pos, yellow_cube_pos,  gripper_pos_0, yellow_cube_pos_0):
    # Lift yellow object and put it on the left
    if yellow_cube_pos[2] > yellow_cube_pos_0[2] + 2 * epsilon and yellow_cube_pos[1] > yellow_cube_pos_0[1] + 0.7 * epsilon and \
            eucl_dist(gripper_pos, yellow_cube_pos) < 1.2 * epsilon:
        return 0
    else:
        return -1


def r16(gripper_pos, yellow_cube_pos,  gripper_pos_0, yellow_cube_pos_0):
    # Lift yellow object and put it on the right
    if yellow_cube_pos[2] > yellow_cube_pos_0[2] + 2 * epsilon and yellow_cube_pos[1] < yellow_cube_pos_0[1] - 0.7 * epsilon and \
            eucl_dist(gripper_pos, yellow_cube_pos) < 1.2 * epsilon:
        return 0
    else:
        return -1


def r17(gripper_pos, yellow_cube_pos,  gripper_pos_0, yellow_cube_pos_0):
    # Lift yellow object and place it further
    if yellow_cube_pos[2] > yellow_cube_pos_0[2] + 2 * epsilon and yellow_cube_pos[0] > yellow_cube_pos_0[0] + 0.7 * epsilon and \
            eucl_dist(gripper_pos, yellow_cube_pos) < 1.2 * epsilon:
        return 0
    else:
        return -1


def r18(gripper_pos, yellow_cube_pos, gripper_pos_0, yellow_cube_pos_0):
    # Lift yellow object and place it closer
    if yellow_cube_pos[2] > yellow_cube_pos_0[2] + 2 * epsilon and yellow_cube_pos[0] < yellow_cube_pos_0[0] - 0.7 * epsilon and \
            eucl_dist(gripper_pos, yellow_cube_pos) < 1.2 * epsilon:
        return 0
    else:
        return -1



task_instructions = [r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13,
                     r14, r15, r16, r17, r18]

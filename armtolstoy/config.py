from utils.gep_utils import Bounds

#ARM CONFIG
ARM_B = Bounds()

state_names =  ['hand_x', 'hand_y', 
                'gripper', 
                'stick1_x', 'stick1_y', 
                'stick2_x', 'stick2_y',
                'magnet1_x', 'magnet1_y', 
                'scratch1_x', 'scratch1_y']

# ~ ARM_B.add('angle0', [-1.,1.])
# ~ ARM_B.add('angle1', [-1.,1.])
# ~ ARM_B.add('angle2', [-1.,1.])
ARM_B.add('hand_x', [-1.,1.])
ARM_B.add('hand_y', [-1.,1.])
ARM_B.add('gripper', [-1.,1.])
# ~ ARM_B.add('stick1_handle_x', [-1.5,1.5])
# ~ ARM_B.add('stick1_handle_y', [-1.5,1.5])
ARM_B.add('stick1_x', [-1.5,1.5])
ARM_B.add('stick1_y', [-1.5,1.5])
# ~ ARM_B.add('stick2_handle_x', [-1.5,1.5])
# ~ ARM_B.add('stick2_handle_y', [-1.5,1.5])
ARM_B.add('stick2_x', [-1.5,1.5])
ARM_B.add('stick2_y', [-1.5,1.5])

ARM_B.add('magnet1_x', [-1.5,1.5])
ARM_B.add('magnet1_y', [-1.5,1.5])

ARM_B.add('scratch1_x',[-1.5,1.5])
ARM_B.add('scratch1_y',[-1.5,1.5])


ARM_OBJECTS = [['hand_x', 'hand_y', 'gripper'],
               ['stick1_x', 'stick1_y'],
               ['stick2_x', 'stick2_y'],
               ['magnet1_x', 'magnet1_y'],
               ['scratch1_x', 'scratch1_y']]
               
ARM_OBJECTS_IDX = [[0,3],[3,5],[5,7],[7,9],[9,11]]



def get_env_bounds(name):
    if name == 'arm_env':
        return ARM_B
    else:
        print('UNKNOWN ENV')

def get_objects(name):
    if name == 'arm_env':
        return ARM_OBJECTS, ARM_OBJECTS_IDX
    else:
        print('UNKNOWN ENV')




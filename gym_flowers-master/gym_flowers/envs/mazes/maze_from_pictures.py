import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import pickle
from PIL import Image
import os

path = '/home/flowers/Downloads/IMG_20181029_111648.jpg'

saving_path = './mazes/mazes_from_pics/'
maze_id = 'maze_pic_0'
assert not os.path.exists(saving_path + maze_id + '_descriptor.txt'), "a maze was already saved under id " + maze_id
scale = 10
image_size = (50 * scale, 60 * scale)
dark_wall = True # set to True if walls are drawn darker than rooms

# convert to gray scale
def rgb2gray(rgb, dark_wall):
    if dark_wall:
        return np.dot(255 - rgb[...,:3], [ 0.299, 0.587, 0.114])
    else:
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

# modify contrast
def change_contrast(img, factor):
    def contrast(pixel):
        return 128 + factor * (pixel - 128)
    return img.point(contrast)

# convert image to gray scale
image = rgb2gray(plt.imread(path)[:,:,:3], dark_wall)

# resize image
image = Image.fromarray(image.astype(np.uint8))
image2 = image.resize((image_size[1], image_size[0]), Image.ANTIALIAS)

# increase contrast
image3 = change_contrast(image2, factor=5.0)
image3.show()

# binarize image with threshold
image3=np.array(image3)
plt.figure()
plt.hist(image3.ravel())
# threshold = image3.ravel().mean()
threshold = 0
grid = np.zeros([image3.shape[0], image3.shape[1]])
inds = np.argwhere(image3>threshold)
for ind in inds:
    grid[ind[0], ind[1]] = 1

# find starting position in top right corner
directions = ['d', 'l', 'u', 'l', 'd', 'r']
n_movements = [1, 1, 1, 1, 2, 2]
i_dir = 0
found = False
x = image_size[1] - scale
y = scale

while not found:
    for i in range(n_movements[i_dir]):
        if directions[i_dir] == 'd':
            y += 1

        elif directions[i_dir] == 'l':
            x -= 1
        elif directions[i_dir] == 'u':
            y -= 1
        elif directions[i_dir] == 'r':
            x += 1
        try:
            if grid[y-scale: y+scale, x-scale: x+scale].sum() == 0:
                agent_x = x - scale
                agent_y = y + scale
                found = True
        except:
            pass

    if i_dir in [1,2,4,5]:
        n_movements[i_dir] += 2
    i_dir = (i_dir + 1) % len(directions)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(grid)
ax.add_patch(Circle((agent_x, agent_y), scale))
params = dict(agent_position = (agent_y, agent_x),
              scale=scale,
              width=image_size[1],
              height=image_size[0],
              maze_id=maze_id)

plt.savefig(saving_path + maze_id + '.png')
np.savetxt(saving_path + maze_id + '_descriptor.txt', grid)
# save params
with open(saving_path + maze_id+'_params.pkl', 'wb') as f:
    pickle.dump(params, f)

plt.show()
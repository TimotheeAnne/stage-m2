# --------------------------------------------------------------------
# An implementation of the "Recursive Division" algorithm.
# The maze descriptor is a numpy array saved in .txt, 1 for wall pixels.
# The corresponding image and the dictionary of parameters are saved in
#  .png and .pkl (pickle) respectively.
# Parameters allow to control the room density, size of walls, doors etc.
# Gym environments can be created from the maze description file (.txt).
# --------------------------------------------------------------------
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

load_and_plot = False
save = True
loading_path = './mazes/maze_0.txt'
saving_path = './mazes/'

if load_and_plot:
    assert os.path.exists(loading_path), "no existing maze file at " + loading_path
    # open maze descriptor and plot
    grid = np.loadtxt(loading_path)
    plt.imshow(grid)
    plt.show()
else:
    # Define parameters
    seed = np.random.randint(0, 1e6)
    maze_id = 'maze'+str(seed)
    scale = 10
    size_door = 3 * scale
    param_stop = 7 * scale # controls number of rooms
    min_width = 3 * scale # control size of rooms
    width = 60 * scale
    height = 50 * scale
    params = dict(scale=scale, size_door=size_door, param_stop=param_stop, width=width, height=height, seed=seed, maze_id=maze_id)


def generate_maze(params):
    out = divide(grid, 1 * scale, 1 * scale, width - 2 * scale, height - 2 * scale, choose_orientation(width, height))
    return out

def choose_orientation(width, height):
  if width < height:
    return 'horizontal'
  elif height < width:
    return 'vertical'
  else:
    np.random.choice(['horizontal', 'vertical'])

def divide(grid, x, y, width, height, orientation):
    if width <= param_stop or height <= param_stop:
        return grid

    horizontal = orientation == 'horizontal'

    # in what direction will the wall be drawn?
    dx = 1 if horizontal else 0
    dy = 0 if horizontal else 1

    # how long will the wall be?
    length = width if horizontal else height

    # where will the wall be drawn from?
    wx = x
    wy = y
    if horizontal:
        add_y = np.random.randint(min_width, height - min_width) // scale * scale
        i = 0
        while grid[wy + add_y, wx - 1] == 0 or grid[wy + add_y, wx + length] == 0:
            add_y = np.random.randint(min_width, height - min_width) // scale * scale
            i += 1
            if i > 49:
                break
        wy += add_y
    else:
        add_x = np.random.randint(min_width, width - min_width) // scale * scale
        i = 0
        while grid[wy-1, wx + add_x] == 0 or grid[wy + length, wx + add_x] == 0:
            add_x = np.random.randint(min_width, width - min_width) // scale * scale
            i += 1
            if i > 49:
                break
        wx += add_x

    if i == 50:
        return grid

    # where will the passage through the wall exist?
    px = wx
    py = wy
    print(wx, wy)
    if horizontal:
        px +=  np.random.randint(1, width-1 - size_door) // scale * scale
    else:
        py += np.random.randint(1, height-1 - size_door) // scale * scale

    for i in range(length):
        if horizontal:
            if wx >= px + size_door or wx < px:
                grid[wy: wy + scale, wx] = 1
        else:
            if wy >= py + size_door or wy < py:
                grid[wy, wx: wx + scale] = 1

        wx += dx
        wy += dy

    plt.imshow(grid)

    nx, ny = x, y
    if horizontal:
        w, h = [width, wy-y]
    else:
        w, h = [wx - x, height]
    grid = divide(grid, nx, ny, w, h, choose_orientation(w, h))

    if horizontal:
        nx, ny = [x, wy + scale]
        w, h = [width, y + height- wy - 1 * scale]
    else:
        nx, ny = [wx + scale , y]
        w, h = [x + width - wx - 1 * scale, height]
    grid = divide(grid, nx, ny, w, h, choose_orientation(w, h))

    return grid


if __name__ == '__main__':
    if not load_and_plot:
        grid = np.zeros([height, width])
        grid[:, :scale] = grid[:scale, :] = grid[-scale:, :] = grid[:, -scale:] = 1
        np.random.seed(seed)
        grid = generate_maze(params)
        plt.imshow(grid)

        # find starting position in top right corner
        directions = ['d', 'l', 'u', 'l', 'd', 'r']
        n_movements = [1, 1, 1, 1, 2, 2]
        i_dir = 0
        found = False
        x = width - scale
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
                    if grid[y - scale: y + scale, x - scale: x + scale].sum() == 0:
                        agent_x = x - scale
                        agent_y = y + scale
                        found = True
                except:
                    pass

            if i_dir in [1, 2, 4, 5]:
                n_movements[i_dir] += 2
            i_dir = (i_dir + 1) % len(directions)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(grid)
        ax.add_patch(Circle((agent_x, agent_y), scale))
        params['agent_position'] = (agent_y, agent_x)

        if save:
            plt.savefig(saving_path + maze_id + '.png')
            np.savetxt(saving_path + maze_id + '_descriptor.txt', grid)
            # save params
            with open(saving_path + maze_id+'_params.pkl', 'wb') as f:
                pickle.dump(params, f)

        plt.show()







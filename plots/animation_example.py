# Some examples of how matplotlib animation could be used for our saving 
# movies illustrating the simulations for the plume project

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Set up formatting for the movie files
fps = 15
Writer = animation.writers['ffmpeg']
writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=1800)

#'''
# Generate a big 3D matrix of data where each slice [i,:,:] represents an
# image in the animation
# In our simulation, we would just be sure that the output of each timestep of 
# simulation gets appended to a big matrix

# I just chose some example sizes for our environment size and time range
num_timesteps = 100
x_size = 400
y_size = 400

# Create a placeholder ndarray for the data
data = np.zeros((num_timesteps,x_size,y_size))

# Step through "simulation" timesteps and generate data -- this would be your main simulation loop
for i in np.arange(num_timesteps):
    frame = np.random.rand(x_size,y_size)
    data[i,:,:] = frame

# Once simulation is finished, go through and generate a list where you append the output
# of each visualization command (imshow in this case) so ims is the series of frames your
# want to have in the video.  For reasons that I don't understand, the thing you have to
# append must be a length 1 array, hence the append([blah]) syntax
fig1 = plt.figure()
env = []
for i in np.arange(len(data)):
    env.append(plt.imshow(data[i,:,:]))
    
# We can also generate some data for the robot positions
robots = []
robot_x = 200
robot_y = 100
dtheta = 2*np.pi/(len(data)-1)
for i in np.arange(len(data)):
    robot_x += 5*np.cos(i*dtheta)
    robot_y += 5*np.sin(i*dtheta)
    robots.append(plt.scatter(robot_x,robot_y))

# To animate multiple "artists" in one frame, create a list where each  
# row is a different frame with all of the artists in it
ims = [list(a) for a in zip(env, robots)]

# Now we visualize the array as an animation
interval = (1.0/fps)*1000;
ani = animation.ArtistAnimation(fig1, ims, repeat=False, interval=interval, blit=True)
ani.save('sim_data.mp4', writer=writer)

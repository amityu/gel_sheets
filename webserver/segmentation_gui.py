import tkinter as tk
from tkinter import filedialog

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from scipy.ndimage import gaussian_filter

mpl.use('TkAgg')

def get_file_location():
    root = tk.Tk() # Create the main window
    root.withdraw() # Hide the main window

    # Customizing the file dialog
    gel_file_location = filedialog.askopenfilename(
        title="Select gel npy file",  # Custom title text
        initialdir="/",  # Initial directory, you can change this
        filetypes=(("Numpy", "*.npy"),)  # File type filters
    )
    height_file_location = filedialog.askopenfilename(
        title="Select height npy file",  # Custom title text
        initialdir="/",  # Initial directory, you can change this
        filetypes=(("Numpy", "*.npy"),)  # File type filters
    )

    return gel_file_location, height_file_location

gel_file, height_file = get_file_location()
print(f"Selected file: {gel_file}, {height_file}")
gel_corrected = np.load(gel_file)
surface = np.load(height_file)


# Create a dummy 3D numpy array as an example
# Replace this with your actual "surface" array
plot_sigma = 0
# Initialize t, y coordinates
global y,t
t = 0
y = gel_corrected.shape[2]//2
vmin = np.nanmin(surface)
vmax = np.nanmax(surface)
# Function to update the plot
def update_plot():
    global y,t

    h = surface[t]

    ax1.clear()
    ax2.clear()
    img = gel_corrected[t,:,y,:]
    ax1.imshow(img,origin='lower', cmap='coolwarm')
    ax1.set_xlabel('X (Pixels)')
    ax1.set_ylabel('Z (Pixels)')
    ax1.set_title('Gel Corrected by illumination filter \n y=%d'%y)

    ax1.plot(gaussian_filter(h[y,: ],sigma = plot_sigma ), 'y', linestyle='--')
    im= ax2.imshow(h, origin='lower', cmap='coolwarm', vmax = vmax, vmin = vmin)
    ax2.hlines(y=y, xmin=0, xmax=h.shape[0], color='b')

    ax2.set_xlabel('X (Pixels)')
    ax2.set_ylabel('Y (Pixels)')
    ax2.set_title('Surface computed \n y=%d'%y)
    fig.suptitle('Segmentation Time=%d ' % t)
    plt.draw()

# Function to handle key presses
def on_key(event):
    global t, y
    if event.key == 'up':
        y = (y + 1) % surface.shape[1]
        update_plot()
    elif event.key == 'down':
        y = (y - 1) % surface.shape[1]
        update_plot()
    elif event.key == 'right':
        t = (t + 1) % surface.shape[0]
        update_plot()
    elif event.key == 'left':
        t = (t - 1) % surface.shape[0]
        update_plot()

# Create a figure and plot the initial surface
fig,(ax1, ax2) = plt.subplots(1,2, figsize = (15,5))
h = surface[0]

sm = ScalarMappable(cmap='coolwarm')
sm.set_array(h)
cbar = plt.colorbar(sm ,ax= ax2)
ax1.clear()
ax2.clear()
img = gel_corrected[t,:,y,:]
ax1.imshow(img,origin='lower', cmap='coolwarm')
ax1.set_xlabel('X (Pixels)')
ax1.set_ylabel('Z (Pixels)')
ax1.set_title('Gel Corrected by illumination filter \n y=%d'%y)

ax1.plot(gaussian_filter(h[y,: ],sigma = plot_sigma ), 'y', linestyle='--')
im= ax2.imshow(h, origin='lower', cmap='coolwarm', vmax = vmax, vmin = vmin )
ax2.hlines(y=y, xmin=0, xmax=h.shape[0], color='b')

ax2.set_xlabel('X (Pixels)')
ax2.set_ylabel('Y (Pixels)')
ax2.set_title('Surface computed \n y=%d'%y)
fig.suptitle('Segmentation Time=%d ' % t)
# Connect the key press event to the on_key function
fig.canvas.mpl_connect('key_press_event', on_key)

# Show the plot
plt.show()

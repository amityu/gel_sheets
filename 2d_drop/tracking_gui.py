from bokeh.io import curdoc
from bokeh.models import ColumnDataSource, Slider, Select, Button, Toggle, RangeSlider
from bokeh.layouts import column, row
from bokeh.plotting import figure
import tifffile
import pandas as pd
import numpy as np
from PIL import Image
import os
from bokeh.models import LinearColorMapper
from bokeh.palettes import Viridis256


# Load images and tracking data
def load_images(_image_folder):
    files = os.listdir(_image_folder)
    files = [file for file in files if file.endswith('.tif')]
    files = sorted(files)
    _images = []
    for file in files:
        # load the image file as a numpy memmap
        imarray = tifffile.imread(os.path.join(_image_folder , file), mode='r')
        #imarray[imarray==0] = np.nan
        # append numpy mem-map in the list
        _images.append(imarray)
    return _images


def load_tracking_data(_csv_file):
    return pd.read_csv(_csv_file, skiprows=[1,2,3])


# Initialize data
DATA_PATH = r'C:\Users\amityu\Gel_Drop_Data'
image_folder = os.path.join(DATA_PATH , r'175_950_ex1_visualize')
print(image_folder)
csv_file = os.path.join(image_folder, 'first_spots.csv')
images = np.array(load_images(image_folder))
tracking_data = load_tracking_data(csv_file)
#color_mapper = LinearColorMapper(palette=Viridis256, low=np.percentile(images,5), high=np.nanpercentile(images, 90))

# Create ColumnDataSource for the image and tracks
image_source = ColumnDataSource(data={'image': [images[0]]})
track_source = ColumnDataSource(data={'x': [], 'y': []})
selected_track_source = ColumnDataSource(data={'x': [], 'y': []})

# Create plot
plot = figure(width=800, height=800, x_range=(0, images[0].shape[1]), y_range=(0, images[0].shape[0]), tools="tap")
plot.image(image='image', x=0, y=0, dw=images[0].shape[1], dh=images[0].shape[0], source=image_source)#, color_mapper=color_mapper,)
track_renderer = plot.circle('x', 'y', size=5, color='red', source=track_source)
selected_track_renderer = plot.circle('x', 'y', size=5, color='blue', source=selected_track_source)
from bokeh.models import SaveTool

# Define additional tools
additional_tools = [SaveTool()]

# Add tools to plot
plot.add_tools(*additional_tools)
# Widgets
frame_slider = Slider(start=0, end=len(images), value=0, step=1, title="Frame")
track_selector = Select(title="Track ID", value="All", options=['All'] + list(tracking_data['TRACK_ID'].unique().astype(str)))
show_all_tracks = Toggle(label="Show All Tracks", button_type="success", active=True)
frame_ahead_slider = Slider(start=0, end=50, value=0, step=1, title="Frames Ahead")

# Callbacks
def update_image(attr, old, new):
    frame = frame_slider.value
    image_source.data = {'image': [images[frame]]}
    update_tracks(attr, old, new)


def update_tracks(attr, old, new):
    frame = frame_slider.value
    frames_ahead = frame_ahead_slider.value
    if show_all_tracks.active:
        # Show all tracks
        tracks = tracking_data[(tracking_data['FRAME'].astype(int) >= frame) & (tracking_data['FRAME'].astype(int) <= frame + frames_ahead)]
        track_source.data = {'x': tracks['POSITION_X'], 'y': tracks['POSITION_Y']}
        selected_track_source.data = {'x': [], 'y': []}
    else:
        # Show selected track
        selected_track_id = track_selector.value
        if selected_track_id != "All":
            tracks = tracking_data[tracking_data['TRACK_ID'].astype(int) == int(selected_track_id)]

            #selected_tracks = tracking_data[(tracking_data['TRACK_ID'].astype(int) == int(selected_track_id)) &
            #                                   (tracking_data['FRAME'].astype(int) >= frame) &
            ##                                  (tracking_data['FRAME'].astype(int) <= frame + frames_ahead)]
        else:
            tracks = tracking_data
            #selected_track_source.data = {'x': selected_tracks['POSITION_X'], 'y': selected_tracks['POSITION_Y']}
        #track_source.data = {'x': [], 'y': []}
        track_source.data = {'x': tracks['POSITION_X'], 'y': tracks['POSITION_Y']}



def update_track_selector(attr, old, new):
    current_frame = frame_slider.value  # Assuming frame_slider holds the current frame
    current_frame_tracks = tracking_data[tracking_data['FRAME'].astype(int) == current_frame]['TRACK_ID'].unique().astype(str)
    track_selector.options = ['All'] + list(current_frame_tracks)
    update_tracks(attr, old, new)  # Existing functionality

def toggle_tracks(attr, old, new):
    show_all_tracks.label = "Show All Tracks" if show_all_tracks.active else "Show Selected Track"
    update_track_selector(attr, old,new)
    update_tracks(attr, old, new)

# Event listeners
frame_slider.on_change('value', update_image)
track_selector.on_change('value', update_track_selector)
show_all_tracks.on_change('active', toggle_tracks)
frame_ahead_slider.on_change('value', update_tracks)

# Layout
layout = column(
    plot,
    row(frame_slider, track_selector, show_all_tracks),
    frame_ahead_slider
)

# Add to document
curdoc().add_root(layout)
curdoc().title = "Track Viewer"

from bokeh.io import curdoc
from bokeh.models import ColumnDataSource, Slider, Select, Button, Toggle, RangeSlider, SaveTool, TapTool
from bokeh.layouts import column, row
from bokeh.plotting import figure
from bokeh.events import Tap
import tifffile
import pandas as pd
import numpy as np
from PIL import Image
import os


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
image_folder = os.path.join(DATA_PATH , r'175_950_ex1_clip')
print(image_folder)
csv_file = os.path.join(DATA_PATH, 'local_spots.csv')
images = np.array(load_images(image_folder))
tracking_data = load_tracking_data(csv_file)
#tracks = tracking_data[tracking_data['FRAME']==488]
#tracks.to_csv(DATA_PATH + 'frame488.csv')
#color_mapper = LinearColorMapper(palette=Viridis256, low=np.percentile(images,5), high=np.nanpercentile(images, 90))

# Create ColumnDataSource for the image and tracks
image_source = ColumnDataSource(data={'image': [images[0]]})
track_source = ColumnDataSource(data={'x': [], 'y': []})
selected_track_source = ColumnDataSource(data={'x': [], 'y': []})

# Create plot
plot = figure(width=700, height=700, x_range=(0, images[0].shape[1]), y_range=(0, images[0].shape[0]), tools="tap")
plot.image(image='image', x=0, y=0, dw=images[0].shape[1], dh=images[0].shape[0], source=image_source)#, color_mapper=color_mapper,)
#track_renderer = plot.circle('x', 'y', size=5, color='red', source=track_source, radius=4, fill_color=None)
selected_track_renderer = plot.circle('x', 'y', size=5, color='blue', source=selected_track_source,radius=4, fill_color=None)

# Define additional tools
additional_tools = [SaveTool()]

# Add tools to plot
plot.add_tools(*additional_tools)
# Widgets
frame_slider = Slider(start=0, end=len(images)-1, value=0, step=1, title="Frame")
track_selector = Select(title="Track ID", value="All", options=['All'] + list(tracking_data['TRACK_ID'].unique().astype(str)))
#show_all_tracks = Toggle(label="Show All Tracks", button_type="success", active=True)
hold_track = Toggle(label="Hold_track", button_type="success", active=True)

frame_ahead_slider = Slider(start=0, end=1000, value=200, step=10, title="Frames Ahead")
taptool = plot.select(type=TapTool)

# Attach the callback

# Callbacks
def update_frame(attr, old, new):
    frame = frame_slider.value
    image_source.data = {'image': [images[frame]]}
    update_tracks(attr, old, new)



def get_coordinates(event):
    x = event.x
    y = event.y
    tracks = tracking_data[tracking_data['FRAME'] == frame_slider.value]
    xs = tracks['POSITION_X']
    ys = tracks['POSITION_Y']
    #selected_track_source.data = {'x': xs, 'y': ys}
    # Calculate the Euclidean distance for each track
    distances = np.sqrt((tracks['POSITION_X'] - x) ** 2 + (tracks['POSITION_Y'] - y) ** 2)



    # Find the track ID which has the minimum distance to the click point
    closest_track_id = str(tracks.loc[distances.idxmin(), 'TRACK_ID'])
    # Set the value of track_selector to the closest track
    selected_track_source.data = {'x':  tracks.loc[distances.idxmin(), ['POSITION_X']], 'y': tracks.loc[distances.idxmin(), ['POSITION_Y']]}
    track_selector.value = closest_track_id

def update_tracks(attr, old, new):
    frame = frame_slider.value
    frames_ahead = frame_ahead_slider.value
    '''if show_all_tracks.active:
        # Show all tracks
        tracks = tracking_data[(tracking_data['FRAME'].astype(int) >= frame) & (tracking_data['FRAME'].astype(int) <= frame + frames_ahead)]

        track_source.data = {'x': tracks['POSITION_X'], 'y': tracks['POSITION_Y']}
        selected_track_source.data = {'x': [], 'y': []}'''
    #else:
    # Show selected track
    selected_track_id = track_selector.value
    if (selected_track_id != "All") & (hold_track.active):
        tracks = tracking_data[(tracking_data['TRACK_ID'].astype(int) == int(selected_track_id)) &(tracking_data['FRAME'].astype(int) == int(frame))]
        #tracks = tracks.iloc[::15,:]
        #selected_tracks = tracking_data[(tracking_data['TRACK_ID'].astype(int) == int(selected_track_id)) &
        #                                   (tracking_data['FRAME'].astype(int) >= frame) &
        ##                                  (tracking_data['FRAME'].astype(int) <= frame + frames_ahead)]
        selected_track_source.data = {'x': tracks['POSITION_X'], 'y': tracks['POSITION_Y']}
    else:
        tracks = tracking_data
        #selected_track_source.data = {'x': selected_tracks['POSITION_X'], 'y': selected_tracks['POSITION_Y']}
    #track_source.data = {'x': [], 'y': []}




def update_track_selector(attr, old, new):
    current_frame = frame_slider.value  # Assuming frame_slider holds the current frame
    current_frame_tracks = tracking_data[tracking_data['FRAME'].astype(int) == current_frame]['TRACK_ID'].unique().astype(str)
    track_selector.options = ['All'] + list(current_frame_tracks)
    update_tracks(attr, old, new)  # Existing functionality

'''def toggle_tracks(attr, old, new):
    show_all_tracks.label = "Show All Tracks" if show_all_tracks.active else "Show Selected Track"
    update_track_selector(attr, old,new)
    update_tracks(attr, old, new)'''

def toggle_hold_track(attr, old, new):
    hold_track.label = "Hold Track" if hold_track.active else "Move tracks"
    update_track_selector(attr, old,new)
    update_tracks(attr, old, new)

# Event listeners
frame_slider.on_change('value', update_frame)
track_selector.on_change('value', update_track_selector)
#show_all_tracks.on_change('active', toggle_tracks)
hold_track.on_change('active', toggle_hold_track)
plot.on_event(Tap, get_coordinates)
frame_ahead_slider.on_change('value', update_tracks)

# Layout
layout = column(
    plot,
    row(frame_slider, track_selector,#, show_all_tracks),
    frame_ahead_slider, hold_track)
)

# Add to document
curdoc().add_root(layout)
curdoc().title = "Track Viewer"

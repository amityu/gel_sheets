from bokeh.io import curdoc
from bokeh.models import ColumnDataSource, Slider, Select, Button, Toggle, RangeSlider, SaveTool, TapTool, CheckboxGroup
from bokeh.layouts import column, row
from bokeh.plotting import figure
from bokeh.events import Tap
from bokeh.palettes import Category20
from skimage.filters import gaussian
import tifffile
import pandas as pd
import numpy as np
from PIL import Image
import os
import random

# data for making the app available to other users
port= 5006
print('--------------------to open for web-----------------------------------------------')
print('bokeh serve --show tracking_gui.py --address 0.0.0.0 --allow-websocket-origin=132.72.216.33:{}'.format(port))
print('open ip 132.72.216.33:{}'.format(port))
print('----------------------------------------------------------------------------------')
# Load images and tracking data - not used
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


# the app loads each frame when needed so that app launches faster
def get_images_list(_image_folder):
    files = os.listdir(_image_folder)
    files = [os.path.join(image_folder,file) for file in files if file.endswith('.tif')]

    return sorted(files)



def load_tracking_data(_csv_file):

    return pd.read_csv(_csv_file)


# Initialize data
DATA_PATH = r'C:\Users\amityu\Gel_Drop_Data'



movie = 'dome'
if movie == 'eye':
    local_path = 'eye_local'
    image_folder = os.path.join(DATA_PATH , 'eye_clip')
    track_list_file_path = r'C:\Users\amityu\Gel_Drop_Data\masks\particle_eye813_995_.csv'

elif movie =='dome':

    track_list_file_path = r'C:\Users\amityu\Gel_Drop_Data\masks\particle_dome530_743_.csv'

    local_path = '175_950_ex1_local'
    image_folder = os.path.join(DATA_PATH, r'175_950_ex1_clip')
elif movie == 'e561e1':
    #track_list_file_path = None# r'C:\Users\amityu\Gel_Drop_Data\masks\particle_5651200_1450_.csv'
    track_list_file_path = r'C:\Users\amityu\Gel_Drop_Data\masks\particle_5651200_1450_.csv'
    particle_list_flag = True
    local_path = 'e561e1_local'
    image_folder = os.path.join(DATA_PATH, 'e561e1_clip')

LOCAL_PATH = os.path.join(DATA_PATH, local_path)

csv_file = os.path.join(LOCAL_PATH, 'trackmate.csv')
stat_file = os.path.join(LOCAL_PATH, 'tracks.csv')
stats_df = pd.read_csv(stat_file)

# filtering tracks if particle_list flag is False
min_frame = 0
max_frame = stats_df['MAX_FRAME'].max()
min_duration = 200
max_duration = max_frame
max_track_no = 50
color_list = Category20[20]

tracks_list = list(stats_df[(stats_df.DURATION >=  min_duration) & (stats_df.DURATION <= max_duration) & (stats_df.MIN_FRAME >=  min_frame) & (stats_df.MIN_FRAME <= max_frame) ]['TRACK_ID'])
tracks_list = random.sample(tracks_list, min(len(tracks_list),max_track_no))
particle_list_flag = True
if particle_list_flag:
    tracks_list = pd.read_csv(track_list_file_path)['id'].to_list()

images_list = get_images_list(image_folder)

tracking_data = load_tracking_data(csv_file)
tracking_data = tracking_data[tracking_data['TRACK_ID'].isin(tracks_list)]

img0 = tifffile.imread(images_list[0])

# Create ColumnDataSource for the image and tracks
image_source = ColumnDataSource(data={'image': [gaussian(img0,1)]})
track_source = ColumnDataSource(data={'x': [], 'y': []})
selected_track_source = ColumnDataSource(data={'x': [], 'y': []})


# Create plot
plot = figure(width=700, height=700, x_range=(0, img0.shape[1]), y_range=(0, img0.shape[0]), tools="tap")
plot.image(image='image', x=0, y=0, dw=img0.shape[1], dh=img0.shape[0], source=image_source)#, color_mapper=color_mapper,)
#track_renderer = plot.circle('x', 'y', size=5, color='red', source=track_source, radius=4, fill_color=None)
selected_track_renderer = plot.circle('x', 'y',  color='yellow', source=selected_track_source,radius=6, fill_color=None, line_width =3)
# Define the checkbox group
checkbox_group = CheckboxGroup(labels=["show only future born tracks", "hide dead tracks"], active=[0, 1])
#full_track_renderer = plot.circle('x', 'y',  color='green', source=selected_track_source,radius=6, fill_color=None, line_width =3)
track_lines = []
for i in range(len(tracks_list)):
    track_i = tracking_data[tracking_data['TRACK_ID'] == tracks_list[i]]
    line = plot.line(list(track_i.POSITION_X), list(track_i.POSITION_Y),  line_color = color_list[i%len(color_list)], line_width = 2)
    track_lines.append(line)


# Define additional tools
additional_tools = [SaveTool()]

# Add tools to plot
plot.add_tools(*additional_tools)
# Widgets
frame_slider = Slider(start=0, end=len(images_list)-1, value=0, step=1, title="Frame")
#frame_slider = Slider(start=100, end=180, value=0, step=1, title="Frame")

track_selector = Select(title="Track ID", value="All", options=['All'] + list(tracking_data['TRACK_ID'].unique().astype(str)))
#show_all_tracks = Toggle(label="Show All Tracks", button_type="success", active=True)
full_tracks = Toggle(label="toggle_tracks_visibility", button_type="success", active=True)

frame_ahead_slider = Slider(start=0, end=1000, value=200, step=10, title="Frames Ahead")
taptool = plot.select(type=TapTool)

# Attach the callback


# Callbacks
def update_frame(attr, old, new):
    frame = frame_slider.value
    #image_source.data = {'image': [gaussian(images[frame],1)]}
    image_source.data = {'image': [gaussian(tifffile.imread(images_list[frame]),1)]}
    update_tracks(attr, old, new)
    update_track_selection()



#coosing track from cursor
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

    # Show selected track
    selected_track_id = track_selector.value
    if (selected_track_id != "All"):# & (hold_track.active):
        tracks = tracking_data[(tracking_data['TRACK_ID'].astype(int) == int(selected_track_id)) &(tracking_data['FRAME'].astype(int) == int(frame))]


        selected_track_source.data = {'x': tracks['POSITION_X'], 'y': tracks['POSITION_Y']}
    else:
        tracks = tracking_data[tracking_data['FRAME'].astype(int) == int(frame)]

        selected_track_source.data = {'x': tracks['POSITION_X'], 'y': tracks['POSITION_Y']}




def update_track_selector(attr, old, new):
    current_frame = frame_slider.value  # Assuming frame_slider holds the current frame
    current_frame_tracks = tracking_data[tracking_data['FRAME'].astype(int) == current_frame]['TRACK_ID'].unique().astype(str)
    track_selector.options = ['All'] + list(current_frame_tracks)
    update_tracks(attr, old, new)  # Existing functionality

def toggle_full_track(attr, old, new):
   for line in track_lines:
       line.visible = not line.visible

# update tracks that were future born or died
def update_track_selection():
    current_frame= frame_slider.value
    future_born_tracks = set(stats_df[stats_df['MIN_FRAME'].astype(int) > current_frame]['TRACK_ID'].unique().astype(str))# Define a callback function for the checkbox group
    #print(born_tracks)
    died_tracks = set(stats_df[stats_df['MAX_FRAME'].astype(int) < current_frame]['TRACK_ID'].unique().astype(str))#
    selected = checkbox_group.active
    active_tracks = set(list(map(str,tracks_list)))
    if (0 in selected) & (1  not in selected):
        active_tracks = future_born_tracks

    elif (0 not in selected) & (1  in selected):
        active_tracks = active_tracks - died_tracks
    elif (0  in selected) & (1 in selected):
        active_tracks = future_born_tracks - died_tracks
    for _i, _track_i in enumerate(tracks_list):
         track_lines[_i].visible = str(_track_i) in active_tracks
def checkbox_callback(attr, old, new):
    update_track_selection()
# Event listeners
frame_slider.on_change('value', update_frame)
track_selector.on_change('value', update_track_selector)
#show_all_tracks.on_change('active', toggle_tracks)
full_tracks.on_change('active', toggle_full_track)
plot.on_event(Tap, get_coordinates)
frame_ahead_slider.on_change('value', update_tracks)
checkbox_group.on_change('active', checkbox_callback)
# Layout
layout = column(
    row(checkbox_group,plot),
    row(frame_slider, track_selector,
    frame_ahead_slider, full_tracks)
)

# Add to document
curdoc().add_root(layout)
curdoc().title = "Track Viewer"

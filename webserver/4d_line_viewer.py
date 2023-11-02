import os
# import disk from skimage

#%%
PROJECT_PATH = 'C:/Users/amityu/DataspellProjects/gel_sheets/'
DATA_PATH = 'C:/Users/amityu/Gel_Sheet_Data/'
#movie = 'Control'
#movie = '130721'
#movie ='140721'
#movie ='150721'
#movie ='100621'
movie ='130721_CCA60_RAW'
#movie ='280523 AM100 568'
ADD_PATH = os.path.join(PROJECT_PATH, "add_data/", movie + "/")


MOVIE_PATH = DATA_PATH +  movie + '/'
GRAPH_PATH = 'C:/Users/amityu/Gel_Sheet_Graph/'

from bokeh.plotting import figure, curdoc
from bokeh.layouts import column, row
from bokeh.models import Slider, ColumnDataSource
import numpy as np

distribution = np.load(MOVIE_PATH + 'tmp/distribution_full.npy')
class DataExplorer:
    def __init__(self, data):
        self.data = data
        self.t, self.r, self.y, self.x = data.shape

        # Initialize sliders for selecting a specific (r, y, x) point
        self.t_slider = Slider(start=0, end=self.t-1, value=0, step=1, title="T Axis")
        self.y_slider = Slider(start=0, end=self.y-1, value=self.y//2, step=1, title="Y Axis")
        self.x_slider = Slider(start=0, end=self.x-1, value=self.x//2, step=1, title="X Axis")

        # Initialize data source
        self.source = ColumnDataSource(data={'r': np.arange(self.r-2), 'value': data[0, 2:, self.y//2, self.x//2]})

        # Create line plot
        self.p = figure(title="Line Plot Along R Axis", width=600, height=300)
        self.p.line('r', 'value', source=self.source)

        # Set up callbacks
        self.t_slider.on_change('value', self.update_data)
        self.y_slider.on_change('value', self.update_data)
        self.x_slider.on_change('value', self.update_data)

    def update_data(self, attr, old, new):
        t = int(self.t_slider.value)
        y = int(self.y_slider.value)
        x = int(self.x_slider.value)
        new_data = {'r': np.arange(self.r), 'value': self.data[t, :, y, x]}
        self.source.data = new_data

    def create_layout(self):
        sliders = column(self.t_slider, self.y_slider, self.x_slider)
        layout = row(sliders, self.p)
        return layout

explorer = DataExplorer(distribution)
curdoc().add_root(explorer.create_layout())

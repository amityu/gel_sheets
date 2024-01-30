'''
this is a bokeh app that allows you to view a line scans of a 4d data set along the second axis(Z)
'''



import os
from bokeh.plotting import figure, curdoc
from bokeh.layouts import column, row
from bokeh.models import Slider, ColumnDataSource
import numpy as np
from bokeh.server.server import Server
from bokeh.models import Select
from scipy.ndimage import gaussian_filter1d

#%%
PROJECT_PATH = 'C:/Users/amityu/DataspellProjects/gel_sheets/'
DATA_PATH = 'C:/Users/amityu/Gel_Sheet_Data/'
#movie = 'Control'
#movie = '130721'
#movie ='140721'
#movie ='150721'
#movie ='100621'
#movie ='130721_CCA60_RAW'
movie ='280523 AM100 568_1'
ADD_PATH = os.path.join(PROJECT_PATH, "add_data/", movie + "/")


MOVIE_PATH = DATA_PATH +  movie + '/'
GRAPH_PATH = 'C:/Users/amityu/Gel_Sheet_Graph/'


data = np.load(MOVIE_PATH + 'np/motors_norm.npy')


def modify_doc(doc):


    class DataExplorer:
        def __init__(self, data):
            self.data = data
            self.t, self.z, self.y, self.x = data.shape

            # Initialize sliders for selecting a specific (r, y, x) point
            self.t_slider = Slider(start=0, end=self.t-1, value=0, step=1, title="T Axis")
            self.y_slider = Slider(start=0, end=self.y-1, value=self.y//2, step=1, title="Y Axis")
            self.x_slider = Slider(start=0, end=self.x-1, value=self.x//2, step=1, title="X Axis")

            # Initialize data source
            self.source = ColumnDataSource(data={'z': np.arange(self.z-2), 'value': data[0, 2:, self.y//2, self.x//2]})

            # Create line plot
            self.p = figure(title="Line Plot Along Z Axis", width=600, height=300, y_range=(0.6, 1.5))
            self.p.line('z', 'value', source=self.source)
            self.p.xaxis.axis_label = "Z axis"
            self.p.yaxis.axis_label = "(Intensity) Value"
            # Create a dropdown menu for sigma
            self.sigma_select = Select(title="Sigma:", value="0", options=[str(i/2) for i in range(20)])
            self.sigma_select.on_change('value', self.update_sigma)

        # Set up callbacks
            self.t_slider.on_change('value', self.update_data)
            self.y_slider.on_change('value', self.update_data)
            self.x_slider.on_change('value', self.update_data)

        def update_data(self, attr, old, new):
            t = int(self.t_slider.value)
            y = int(self.y_slider.value)
            x = int(self.x_slider.value)
            new_data = {'z': np.arange(self.z), 'value': self.data[t, :, y, x].copy()}
            self.source.data = new_data
            self.update_sigma(None, None, None)
        def update_sigma(self, attr, old, new):
            # Retrieve the current data
            t = int(self.t_slider.value)
            y = int(self.y_slider.value)
            x = int(self.x_slider.value)

            # Apply the sigma
            sigma = float(self.sigma_select.value)
            line_values = self.data[t, :, y, x].copy()

            if sigma > 0:
                filtered_data = gaussian_filter1d(line_values, sigma=sigma)
            else:
                filtered_data = line_values
            new_data = {'z': np.arange(self.z), 'value': filtered_data}


            self.source.data = new_data



        def create_layout(self):
            sliders = column(self.t_slider, self.y_slider, self.x_slider, self.sigma_select)
            layout = row(sliders, self.p)
            return layout



    explorer = DataExplorer(data)
    doc.add_root(explorer.create_layout())

def bokeh_app(doc):
    modify_doc(doc)

# Create and start the Bokeh server
server = Server({'/': bokeh_app}, port=5000, address='132.72.216.33', allow_websocket_origin=["*"])
print(server.address + ':' + str(server.port))

server.start()

# Run the server until stopped
server.io_loop.add_callback(server.show, "/")
server.io_loop.start()
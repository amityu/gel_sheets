import os

from skimage.draw import disk

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
#movie ='280523 AM100 568_1'
ADD_PATH = os.path.join(PROJECT_PATH, "add_data/", movie + "/")


MOVIE_PATH = DATA_PATH +  movie + '/'
GRAPH_PATH = 'C:/Users/amityu/Gel_Sheet_Graph/'
import numpy as np

from bokeh.models import ColumnDataSource
from bokeh.plotting import figure

distribution = np.load(MOVIE_PATH + 'tmp/distribution_full%d.npy'%0)

# Prepare 3D data (t, y, x)
rr,cc = disk((256, 256), 20, shape=distribution.shape[1:])
disc = np.zeros(distribution.shape[1:])
disc[rr,cc] = 1

data = np.zeros((62, distribution.shape[1], distribution.shape[2]))
for t in range(62):
     data[t] = disc


class FunctionLine:
    def __init__(self, data):
        self.data = data
        self.time_index = 0

        self.num_images = data.shape[0]
        self.source = ColumnDataSource(data={'line': [data[0,:,data.shape[2]//2, data.shape[3]//2]]})
        self.p = figure(width=800, height=800, x_range=(0, 10), y_range=(0, 10), title="Curvature")
        #label = Label(x=70, y=150, x_units='screen', y_units='screen',
        #              text=r'$$K(x, y) = \frac{z_{xx} \cdot z_{yy} - (z_{xy})^2}{(1 + z_x^2 + z_y^2)^2}$$',
        #              text_font_size='16pt', background_fill_alpha=0.3, text_color = 'red')
        #self.p.add_layout(label)
        # Create a diverging palette
        #mapper = LogColorMapper(palette='Viridis256', low=vmin, high=vmax)
        #mapper = LinearColorMapper(palette='Viridis256', low=vmin, high=vmax)

        #color_bar = ColorBar(color_mapper=mapper, location=(0,0))

        self.p.line(x=np.arange(self.data.shape[1]), y='line', source=self.source)
        #self.p.add_layout(color_bar, 'right')
        #hover = HoverTool(renderers=[renderer], tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image{0.000}")])
        #self.p.add_tools(hover)

    def update_line(self,t,y,x):
        if 0 <= t < self.num_images:
            self.source.data = {'line': [self.data[t,:, y, x]]}
        else:
            raise IndexError("Image index out of bounds!")

class RodArea:
    def __init__(self, radial_distribution, function_line):
        self.rod_map = np.zeros((radial_distribution.shape[0], radial_distribution.shape[2], radial_distribution.shape[3]))
        rr_disc, cc_disc = disk((max_r,max_r), max_r)
        for t in range(radial_distribution.shape[0]):
            self.rod_map[t, rr, cc] = t/radial_distribution.shape[0]
        self.image_index = 0
        #self.num_images = .shape[0]
        self.source = ColumnDataSource(data={'image': [rod_map[0,:, :]]})
        self.p = figure(width=800, height=800, x_range=(0, 10), y_range=(0, 10), title="Curvature", tools='tap')
        self.p.image(image='image', x=0, y=0, dw=10, dh=10, source=self.source, color_mapper=LinearColorMapper(palette=Viridis256))
        self.function_line = function_line

    def tap_callback(self, event):
        try:
            x, y = event.x, event.y
            self.function_line.update_line(t=self.image_index, y=y, x=x)
        except(e):
            print(e)
            pass

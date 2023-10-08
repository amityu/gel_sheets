import numpy as np
from bokeh.io import push_notebook, show, output_notebook
from bokeh.plotting import figure, curdoc
from bokeh.models import LinearColorMapper, Slider, ColorBar
from bokeh.layouts import column
from bokeh.palettes import Viridis256
from bokeh.models.sources import ColumnDataSource
from skimage.filters import sobel_h, sobel_v
import numpy as np
import os

from scipy.ndimage import binary_dilation
#%% md

PROJECT_PATH = 'C:/Users/amityu/DataspellProjects/gel_sheets/'
DATA_PATH = 'D:/Data/'
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
def modify_doc(doc):
    global spike, curvature
    print(np.nanmin(curvature), np.nanmax(curvature))
    # Example 3D data
    data_3d = curvature  # Replace this with your 3D array

    # Create ColumnDataSource
    source = ColumnDataSource(data={'image': [data_3d[0,:, :]]})

    # Set up plot
    p = figure(width=800, height=800, x_range=(0, 10), y_range=(0, 10), title="Curvature")
    mapper = LinearColorMapper(palette="Viridis256", low=-1, high=3)

    color_bar = ColorBar(color_mapper=mapper, location=(0,0))

    p.image(image='image', x=0, y=0, dw=10, dh=10, source=source, color_mapper=LinearColorMapper(palette=Viridis256))
    p.add_layout(color_bar, 'right')
    # Set up slider
    slider = Slider(start=0, end=data_3d.shape[0]-1, value=0, step=1, title="Index")

    # Update function
    def update(attr, old, new):
        source.data = {'image': [data_3d[slider.value,:, :]]}

    slider.on_change('value', update)

    # Organize layout
    layout = column(p, slider)

    # Add to doc

    doc.add_root(layout)
def gaussian_curvature(surface):
    nan_mask = np.isnan(surface)
    dilated_nan_mask = binary_dilation(nan_mask)
    # Compute the partial derivatives of the surface using convolution
    dx = sobel_h(surface)
    dy = sobel_v(surface)
    dxx = sobel_h(dx)
    dyy = sobel_v(dy)
    dxy = sobel_v(dx)
    # Compute the curvature
    curvature = (dxx * dyy - dxy**2) / (1 + dx**2 + dy**2)**2
    curvature[dilated_nan_mask] = np.nan
    return curvature

if __name__ == '__main__':
    global spike, curvature
    from bokeh.server.server import Server
    from bokeh.application import Application
    from bokeh.application.handlers.function import FunctionHandler
    spike = np.load(MOVIE_PATH + 'tmp/filtered_spike.npy')
    curvature = np.zeros_like(spike)
    sigma = 5
    for t,img in enumerate(spike):
        z = img  # Some example data with noise
        curvature[t] = gaussian_curvature(z)

    print(np.nanmin(curvature), np.nanmax(curvature))

    apps = {'/': Application(FunctionHandler(modify_doc))}
    server = Server(apps, port=5004)
    server.start()
    server.io_loop.start()

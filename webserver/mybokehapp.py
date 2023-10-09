import numpy as np
from bokeh.io import push_notebook, show, output_notebook
from bokeh.plotting import figure, row
from bokeh.models import LinearColorMapper, Slider, ColorBar, HoverTool
from bokeh.layouts import column
from bokeh.palettes import Viridis256, all_palettes, linear_palette
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
    source1 = ColumnDataSource(data={'image': [data_3d[0,:, :]]})
    source2 = ColumnDataSource(data={'image': [spike[0,:, :]]})

    # Set up plot
    p1 = figure(width=800, height=800, x_range=(0, 10), y_range=(0, 10), title="Curvature")

    # Create a diverging palette

    mapper1 = LinearColorMapper(palette='Viridis256', low=vmin, high=vmax)

    color_bar1 = ColorBar(color_mapper=mapper1, location=(0,0))

    renderer1 = p1.image(image='image', x=0, y=0, dw=10, dh=10, source=source1, color_mapper=LinearColorMapper(palette=Viridis256))
    p1.add_layout(color_bar1, 'right')
    hover1 = HoverTool(renderers=[renderer1], tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image{0.000}")])
    p1.add_tools(hover1)

    p2 = figure(width=800, height=800, x_range=(0, 10), y_range=(0, 10), title="surface")
    mapper2 = LinearColorMapper(palette="Viridis256", low=0, high=100)

    color_bar2 = ColorBar(color_mapper=mapper2, location=(0,0))

    p2.image(image='image', x=0, y=0, dw=10, dh=10, source=source2, color_mapper=LinearColorMapper(palette=Viridis256))
    p2.add_layout(color_bar2, 'right')
    renderer2 = p2.image(image='image', x=0, y=0, dw=10, dh=10, source=source2, color_mapper=LinearColorMapper(palette=Viridis256))
    hover2 = HoverTool(renderers=[renderer2], tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")])
    p2.add_tools(hover2)

    # Set up slider
    slider = Slider(start=0, end=data_3d.shape[0]-1, value=0, step=1, title="Index")

    # Update function
    def update(attr, old, new):
        curv_image = data_3d[slider.value,:, :].copy()
        #curv_image[curv_image > 0] = 1

        source1.data = {'image': [curv_image]}
        source2.data = {'image': [spike[slider.value,:, :]]}


    slider.on_change('value', update)

    # Organize layout
    layout = column(row(p1,p2), slider)

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
    curvature = curvature * 1000
    vmin = -50
    vmax = 50
    print(np.nanmin(curvature), np.nanmax(curvature))

    apps = {'/': Application(FunctionHandler(modify_doc))}
    server = Server(apps, port=5004)
    server.start()
    server.io_loop.start()

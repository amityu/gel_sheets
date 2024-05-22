import os

import numpy as np
from bokeh.layouts import column
from bokeh.models import LinearColorMapper, Slider, ColorBar, HoverTool, Label
from bokeh.models.sources import ColumnDataSource
from bokeh.palettes import Viridis256
from bokeh.plotting import figure, row
from scipy.ndimage import binary_dilation
from skimage.filters import sobel_h, sobel_v

#%% md

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


class CurveImage:
    def __init__(self, data):
        self.data = data
        self.image_index = 0
        self.num_images = data.shape[0]
        self.source = ColumnDataSource(data={'image': [data[0,:, :]]})
        self.p = figure(width=800, height=800, x_range=(0, 10), y_range=(0, 10), title="Curvature")
        label = Label(x=70, y=150, x_units='screen', y_units='screen',
                  text=r'$$K(x, y) = \frac{z_{xx} \cdot z_{yy} - (z_{xy})^2}{(1 + z_x^2 + z_y^2)^2}$$',
                  text_font_size='16pt', background_fill_alpha=0.3, text_color = 'red')
        self.p.add_layout(label)
        # Create a diverging palette
        #mapper = LogColorMapper(palette='Viridis256', low=v_min, high=v_max)
        mapper = LinearColorMapper(palette='Viridis256', low=vmin, high=vmax)

        color_bar = ColorBar(color_mapper=mapper, location=(0,0))

        renderer = self.p.image(image='image', x=0, y=0, dw=10, dh=10, source=self.source, color_mapper=mapper)
        self.p.add_layout(color_bar, 'right')
        hover = HoverTool(renderers=[renderer], tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image{0.000}")])
        self.p.add_tools(hover)

    def update_curve(self,attr, old, new):
        if 0 <= new < self.num_images:
            self.source.data = {'image': [self.data[new,:, :]]}
        else:
            raise IndexError("Image index out of bounds!")


def modify_doc(doc):
    global spike, curvature
    print(np.nanmin(curvature), np.nanmax(curvature))
    # Example 3D data
    cimage = CurveImage(curvature)
    # Create ColumnDataSource

    source2 = ColumnDataSource(data={'image': [spike[0,:, :]]})

    # Set up plot

    p2 = figure(width=800, height=800, x_range=(0, 10), y_range=(0, 10), title="surface")
    mapper2 = LinearColorMapper(palette="Viridis256", low=0, high=100)

    color_bar2 = ColorBar(color_mapper=mapper2, location=(0,0))

    p2.image(image='image', x=0, y=0, dw=10, dh=10, source=source2, color_mapper=LinearColorMapper(palette=Viridis256))
    p2.add_layout(color_bar2, 'right')
    renderer2 = p2.image(image='image', x=0, y=0, dw=10, dh=10, source=source2, color_mapper=LinearColorMapper(palette=Viridis256))
    hover2 = HoverTool(renderers=[renderer2], tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")])
    p2.add_tools(hover2)

    # Set up slider
    slider1 = Slider(start=0, end=spike.shape[0]-1, value=0, step=1, title="Index")
    slider2 = Slider(start=0, end=spike.shape[0]-1, value=0, step=1, title="Index")

    def update(attr, old, new):

        source2.data = {'image': [spike[slider2.value,:, :]]}

    slider1.on_change('value', cimage.update_curve)
    slider2.on_change('value', update)

    # Organize layout
    layout = row(column(cimage.p, slider1), column([p2,slider2], sizing_mode = 'stretch_width'), sizing_mode='stretch_width')

    # Add to doc

    doc.add_root(layout)
    doc.template = f"""
    {{{{ bokeh_css }}}}
    {{{{ bokeh_js }}}}
    <META http-equiv="Content-Type" content="text/html; charset=UTF-8">
    {MATHJAX_SCRIPT}
    {{{{ plot_div|indent(8) }}}}
    {{{{ plot_script|indent(8) }}}}
    """
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

def signed_log(x):
    return np.sign(x[~np.isnan(x)]) * np.log1p(np.abs(x[~np.isnan(x)]))
if __name__ == '__main__':
    global spike,  vmin,vmax
    from bokeh.server.server import Server
    from bokeh.application import Application
    from bokeh.application.handlers.function import FunctionHandler
    from tornado.ioloop import IOLoop
    spike = np.load(MOVIE_PATH + 'tmp/filtered_spike.npy')
    curvature = np.zeros_like(spike)
    sigma = 5
    for t,img in enumerate(spike):
        z = img  # Some example data with noise
        curvature[t] = gaussian_curvature(z)
    curvature = curvature * 1000
    vmin = np.nanmin(signed_log(curvature))
    vmax = np.nanmax(signed_log(curvature))
    print(vmin,vmax)
    MATHJAX_SCRIPT = """
    <script type="text/javascript" async
    src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML">
    </script>
    """

    apps = {'/myapp': Application(FunctionHandler(modify_doc))}
    server = Server(apps,  io_loop=IOLoop.current(), port=5000, address='132.72.216.33', allow_websocket_origin=["*"])
    print(server.address + ':' + str(server.port))
#   server = Server(apps, port=5000)
    server.start()
    server.io_loop.add_callback(server.show, "/myapp")

    server.io_loop.start()

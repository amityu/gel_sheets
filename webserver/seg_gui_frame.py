import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, render_template, jsonify
from matplotlib.cm import ScalarMappable
from scipy.ndimage import gaussian_filter
from preprocessing import preprocessing_v2 as pp
# app.py
global y,t
DATA_PATH = 'D:/Gel_Sheet_Data/'

#movie = 'Control'
#movie = '130721'
#movie ='140721'
#movie ='150721'
#movie ='100621'
#movie = '130721_CCA60_RAW'
#movie ='280523 AM100 568_3'
#movie = 'control_1_050721'
#movie = 'cca120'
#movie = 'cca40'
#movie = 'cca300'
#movie = 'cca60_dist'
#movie = '160624c10'
#movie = '270524c8'
movie = '140824'
#movie = '140824long'
#movie = 'cca120_am200'
#movie ='280523 AM100 568_3'
MOVIE_PATH = DATA_PATH +  movie + '/'
TMP_PATH = 'D/Gel_Sheet_Graph/tmp/'
#ex_data = pp.get_ex_data(MOVIE_PATH)
global y,t, image_url

image_url = 'static/images/segimage.png'
plt.plot([1,2,3,4])
plt.savefig(image_url)
plt.close()

#gel = np.load(MOVIE_PATH + 'tmp/gel_norm_before_stabilize.npy', mmap_mode='r')#[:, 20:,80:]
gel = np.load(MOVIE_PATH + 'np/gel_norm.npy', mmap_mode='r')#[:, 20:,80:]

#surface = np.load(MOVIE_PATH + 'tmp/height45_s3.npy')#[:, 20:,80:]
surface = np.load(MOVIE_PATH + 'np/height.npy'.format(movie), mmap_mode='r')
# Replace this with your actual "surface" array
plot_sigma = 0


# Initialize t, y coordinates
t = 0
y = gel.shape[2]//2

vmin = np.nanmin(surface)
vmax = np.nanmax(surface)
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/arrow/<direction>')
def arrow(direction):
    global t, y, image_url

    if direction == 'up':
        y = (y + 5) % surface.shape[1]


    elif direction == 'down':
        y = (y - 5) % surface.shape[1]

    elif direction == 'left':
        t = (t - 1) % surface.shape[0]

    elif direction == 'right':
        t = (t + 1) % surface.shape[0]
    else:
        image_url = ""

    update_plot()
    return jsonify({"image_url": image_url})
def update_plot():
    fig,(ax1, ax2) = plt.subplots(1,2, figsize = (15,5))


    h = surface[t]

    sm = ScalarMappable(cmap='coolwarm')
    sm.set_array(h)
    cbar = plt.colorbar(sm ,ax= ax2)

    img = gel[t,:,y,:]
    im1 = ax1.imshow(img,origin='lower', cmap='coolwarm', aspect='auto')
    ax1.set_xlabel('X (Pixels)')
    ax1.set_ylabel('Z (Pixels)')
    ax1.set_title('Gel Corrected by illumination filter \n y=%d'%y)

    ax1.plot(gaussian_filter(h[y,: ],sigma = plot_sigma ), 'y', linestyle='--')
    fig.colorbar(im1, ax =ax1)

    im= ax2.imshow(h, origin='lower', cmap='coolwarm', vmax = vmax, vmin = vmin)
    ax2.hlines(y=y, xmin=0, xmax=h.shape[0], color='b')

    ax2.set_xlabel('X (Pixels)')
    ax2.set_ylabel('Y (Pixels)')
    ax2.set_title('Surface computed \n y=%d'%y)
    try:
        fig.suptitle('Segmentation Time=%d \n Gel %s' % (t,ex_data['name']))
    except:
        fig.suptitle('Segmentation Time=%d \n Gel %s' % (t,movie))

    plt.savefig(image_url)
    plt.close()




if __name__ == '__main__':

    app.run(debug=False, host= '0.0.0.0' , port=5035)
    print('hello')


    # Function to update the plot
    # Function to handle key presses


from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

'''@app.route('/get_plot')
def get_plot():
    # Sample code to generate a plot
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    plt.plot(x, y)

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return jsonify({'plot_url': 'data:image/png;base64,{}'.format(plot_url)})'''
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)


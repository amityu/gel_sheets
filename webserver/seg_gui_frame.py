# app.py

from flask import Flask, render_template, jsonify
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/arrow/<direction>')
def arrow(direction):
    if direction == 'up':
        image_url = "/static/images/up_arrow.jpg"
    elif direction == 'down':
        image_url = "/static/images/down_arrow.jpg"
    elif direction == 'left':
        image_url = "/static/images/left_arrow.jpg"
    elif direction == 'right':
        image_url = "/static/images/right_arrow.jpg"
    else:
        image_url = ""

    return jsonify({"image_url": image_url})

if __name__ == '__main__':
    app.run(debug=True, port=8080)

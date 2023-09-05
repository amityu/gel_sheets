from flask import Flask

# Create a Flask app
app = Flask(__name__)

# Define a route and function to handle requests
@app.route('/')
def hello_world():
    return 'Hello, World! This is a Flask web app.'

# Run the app when the script is executed
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1000)

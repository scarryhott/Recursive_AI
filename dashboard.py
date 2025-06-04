from flask import Flask, jsonify
from recursive_ai import RecursiveAI

app = Flask(__name__)

ai = RecursiveAI()

@app.route('/')
def index():
    data = [node.__dict__ for node in ai.history]
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)

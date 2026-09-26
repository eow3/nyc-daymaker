import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/plan", methods=["POST"])
def plan():
    data = request.json
    return jsonify({"plan": f"Mood: {data['mood']}, time: {data['hours']} hrs. Plan coming soon!"})

if __name__ == "__main__":
    app.run(debug=True)
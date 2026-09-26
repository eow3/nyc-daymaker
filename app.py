import os
import json
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
app = Flask(__name__)

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL = "gemini-3.8-flash"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/plan", methods=["POST"])
def plan():
    data = request.json
    now = datetime.now().strftime("%A %I:%M %p")

    prompt = f"""
You are a friendly NYC local helping someone who is tired of deciding what to do.
It is currently {now}. Their mood is "{data['mood']}" and they have {data['hours']} hours.

Suggest 3 to 4 real places in New York City, in a sensible order, that fit the mood,
time of day, and time available. Keep stops close enough to walk or take one subway ride.

Respond ONLY with JSON in this exact format:
{{
  "summary": "one short, upbeat sentence describing the plan",
  "stops": [
    {{"name": "place name", "lat": 40.0, "lng": -73.0, "time": "4:00 PM", "why": "one short sentence"}}
  ]
}}
"""

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        result = json.loads(response.text)
        return jsonify({"plan": result["summary"], "stops": result["stops"]})
    except Exception as e:
        print("Gemini error:", e)
        return jsonify({"plan": "Something went wrong making your plan. Try again!", "stops": []}), 500

if __name__ == "__main__":
    app.run(debug=True)
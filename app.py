import os
import json
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from google import genai
from google.genai import types
from tavily import TavilyClient
import time

load_dotenv()
app = Flask(__name__)

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
MODELS = ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite"]


def find_events(mood):
    """Search for things happening in NYC today. Returns a text summary for Gemini."""
    today = datetime.now().strftime("%A %B %d %Y")
    try:
        results = tavily.search(
            query=f"{mood} things to do in New York City today {today} events",
            max_results=5,
        )
        lines = []
        for r in results.get("results", []):
            lines.append(f"- {r['title']}: {r['content'][:300]} (source: {r['url']})")
        return "\n".join(lines) if lines else "No live events found."
    except Exception as e:
        print("Tavily error:", e)
        return "No live events found."

def ask_gemini(prompt):
    """Try each model in order, moving on if one is busy or rate-limited."""
    for model in MODELS:
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json"),
            )
            print(f"Plan made with {model}")
            return json.loads(response.text)
        except Exception as e:
            print(f"{model} failed:", str(e)[:120])
            if any(code in str(e) for code in ["503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED"]):
                time.sleep(1)
                continue
            raise
    raise RuntimeError("All models busy")



@app.route("/")
def home():
    return render_template("index.html")


@app.route("/plan", methods=["POST"])
def plan():
    data = request.json
    now = datetime.now().strftime("%A %I:%M %p")
    events = find_events(data["mood"])

    prompt = f"""
You are a friendly NYC local helping someone who is tired of deciding what to do.
It is currently {now}. Their mood is "{data['mood']}" and they have {data['hours']} hours.

Here are live web search results about what's happening in NYC today:
{events}

Suggest 3 to 4 real places in New York City, in a sensible order, that fit the mood,
time of day, and time available. Keep stops close enough to walk or take one subway ride.
If one of the search results is a real event happening today that fits, include it as a
stop and put its source URL in "link". Otherwise set "link" to "".
Only use events that are clearly happening today; ignore anything old or vague.

Respond ONLY with JSON in this exact format:
{{
  "summary": "one short, upbeat sentence describing the plan",
  "stops": [
    {{"name": "place name", "lat": 40.0, "lng": -73.0, "time": "4:00 PM", "why": "one short sentence", "link": ""}}
  ]
}}
"""

    try:
        result = ask_gemini(prompt)
        return jsonify({"plan": result["summary"], "stops": result["stops"]})
    except Exception as e:
        if "429" in str(e):
            msg = "Too many plans too fast! Wait a minute and try again."
        else:
            msg = "NYC is busy right now. Try again in a moment!"
        return jsonify({"plan": msg, "stops": []}), 500


if __name__ == "__main__":
    app.run(debug=True)
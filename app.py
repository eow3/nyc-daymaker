import os
import json
import time
from datetime import datetime

import requests
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from google import genai
from google.genai import types
from tavily import TavilyClient

load_dotenv()
app = Flask(__name__)

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
MODELS = ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite"]

MOODS = ["chill", "adventurous", "social", "creative"]

BUDGETS = {
    "free": "Only suggest free activities (parks, free museums, free events, public spaces).",
    "low": "Keep it cheap, around $20 per person or less for the whole plan.",
    "medium": "A moderate budget is fine, around $50 per person for the whole plan.",
}

GROUPS = {
    "solo": "They are going solo, so pick spots that are comfortable to enjoy alone.",
    "date": "It's a date, so pick romantic or conversation-friendly spots.",
    "friends": "They're with friends, so pick lively, group-friendly spots.",
    "kids": "They have kids with them, so every stop must be family-friendly.",
}

WEATHER_CODES = {
    0: "clear skies", 1: "mostly clear", 2: "partly cloudy", 3: "overcast",
    45: "foggy", 48: "foggy", 51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain", 71: "light snow", 73: "snow",
    75: "heavy snow", 80: "rain showers", 81: "rain showers", 82: "heavy rain showers",
    95: "thunderstorms", 96: "thunderstorms with hail", 99: "thunderstorms with hail",
}


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


def get_weather():
    """Get current NYC weather. Returns a short description for Gemini."""
    try:
        res = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": 40.7128,
                "longitude": -74.0060,
                "current": "temperature_2m,apparent_temperature,weather_code,precipitation",
                "temperature_unit": "fahrenheit",
                "timezone": "America/New_York",
            },
            timeout=5,
        )
        now = res.json()["current"]
        condition = WEATHER_CODES.get(now["weather_code"], "mixed conditions")
        return f"{round(now['temperature_2m'])}°F (feels like {round(now['apparent_temperature'])}°F), {condition}"
    except Exception as e:
        print("Weather error:", e)
        return "unknown"


def ask_gemini(prompt):
    """Try each model in order, moving on if one is busy or rate-limited."""
    for model in MODELS:
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json"),
            )
            print(f"Answered by {model}")
            return json.loads(response.text)
        except Exception as e:
            print(f"{model} failed:", str(e)[:120])
            if any(code in str(e) for code in ["503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED"]):
                time.sleep(1)
                continue
            raise
    raise RuntimeError("All models busy")


def settings_lines(data):
    """Turn the user's choices into safe prompt lines (only allowed values get through)."""
    mood = data.get("mood") if data.get("mood") in MOODS else "chill"
    budget_line = BUDGETS.get(data.get("budget"), BUDGETS["low"])
    group_line = GROUPS.get(data.get("group"), GROUPS["solo"])
    access_line = (
        "Every stop must be wheelchair accessible and step-free. If they need the subway, "
        "only route through stations with elevators, and mention accessibility in 'why'."
        if data.get("accessible") is True else ""
    )
    return mood, budget_line, group_line, access_line


def error_message(e):
    if "429" in str(e):
        return "Too many requests too fast! Wait a minute and try again."
    return "NYC is busy right now. Try again in a moment!"


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/plan", methods=["POST"])
def plan():
    data = request.json
    now = datetime.now().strftime("%A %I:%M %p")
    mood, budget_line, group_line, access_line = settings_lines(data)

    try:
        hours = max(1, min(12, int(data.get("hours", 3))))
    except (TypeError, ValueError):
        hours = 3

    events = find_events(mood)
    weather = get_weather()
    print("Weather:", weather)

    lat, lng = data.get("lat"), data.get("lng")
    if lat and lng and 40.49 <= lat <= 40.92 and -74.27 <= lng <= -73.68:
        location_line = f"They are currently near latitude {lat}, longitude {lng}. Start the plan close to where they are."
    else:
        location_line = "Their location is unknown, so pick a lively, easy-to-reach neighborhood."

    prompt = f"""
You are a friendly NYC local helping someone who is tired of deciding what to do.
It is currently {now}. The weather in NYC right now is {weather}.
{location_line}
Their mood is "{mood}" and they have {hours} hours.
{budget_line}
{group_line}
{access_line}
If it's raining, very cold, or very hot, favor indoor spots. If it's nice out, favor outdoor ones.
Every stop MUST be open at the time you schedule it. Many museums and shops close by 5 or 6 PM,
so for evening plans favor places known to be open late.

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
        return jsonify({"plan": result["summary"], "stops": result["stops"], "weather": weather})
    except Exception as e:
        return jsonify({"plan": error_message(e), "stops": []}), 500


@app.route("/offroute", methods=["POST"])
def offroute():
    data = request.json
    mood, budget_line, group_line, access_line = settings_lines(data)
    stops = data.get("stops", [])[:8]
    names = ", ".join(str(s.get("name", ""))[:80] for s in stops)
    weather = get_weather()
    now = datetime.now().strftime("%A %I:%M %p")

    prompt = f"""
You are a NYC local who knows the city's best-kept secrets.

It is currently {now}. Someone's current plan includes: {names}.
The weather right now is {weather}. Their mood is "{mood}".
The place MUST be open right now and for at least the next hour. Museums, galleries, and
shops often close by 5 or 6 PM, so late in the day favor places known to stay open late.

{budget_line}
{group_line}
{access_line}

Suggest ONE lesser-known hidden gem within a short walk of those stops that most tourists
and even many locals miss. It must be a real place, not a chain, not a famous landmark,
and not already in their plan.

Respond ONLY with JSON in this exact format:
{{"name": "place name", "lat": 40.0, "lng": -73.0, "why": "one short sentence on what makes it special", "link": ""}}
"""

    try:
        gem = ask_gemini(prompt)
        return jsonify({"gem": gem})
    except Exception as e:
        return jsonify({"gem": None, "error": error_message(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
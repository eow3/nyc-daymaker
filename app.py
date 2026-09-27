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
from concurrent.futures import ThreadPoolExecutor

load_dotenv()
app = Flask(__name__)

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
MODELS = ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite"]

MOODS = ["chill", "adventurous", "social", "creative"]

MOOD_HINTS = {
    "chill": "Low-key and unhurried: cozy cafes, quiet parks, bookstores, relaxed views. Minimal walking between stops.",
    "adventurous": "Push the edges: unusual spots, new neighborhoods, active or unexpected experiences they wouldn't normally try.",
    "social": "People and energy: lively spots, markets, events, places good for meeting people or talking with friends.",
    "creative": "Make and discover: galleries, street art, workshops, design shops, live music, anything inspiring.",
}

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

CLOSED_FLAGS = ["permanently closed", "closed permanently", "closed for good", "has closed", "no longer open", "shut down"]

def is_probably_open(name):
    """Ask Tavily whether a place is still operating. Returns False if it looks permanently closed."""
    try:
        r = tavily.search(
            query=f"Is {name} in New York City still open, or is it permanently closed?",
            max_results=2,
            include_answer=True,
        )
        text = (r.get("answer") or "").lower()
        if not text and r.get("results"):
            text = r["results"][0].get("content", "").lower()
        closed = any(flag in text for flag in CLOSED_FLAGS)
        print(f"Open check: {name} -> {'CLOSED' if closed else 'ok'}")
        return not closed
    except Exception as e:
        print("Open check error:", e)
        return True

def verify_stops(stops, backups):
    """Check all stops and backups at the same time. Swap any closed stop for an open backup."""
    candidates = stops + backups

    def check(stop):
        if stop.get("link"):  # came from today's live event search
            return True
        return is_probably_open(stop.get("name", ""))

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(check, candidates))

    stop_ok = results[:len(stops)]
    open_backups = [b for b, ok in zip(backups, results[len(stops):]) if ok]

    final = []
    for stop, ok in zip(stops, stop_ok):
        if ok:
            final.append(stop)
        elif open_backups:
            backup = open_backups.pop(0)
            backup["time"] = stop.get("time", "")
            print(f"Swapped closed '{stop.get('name')}' for '{backup.get('name')}'")
            final.append(backup)
    return final

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

@app.route("/weather")
def weather_route():
    return jsonify({"weather": get_weather()})

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
Their mood is "{mood}": {MOOD_HINTS[mood]} They have {hours} hours.{budget_line}
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

Also include 2 backup stops nearby that fit the same plan, in case a stop turns out to be closed.

Respond ONLY with JSON in this exact format:
{{
  "summary": "one short, upbeat sentence describing the plan",
  "stops": [
    {{"name": "place name", "lat": 40.0, "lng": -73.0, "time": "4:00 PM", "why": "one short sentence", "link": ""}}
  ],
  "backups": [
    {{"name": "place name", "lat": 40.0, "lng": -73.0, "time": "", "why": "one short sentence", "link": ""}}
  ]
}}
"""

    try:
        result = ask_gemini(prompt)
        stops = verify_stops(result.get("stops", [])[:4], result.get("backups", [])[:2])
        return jsonify({"plan": result["summary"], "stops": stops, "weather": weather})
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

    anchor = str(stops[0].get("name", "Manhattan"))[:80] if stops else "Manhattan"
    try:
        search = tavily.search(query=f"hidden gem lesser-known spots near {anchor} New York City 2026", max_results=5)
        found = "\n".join(f"- {r['title']}: {r['content'][:300]}" for r in search.get("results", []))
    except Exception as e:
        print("Tavily error:", e)
        found = "No search results."

    prompt = f"""
You are a NYC local who knows the city's best-kept secrets.
It is currently {now}. Someone's current plan includes: {names}.
The weather right now is {weather}. Their mood is "{mood}".
{budget_line}
{group_line}
{access_line}

Here are recent web results about hidden gems in that area:
{found}

Suggest 3 different lesser-known hidden gems within a short walk of those stops, best first.
Prefer places mentioned in the recent web results above. Each must be a real place that is
currently operating, open right now and for at least the next hour, not a chain, not a famous
landmark, and not already in their plan.

Respond ONLY with JSON in this exact format:
{{"candidates": [
  {{"name": "place name", "lat": 40.0, "lng": -73.0, "why": "one short sentence on what makes it special", "link": ""}}
]}}
"""

    try:
        result = ask_gemini(prompt)
        for gem in result.get("candidates", [])[:3]:
            if is_probably_open(gem.get("name", "")):
                return jsonify({"gem": gem})
        return jsonify({"gem": None, "error": "Couldn't find a hidden gem that's open right now. Try again!"}), 500
    except Exception as e:
        return jsonify({"gem": None, "error": error_message(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
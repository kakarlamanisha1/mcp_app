# server.py
import os
import httpx
from typing import Any, Dict
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
OPENWEATHER_BASE_URL = os.getenv("OPENWEATHER_BASE_URL", "http://api.openweathermap.org/data/2.5")

mcp = FastMCP("weather-assistant")

# CORS - allow your frontend origin(s)
app = FastAPI(title="Weather Assistant API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change this to your specific frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"status": "ok"}

async def _get_weather_data(endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
    if not OPENWEATHER_API_KEY:
        raise ValueError("OPENWEATHER_API_KEY is not set.")
    params["appid"] = OPENWEATHER_API_KEY
    params["units"] = "metric"
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{OPENWEATHER_BASE_URL}/{endpoint}", params=params, timeout=10.0)
        resp.raise_for_status()
        return resp.json()

@mcp.tool()
async def get_current_weather(city: str) -> str:
    try:
        data = await _get_weather_data("weather", {"q": city})
        weather_desc = data["weather"][0]["description"]
        temp = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        return f"Current weather in {city}: {weather_desc}, Temperature: {temp}°C, Humidity: {humidity}%"
    except httpx.HTTPStatusError as e:
        return f"Error fetching weather: {e.response.text}"
    except Exception as e:
        return f"An error occurred: {str(e)}"

@mcp.tool()
async def get_forecast(city: str, days: int = 3) -> str:
    try:
        if days > 5:
            days = 5
        data = await _get_weather_data("forecast", {"q": city})
        forecasts = []
        seen_dates = set()
        for item in data.get("list", []):
            date = item["dt_txt"].split(" ")[0]
            if date not in seen_dates:
                seen_dates.add(date)
                desc = item["weather"][0]["description"]
                temp = item["main"]["temp"]
                forecasts.append(f"{date}: {desc}, {temp}°C")
                if len(forecasts) >= days:
                    break
        return f"Forecast for {city}:\n" + "\n".join(forecasts)
    except httpx.HTTPStatusError as e:
        return f"Error fetching forecast: {e.response.text}"
    except Exception as e:
        return f"An error occurred: {str(e)}"

# mount /mcp (FastMCP provides SSE app)
app.mount("/mcp", mcp.sse_app())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 5000)))

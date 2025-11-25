import os
import httpx
from typing import Any, Dict, List
from fastapi import FastAPI
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# Load environment variables
load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
OPENWEATHER_BASE_URL = os.getenv("OPENWEATHER_BASE_URL", "http://api.openweathermap.org/data/2.5")
# FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
# RAGIE_API_KEY = os.getenv("RAGIE_API_KEY")

# Initialize MCP Server
# FastMCP is an ASGI application that can be run directly or mounted
mcp = FastMCP("weather-assistant")

# Initialize FireCrawl and Ragie clients
# firecrawl_app = None
# ragie_client = None

# try:
#     from firecrawl import FirecrawlApp
#     firecrawl_app = FirecrawlApp(api_key=FIRECRAWL_API_KEY)
# except ImportError:
#     print("FireCrawl library not installed. Install with: pip install firecrawl-py")

# try:
#     from ragie import Ragie
#     ragie_client = Ragie(auth=RAGIE_API_KEY)
# except ImportError:
#     print("Ragie library not installed. Install with: pip install ragie")


async def _get_weather_data(endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Helper to make requests to OpenWeatherMap API."""
    if not OPENWEATHER_API_KEY:
        raise ValueError("OPENWEATHER_API_KEY is not set in environment variables.")
    
    params["appid"] = OPENWEATHER_API_KEY
    params["units"] = "metric"
    
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{OPENWEATHER_BASE_URL}/{endpoint}", params=params)
        response.raise_for_status()
        return response.json()

@mcp.tool()
async def get_current_weather(city: str) -> str:
    """Get the current weather for a specific city.
    
    Args:
        city: The name of the city (e.g., 'London', 'New York').
    """
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
    """Get the weather forecast for a specific city.
    
    Args:
        city: The name of the city.
        days: Number of days to forecast (default 3, max 5 for free tier usually).
    """
    try:
        # Note: Standard free API is 5 day / 3 hour forecast
        data = await _get_weather_data("forecast", {"q": city})
        
        forecasts = []
        seen_dates = set()
        for item in data["list"]:
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

# @mcp.tool()
# async def scrape_weather_news(url: str) -> str:
#     """Scrape weather-related news or information from a given URL using FireCrawl.
    
#     Args:
#         url: The URL of the webpage to scrape for weather information.
#     """
#     if not firecrawl_app:
#         return "FireCrawl is not configured. Please check your installation and API key."
    
#     try:
#         # Scrape the URL
#         print(f"Attempting to scrape: {url}")  # Debug line
#         scrape_result = firecrawl_app.scrape(url=url)
#         print(f"Scrape successful: {type(scrape_result)}")  # Debug line
#         return f"Scraped content from {url}: {scrape_result}"
#     except Exception as e:
#         print(f"Scraping error: {str(e)}")  # Debug line
#         return f"Error scraping {url}: {str(e)}"

# @mcp.tool()
# async def retrieve_weather_knowledge(query: str) -> str:
#     """Retrieve enhanced weather information using Ragie RAG.
    
#     Args:
#         query: The weather-related query to search for in the knowledge base.
#     """
#     if not ragie_client:
#         return "Ragie is not configured. Please check your installation and API key."
    
#     try:
#         # Retrieve relevant information using Ragie
#         # This is a placeholder - actual implementation depends on how you've set up your Ragie knowledge base
#         # For now, we'll use a simple search method
#         search_result = ragie_client.search(query=query)
#         return f"Retrieved knowledge for '{query}': {search_result}"
#     except Exception as e:
#         return f"Error retrieving knowledge for '{query}': {str(e)}"

# Create main FastAPI app and mount MCP
app = FastAPI(title="Weather Assistant API")

# Mount the MCP app
# Note: FastMCP is an ASGI app. We mount it at /mcp
app.mount("/mcp", mcp.sse_app())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


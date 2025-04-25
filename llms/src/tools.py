
# Thanks to Bex T. for explaining how an agent can use the Open Weather API
# https://www.datacamp.com/tutorial/openai-agents-sdk-tutorial

import os
import asyncio
import requests
from agents import Agent, Runner, function_tool

# Agent assumes the environment variable OPENAI_API_KEY exists


@function_tool
def weather_forecast(lat: float, lon: float) -> str:
    '''
    Returns the weather at the specified coordinates using OpenWeatherMap API.
    '''

    print(f"Checking weather at {lat}°N, {lon}° W")
    OPENWEATHERMAP_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY")    
    url = (
        f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}"
        f"&appid={OPENWEATHERMAP_API_KEY}&units=metric"
    )

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        weather_report = (
            f"The weather forecast for {data['name']} is:\n"
            f"- Temperature: {data['main']['temp']}°C "
            f"(feels like {data['main']['feels_like']}°C)\n"
            f"- Conditions: {data['weather'][0]['description']}\n"
            f"- Humidity: {data['main']['humidity']}%\n"
            f"- Wind speed: {data['wind']['speed']} m/s\n"
            f"- Pressure: {data['main']['pressure']} hPa\n"
            f"- Visibility: {data.get('visibility')}\n"
        )
        return weather_report

    except requests.exceptions.RequestException as e:
        return f"Error fetching weather data: {str(e)}"
   

# Create a weather assistant
weather_assistant = Agent(
    name="Weather Assistant",
    instructions=(
        "You are a weather assistant that will use an external API to provide "
        "helpful weather predictions. Always use the weather_forecast tool to "
        "obtain accurate data. If the requested location is ambiguous, always "
        "use the most common location. If you cannot understand the user's "
        "prompt, revert to Irvine, CA. The weather_forecast tool will always "
        "give data in the metric system; you must always convert to standard "
        "English units. Do not return a bulleted list, instead provide a " 
        "short description that includes helpful commentary such as clothing "
        "suggestions or activity recommendations based on the conditions."
    ),
    model='gpt-4o-mini',
    tools=[weather_forecast]
)

async def get_weather_report(prompt):
    weather_report = await Runner.run(weather_assistant, prompt)
    return weather_report.final_output


async def main():
    prompt1 = "What is the weather right now in Las Vegas?"
    prompt2 = "What is the weather in the capital of Canada?"

    weather_report1, weather_report2 = await asyncio.gather(
        get_weather_report(prompt1),
        get_weather_report(prompt2)
    )

    print(f"\n{prompt1}\n")
    print(f"{weather_report1}\n")

    print(f"\n{prompt2}\n")
    print(f"{weather_report2}\n")


if __name__ == '__main__':
    asyncio.run(main())

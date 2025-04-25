
import asyncio
from agents import Agent, Runner, handoff, function_tool


@function_tool
def get_weather(location: str, date: str) -> str:
    return f"The weather in {location} on {date} will be sunny with highs around 75°F."

@function_tool
def check_flights(origin: str, destination: str, date: str) -> str:
    return f"Flights from {origin} to {destination} on {date} start at $249 round-trip."

@function_tool
def get_rental_car_prices(location: str, date: str, duration_days: int) -> str:
    return f"Economy cars in {location} for {duration_days} days starting on {date} start at $45/day."


logistics_agent = Agent(
    name="Vacation Logistics Planner",
    instructions=(
        "You are a helpful travel assistant. Given a travel request, you must call tools to check"
        " the weather, lookup flights, and find rental car options. Do not answer other questions."
    ),
    tools=[ get_weather, check_flights, get_rental_car_prices ]
)

destination_agent = Agent(
    name="Destination Expert",
    instructions=(
        "You are a helpful travel assistant. Given a travel request, you must cheerfully answer"
        " questions relating to the history, culture, politics, and tourist information."
    )
)

vacation_agent = Agent(
    name="Vacation Assistant",
    instructions=(
        "You are a helpful travel assistant. Given a travel request, you may call tools to help"
        " the user. If the question is about logistics such as flights, weather, rental cars,"
        " etc., hand it off to the logistics agent. If the question is about culture, history,"
        " or attractions, hand it off to the destination expert agent. Do not answer any questions"
        " yourself."
    ),
    handoffs=[ handoff(logistics_agent), handoff(destination_agent) ]
)


async def run_vacation_agent(prompt):
    return await Runner.run(vacation_agent, prompt)


async def main():
    print("\nWelcome to the vacation planning program")
    destination = input("Where would you like to go? ")
    prompt = f"Help me plan a vacation to {destination} on May 1st for 5 days."
    while prompt.lower() not in ['quit', 'exit']:
        if len(prompt.strip()) > 0:
            response = await run_vacation_agent(prompt)
            print("\nAssistant:\n" + response.final_output)
        prompt = input("\n> ")
    print()


if __name__ == "__main__":
    asyncio.run(main())

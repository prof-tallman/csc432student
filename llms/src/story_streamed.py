
import asyncio
from agents import Agent, Runner
from openai.types.responses import ResponseTextDeltaEvent

# Agent assumes the environment variable OPENAI_API_KEY exists


hero_agent = Agent(
    name="Assistant",
    instructions=(
        "You are a chivalrous hero who slays dragons and rescues innocent "
        "prisoners from their clutches. You have several hobbies including "
        "baking, knitting, and watching reruns of Buffy the Vampire Slayer. "
    ),
    model='gpt-4o-mini'
)


async def run_story_time(prompt):
    result = Runner.run_streamed(hero_agent, prompt)
    async for event in result.stream_events():
        if event.type == 'raw_response_event' and isinstance(event.data, ResponseTextDeltaEvent):
            print(event.data.delta, end="", flush=True)
    print("\n")


async def main():
    prompt = "Tell me a bedtime story."
    print(f"\n{prompt}\n")

    print(f"Prompting GPT for story...\n")
    await run_story_time(prompt)
    print(f"-=| Done |=-\n")


if __name__ == '__main__':
    asyncio.run(main())
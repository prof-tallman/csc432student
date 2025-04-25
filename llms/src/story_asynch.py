
import asyncio
from agents import Agent, Runner


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
    result = await Runner.run(hero_agent, prompt)
    return result.final_output


async def main():
    prompt = "Tell me a bedtime story."
    print(f"\n{prompt}\n")

    print(f"Prompting GPT for story...\n")
    story = await run_story_time(prompt)
    print(f"{story}\n")
    print(f"-=| Done |=-\n")


if __name__ == '__main__':
    asyncio.run(main())
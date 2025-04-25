
import asyncio
from pydantic import BaseModel
from agents import (
    Agent, 
    Runner,
    input_guardrail, 
    output_guardrail,
    GuardrailFunctionOutput,
    OutputGuardrailTripwireTriggered,
)
from openai.types.responses import ResponseTextDeltaEvent

# Agent assumes the environment variable OPENAI_API_KEY exists


class GuardrailOutput(BaseModel):
    is_triggered: bool
    reasoning: str

politics_agent = Agent(
    name="Politics Check",
    instructions="Check if the user is asking you about US political opinions",
    output_type=GuardrailOutput,
)

sports_agent = Agent(
    name="Sports Check",
    #instructions="Evaluate whether the discussion involves sports",
    instructions="Evaluate whether the statement is negative",
    output_type=GuardrailOutput,
)

@input_guardrail
async def politics_guardrail(ctx, agent, prompt):
    response = await Runner.run(starting_agent=politics_agent, input=prompt)
    return GuardrailFunctionOutput(
        output_info=response.final_output,
        tripwire_triggered=response.final_output.is_triggered,
    )

@output_guardrail
async def anger_guardrail(ctx, agent, prompt):
    response = await Runner.run(starting_agent=sports_agent, input=prompt)
    return GuardrailFunctionOutput(
        output_info=response.final_output,
        tripwire_triggered=response.final_output.is_triggered
    )

family_agent = Agent(
    name="The Family",
    instructions=(
        "The setting is a large family's Thanksgiving dinner celebration. In this family there is "
        "a wonderful young lady who has just brought home her boyfriend to meet everyone for the "
        "very first time. Answer every prompt as the stern, disapproving father who believes that "
        "there is no man who will ever be worthy of his daughter's affections. Always start with "
        "'Father: ' so that we can follow the discussion."
    ),
    model="gpt-4o-mini",
    input_guardrails=[politics_guardrail],
    output_guardrails=[anger_guardrail],
)

async def ask_the_family(prompt):
    result = await Runner.run(starting_agent=politics_agent, input=prompt)
    if result.final_output.is_triggered:
        print(f"\n[You suddenly feel a sharp pain in your side from your girlfriend's elbow]")
        print(f"[Comment: {result.final_output.reasoning}]")
        print(f"[Don't worry. She quickly redirected the conversation to a safer opic.]\n")
        prompt = "\nDaughter: So, how 'bout them Cowboys?"
    
    try:
        response = Runner.run_streamed(
            starting_agent=family_agent,
            input=prompt
        )
        async for event in response.stream_events():
            if (event.type == "raw_response_event" and
                    isinstance(event.data, ResponseTextDeltaEvent)):
                print(event.data.delta, end="", flush=True)

    except OutputGuardrailTripwireTriggered as e:
        print(f"\n\nDaughter: Daddy--be nice, I love him!\n")
        print(f"\n[Comment: {e.guardrail_result.output.output_info.reasoning}]")

async def main():
    prompt1 = (
        "What is my job? That is a great question, thank you for asking. I am a professional "
        "juggler in a travelling circus... and I think the world of your daughter."
    )
    print(f"\nBoyfriend: {prompt1}\n")
    await ask_the_family(prompt1)
    print(f"\n")

    prompt2 = (
        "What do you think of Donald Trump?"
    )
    print(f"\nBoyfriend: {prompt2}\n")
    await ask_the_family(prompt2)
    print(f"\n")


if __name__ == "__main__":
    asyncio.run(main())
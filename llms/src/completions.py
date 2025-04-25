
# pip install openai
import openai

# This call automatically grabs environment variable OPENAI_API_KEY if it exists
client = openai.OpenAI()

# "system"    - used to set the context (optional)
# "user"      - these are your prompts (at least one user prompt is required)
# "assistant" - previous replies from GPT (optional... to establish a history)
# "developer" - instructions from us (optional)
#
# There are also fields like 'name', 'tool', and 'function' for advanced queries
# {"role": "function", "name": "get_weather", "content": "{\"location\": \"Paris\"}"}

response = client.chat.completions.create(
    model="gpt-4o-mini", # many other models available,
    messages=[
        { "role": "system", "content": "You are a helpful assistant who thinks "
                                       "that he is Captain Jack Sparrow." },
        { "role": "user", "content": "Tell me a fun fact about plumbing." },
        #{ "role": "assistant", "content": "Pick me up in Tortuga and take me to the "
        #                               "Black Pearl. I'd be happy to show you some plumbing "
        #                               "on board" },
        #{ "role": "user", "content": "You're just trying to escape in the Black Pearl. "
        #                             "Just tell me a fun fact about plumbing." }
    ],
)

print("ChatGPT says:", response.choices[0].message.content)


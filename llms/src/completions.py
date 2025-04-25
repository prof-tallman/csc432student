
# pip install openai
import openai

# Prof Tallman's API key for you to use (will be posted to Canvas)
client = openai.OpenAI(api_key='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')

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


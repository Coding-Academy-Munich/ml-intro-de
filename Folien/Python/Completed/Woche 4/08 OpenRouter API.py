# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>OpenRouter API</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %%
import json
import os
import requests
from loguru import logger
from dataclasses import dataclass
from textwrap import wrap

# %%
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or "No API Key Found"


# %%
@dataclass
class Message:
    role: str
    content: str

# %%
@dataclass
class Model:
    id: str
    slug: str

# %%
models = {
    "claude": Model(id="anthropic/claude-3.5-sonnet:beta", slug="claude"),
    "phi": Model(id="microsoft/phi-3.5-mini-128k-instruct", slug="phi"),
    "llama": Model(id="meta-llama/llama-3.1-8b-instruct", slug="llama"),
    "qwen": Model(id="qwen/qwen-2.5-72b-instruct", slug="qwen"),
    # "gemini": Model(id="google/gemini-pro-1.5", slug="gemini"),
    "chatgpt": Model(id="openai/chatgpt-4o-latest", slug="chatgpt"),
}


# %%
@dataclass
class OpenRouterProvider:
    api_key: str = OPENROUTER_API_KEY
    api_url: str = OPENROUTER_API_URL

    def send_messages(self, messages: list[Message], model: Model) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        data = json.dumps(
            {
                "model": model.id,
                "messages": [message.__dict__ for message in messages],
            }
        )

        response = requests.post(url=self.api_url, headers=headers, data=data)

        if response.status_code == 200:
            response_json = response.json()
            logger.trace(
                f"Received response message from {model.slug}: "
                f"{response_json['choices'][0]['message']['content']}"
            )
            return response_json["choices"][0]["message"]["content"]
        else:
            raise RuntimeError(f"Failed to convert chunk: {response.text}")

    def chat(self, model: Model) -> list[Message]:
        messages = [Message(role="system", content="You are a helpful assistant.")]
        indent = " " * 2
        while True:
            user_input = input("User: ")
            if user_input.lower() == "exit":
                break
            print(f"\nUser:\n{format_text(user_input)}", flush=True)
            messages.append(Message(role="user", content=user_input))
            response = self.send_messages(messages, model)
            messages.append(Message(role="assistant", content=response))
            print(f"\nResponse:\n{format_text(response)}", flush=True)
        return messages

    def chat_non_interactively(self, model: Model, user_text: list[str]) -> list[Message]:
        user_text = user_text.copy()
        messages = [Message(role="system", content="You are a helpful assistant.")]
        indent = " " * 2
        while user_text:
            content = user_text.pop(0)
            print(f"\nUser:\n{format_text(content)}", flush=True)
            messages.append(Message(role="user", content=content))
            response = self.send_messages(messages, model)
            messages.append(Message(role="assistant", content=response))
            print(f"\nResponse:\n{format_text(response)}", flush=True)
        return messages


# %%
import re
from textwrap import wrap

def format_text(text: str, width: int = 80, indent: int = 2) -> str:
    lines = text.split('\n')
    formatted_lines = []
    in_code_block = False
    code_block: list[str] = []

    for line in lines:
        if line.strip().startswith('```'):
            if in_code_block:
                # End of code block
                formatted_lines.extend([' ' * indent + l for l in code_block])
                formatted_lines.append(' ' * indent + line)
                formatted_lines.append('')  # Add empty line after code block
                in_code_block = False
                code_block = []
            else:
                # Start of code block
                if formatted_lines:
                    formatted_lines.append('')  # Add empty line before code block
                formatted_lines.append(' ' * indent + line)
                in_code_block = True
        elif in_code_block:
            code_block.append(line)
        else:
            # Regular text, wrap it
            wrapped = wrap(line, width - indent, initial_indent=' ' * indent, subsequent_indent=' ' * indent)
            formatted_lines.extend(wrapped)

    # Handle any remaining code block
    if code_block:
        formatted_lines.extend([' ' * indent + l for l in code_block])

    return '\n'.join(formatted_lines)


# %%
EXAMPLE_MESSAGES = [
    Message(role="system", content="Hello, what can I help you with?"),
    Message(role="user", content="What is the capital of Bavaria?"),
]

# %%
USER_INPUTS = [
    "What is the capital of Bavaria?",
    "Tell me more about it.",
    "Write a Python program that prints the numbers from 1 to 100.",
    "Do the same, but for multiples of three print 'Fizz' instead of the number "
    "for the multiples of five print 'Buzz', and for multiples of "
    "15 print 'FizzBuzz'.",
    "If there are three killers in a room and somebody enters the room and shoots "
    "one of the killers dead, how many killers are left in the room?",
]

# %%
if __name__ == "__main__":
    provider = OpenRouterProvider()
    for slug in ["phi", "qwen"]:
        model = models[slug]
        print("=" * 80)
        print(f"Model: {slug}")
        provider.chat_non_interactively(model, USER_INPUTS)
        print()

# %%

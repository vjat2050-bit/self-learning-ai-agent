import os
import json

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MODEL_NAME = "gemini-3.6-flash"
MEMORY_FILE = "memory.json"


def load_memory():
    if not os.path.exists(MEMORY_FILE):
        return []
    with open(MEMORY_FILE, "r") as file:
        try:
            return json.load(file)
        except json.JSONDecodeError:
            return []


def save_memory(memory):
    with open(MEMORY_FILE, "w") as file:
        json.dump(memory, file, indent=4)


def calculator(expression: str) -> str:
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception:
        return "Calculation error"


calculator_tool = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="calculator",
            description="Calculate a mathematical expression.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "expression": types.Schema(
                        type="STRING",
                        description="The mathematical expression to calculate."
                    )
                },
                required=["expression"]
            )
        )
    ]
)

generation_config = types.GenerateContentConfig(
    tools=[calculator_tool]
)


def main():
    MAX_TOOL_CALLS = 3
    memory = load_memory()
    history = []

    memory_context = "\n".join(memory)

    print("Self-Learning AI Agent")
    print("Type 'exit' to quit, or 'remember: <something>' to save a memory.")
    print("Calculator tool: enabled")
    print("Persistent memory: enabled")
    print("-" * 40)

    if memory:
        print(f"Loaded {len(memory)} memory item(s) from {MEMORY_FILE}.")

    while True:
        user_input = input("\nYou: ")

        if user_input.lower() == "exit":
            print("Agent stopped.")
            break

        if user_input.lower().startswith("remember:"):
            fact = user_input[9:].strip()

            if fact:
                memory.append(fact)
                save_memory(memory)
                print("Agent: I'll remember that.")
            else:
                print("Agent: Please tell me what to remember after 'remember:'.")

            continue

        history.append(
            types.Content(role="user", parts=[types.Part.from_text(text=user_input)])
        )

        response = client.models.generate_content(
    model=MODEL_NAME,
    contents=[
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(
                    text=f"""
You are a helpful AI agent.

Here are the user's saved memories:
{memory_context}

Use these memories when they are relevant to the user's question.
Do not mention the memory system unless the user asks about it.

User's current message:
{user_input}
"""
                )
            ]
        )
    ],
    config=generation_config
)

        if response.function_calls:
            if len(response.function_calls) > MAX_TOOL_CALLS:
                print("Agent: Too many tool calls requested. Stopping for safety.")
                continue
            history.append(response.candidates[0].content)

            for call in response.function_calls:
                print("Agent selected tool:", call.name)
                print("Tool arguments:", call.args)

                if call.name == "calculator":
                    result = calculator(call.args["expression"])
                    print("Tool result:", result)

                    history.append(
    types.Content(
        role="user",
        parts=[
            types.Part.from_function_response(
                name="calculator",
                response={
                    "result": result
                }
            )
        ]
    )
)

            tool_response = client.models.generate_content(
                model=MODEL_NAME,
                contents=history,
                config=generation_config
            )

            history.append(tool_response.candidates[0].content)
            print("Agent:", tool_response.text)

        else:
            history.append(response.candidates[0].content)
            print("Agent:", response.text)


if __name__ == "__main__":
    main()
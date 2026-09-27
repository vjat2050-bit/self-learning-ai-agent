import os
import json
import streamlit as st

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

    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)
            return data if isinstance(data, list) else []
    except Exception:
        return []


def save_memory(memory):
    with open(MEMORY_FILE, "w", encoding="utf-8") as file:
        json.dump(memory, file, indent=4, ensure_ascii=False)


def calculator(expression):
    try:
        allowed = set("0123456789+-*/(). %")

        if not all(char in allowed for char in expression):
            return "Calculation error"

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
                        description="Mathematical expression to calculate."
                    )
                },
                required=["expression"]
            )
        )
    ]
)


config = types.GenerateContentConfig(
    tools=[calculator_tool]
)


st.set_page_config(
    page_title="Self-Learning AI Agent",
    page_icon="🤖"
)

st.title("🤖 Self-Learning AI Agent")
st.caption("Gemini-powered agent with persistent memory and tool calling.")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = load_memory()


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


user_input = st.chat_input("Ask something...")


if user_input:

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.write(user_input)

    # MEMORY COMMAND
    if user_input.lower().startswith("remember:"):

        fact = user_input[len("remember:"):].strip()

        if not fact:
            answer = "Please tell me what you want me to remember."

        elif fact in st.session_state.memory:
            answer = "I already remember that."

        else:
            st.session_state.memory.append(fact)
            save_memory(st.session_state.memory)
            answer = "I'll remember that. ✅"

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

        with st.chat_message("assistant"):
            st.write(answer)

    # NORMAL AI MESSAGE
    else:

        memory_context = "\n".join(st.session_state.memory)

        prompt = f"""
You are a helpful AI agent.

Saved memories:
{memory_context}

Use the memories when relevant.

User message:
{user_input}
"""

        try:

            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=config
            )

            # TOOL CALL
            if response.function_calls:

                call = response.function_calls[0]

                if call.name == "calculator":

                    expression = call.args["expression"]
                    result = calculator(expression)

                    tool_response = types.Part.from_function_response(
                        name="calculator",
                        response={
                            "result": result
                        }
                    )

                    final_response = client.models.generate_content(
                        model=MODEL_NAME,
                        contents=[
                            types.Content(
                                role="user",
                                parts=[
                                    types.Part.from_text(text=prompt)
                                ]
                            ),
                            response.candidates[0].content,
                            types.Content(
                                role="user",
                                parts=[tool_response]
                            )
                        ],
                        config=config
                    )

                    answer = final_response.text

                else:
                    answer = "Unknown tool requested."

            else:
                answer = response.text

        except Exception as error:
            answer = f"Error: {error}"

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

        with st.chat_message("assistant"):
            st.write(answer)
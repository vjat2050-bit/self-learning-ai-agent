# Self-Learning AI Agent 🤖

A Python-based AI agent built with Google's Gemini API.

## Features

- 💬 Conversational AI using Gemini
- 🔧 Automatic calculator tool selection
- 🧠 Persistent memory using JSON
- 💾 Save user information with `remember:`
- 🛡️ Tool-call safety limit
- 🖥️ Interactive command-line interface

## Architecture

```text
User
  ↓
Gemini AI Agent
  ↓
Decision
  ↓
┌───────────────┐
│               │
Tool          Memory
│               │
Calculator    memory.json
│               │
└───────┬───────┘
        ↓
    Final Response
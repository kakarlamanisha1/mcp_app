# agent_client.py
import streamlit as st
import asyncio
import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from mcp.client.sse import sse_client
from mcp.client.session import ClientSession
from mcp.types import TextContent

load_dotenv()

# IMPORTANT: include /mcp at the end
SERVER_URL = os.getenv("MCP_SERVER_URL", "https://mcp-app.onrender.com/mcp")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

st.set_page_config(page_title="Weather Agent (MCP + LLM)", page_icon="🤖")
st.title("🤖 Weather Agent")

if not OPENROUTER_API_KEY:
    st.error("OPENROUTER_API_KEY not found in environment variables.")
    st.stop()

client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY)

if "messages" not in st.session_state:
    st.session_state.messages = []

# small keep-alive so Render free-tier doesn't immediately sleep the UI
st.markdown("""
<script>
setInterval(() => fetch(window.location.href), 20000);
</script>
""", unsafe_allow_html=True)

async def call_llm_in_thread(**kwargs):
    """Run the synchronous OpenAI client in a thread to avoid blocking the event loop."""
    def sync_call():
        return client.chat.completions.create(**kwargs)
    return await asyncio.to_thread(sync_call)

async def run_agent_turn(user_input: str):
    """Run a turn of the agent loop."""
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Health-check backend before connecting to SSE
    try:
        import httpx
        resp = httpx.get(SERVER_URL.replace("/mcp", "/health"), timeout=5.0)
        if resp.status_code != 200:
            st.error(f"Backend health check failed: {resp.status_code}")
            return
    except Exception as e:
        st.error(f"Backend health check error: {e}")
        return

    try:
        # sse_client yields (send, recv) - use tuple unpacking
        async with sse_client(SERVER_URL) as (send, recv):
            async with ClientSession(send, recv) as session:
                await session.initialize()

                # Get MCP tools
                mcp_tools = await session.list_tools()

                # Convert MCP tools to OpenAI tool format
                openai_tools = []
                for tool in mcp_tools.tools:
                    # tool.inputSchema should already be a JSON Schema-like object
                    openai_tools.append({
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description or "",
                            "parameters": tool.inputSchema,
                            # "strict" may be required by some tool-call implementations
                            "strict": True
                        }
                    })

                # Prepare messages for LLM
                # Normalize messages to dicts acceptable by OpenRouter/OpenAI wrapper
                llm_messages = []
                for m in st.session_state.messages:
                    if isinstance(m, dict):
                        llm_messages.append(m)
                    else:
                        # In case some objects slipped in; try to coerce
                        try:
                            llm_messages.append({
                                "role": getattr(m, "role", "assistant"),
                                "content": getattr(m, "content", str(m))
                            })
                        except Exception:
                            llm_messages.append({"role": "assistant", "content": str(m)})

                # Call LLM in a thread to avoid blocking
                response = await call_llm_in_thread(
                    model="openai/gpt-4o-mini",
                    messages=llm_messages,
                    tools=openai_tools,
                    tool_choice="auto"
                )

                assistant_msg = response.choices[0].message

                # If LLM requests a tool call
                if getattr(assistant_msg, "tool_calls", None):
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": getattr(assistant_msg, "content", "")
                    })

                    for tool_call in assistant_msg.tool_calls:
                        tool_name = tool_call.function.name
                        try:
                            tool_args = json.loads(tool_call.function.arguments or "{}")
                        except Exception:
                            tool_args = {}

                        # Call MCP tool with robust error handling
                        try:
                            result = await session.call_tool(tool_name, tool_args)
                        except Exception as te:
                            # Append tool error to history instead of crashing
                            err_text = f"Tool {tool_name} call failed: {str(te)}"
                            st.session_state.messages.append({
                                "role": "tool",
                                "tool_call_id": getattr(tool_call, "id", None),
                                "content": err_text
                            })
                            continue

                        # Format the tool output
                        output_text = ""
                        if hasattr(result, "content"):
                            for content in result.content:
                                if isinstance(content, TextContent):
                                    output_text += content.text + "\n"
                                else:
                                    # fallback
                                    output_text += str(content) + "\n"
                        else:
                            output_text = str(result)

                        st.session_state.messages.append({
                            "role": "tool",
                            "tool_call_id": getattr(tool_call, "id", None),
                            "content": output_text
                        })

                    # Finalize: ask LLM to produce final assistant response using updated history
                    # Normalize messages again
                    final_msgs = []
                    for msg in st.session_state.messages:
                        if isinstance(msg, dict):
                            final_msgs.append(msg)
                        else:
                            try:
                                final_msgs.append({
                                    "role": getattr(msg, "role", "assistant"),
                                    "content": getattr(msg, "content", str(msg))
                                })
                            except Exception:
                                final_msgs.append({"role": "assistant", "content": str(msg)})

                    response_final = await call_llm_in_thread(
                        model="openai/gpt-4o-mini",
                        messages=final_msgs
                    )
                    final_content = response_final.choices[0].message.content
                    st.session_state.messages.append({"role": "assistant", "content": final_content})

                else:
                    # No tool calls: normal assistant reply
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": assistant_msg.content if hasattr(assistant_msg, "content") else str(assistant_msg)
                    })

    except Exception as e:
        # Show error to user and append to history for debugging
        st.error(f"Connection / agent runtime error: {e}")
        st.session_state.messages.append({"role": "assistant", "content": f"[agent error] {e}"})

# Streamlit chat rendering
for msg in st.session_state.messages:
    role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", "assistant")
    content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", str(msg))
    if role == "tool":
        with st.chat_message("tool"):
            st.text(f"Tool Output: {content}")
    elif role == "assistant" and content:
        with st.chat_message("assistant"):
            st.markdown(content)
    elif role == "user":
        with st.chat_message("user"):
            st.markdown(content)

if prompt := st.chat_input("Ask about the weather..."):
    # Execute the async turn safely
    asyncio.run(run_agent_turn(prompt))
    st.rerun()

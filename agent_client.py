import streamlit as st
import asyncio
import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from mcp.client.sse import sse_client
from mcp.client.session import ClientSession
from mcp.types import CallToolResult, TextContent


# Load environment variables
load_dotenv()
# Configuration
SERVER_URL = "http://localhost:8000/mcp/sse"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

st.set_page_config(page_title="Weather Agent (MCP + LLM)", page_icon="🤖")
st.title("🤖 Weather Agent")

if not OPENROUTER_API_KEY:
    st.error("OPENROUTER_API_KEY not found in .env")
    st.stop()

client = OpenAI(
    base_url=OPENROUTER_BASE_URL,
    api_key=OPENROUTER_API_KEY,
)

if "messages" not in st.session_state:
    st.session_state.messages = []

async def run_agent_turn(user_input: str):
    """Run a turn of the agent loop."""
    
    # 1. Append user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # 2. Connect to MCP server to get tools
    try:
        async with sse_client(SERVER_URL) as streams:
            async with ClientSession(streams[0], streams[1]) as session:
                await session.initialize()
                
                # Get tools from MCP server
                mcp_tools = await session.list_tools()
                
                # Convert MCP tools to OpenAI tool format
                openai_tools = []
                for tool in mcp_tools.tools:
                    openai_tools.append({
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.inputSchema
                        }
                    })
                
                # 3. Call LLM with tools
                # We need to convert session messages to OpenAI format
                llm_messages = []
                for m in st.session_state.messages:
                    if isinstance(m, dict):
                        llm_messages.append(m)
                    else:
                        # It's a ChatCompletionMessage object
                        llm_messages.append(m)
                
                response = client.chat.completions.create(
                    model="openai/gpt-4o-mini", # Or any other model on OpenRouter
                    messages=llm_messages,
                    tools=openai_tools,
                    tool_choice="auto"
                )
                
                assistant_msg = response.choices[0].message
                
                # 4. Handle tool calls
                if assistant_msg.tool_calls:
                    # Add assistant message with tool calls to history
                    st.session_state.messages.append(assistant_msg)
                    
                    for tool_call in assistant_msg.tool_calls:
                        tool_name = tool_call.function.name
                        tool_args = json.loads(tool_call.function.arguments)
                        
                        # Call MCP tool
                        result = await session.call_tool(tool_name, tool_args)
                        
                        # Format result
                        output_text = ""
                        if hasattr(result, 'content'):
                            for content in result.content:
                                if isinstance(content, TextContent):
                                    output_text += content.text + "\n"
                        else:
                            output_text = str(result)
                            
                        # Add tool output to history
                        st.session_state.messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": output_text
                        })
                    
                    # 5. Get final response from LLM
                    # Re-construct messages for second pass
                    # Note: OpenAI library handles object messages well, but we need to be careful with storage
                    # For simplicity in Streamlit state, we might need to serialize/deserialize or keep objects
                    # Let's just do a second call with the updated history
                    
                    # We need to reconstruct the full list of messages for the API
                    # including the tool calls and tool outputs we just added.
                    # Since we stored them in st.session_state.messages, we can just use that.
                    # BUT, st.session_state.messages now contains objects (assistant_msg) and dicts.
                    # We need to normalize.
                    
                    final_messages = []
                    for msg in st.session_state.messages:
                        if isinstance(msg, dict):
                            final_messages.append(msg)
                        else:
                            # It's an object from OpenAI
                            final_messages.append(msg)

                    response_final = client.chat.completions.create(
                        model="openai/gpt-4o-mini",
                        messages=final_messages
                    )
                    
                    final_content = response_final.choices[0].message.content
                    st.session_state.messages.append({"role": "assistant", "content": final_content})
                    
                else:
                    # No tool call, just text
                    st.session_state.messages.append({"role": "assistant", "content": assistant_msg.content})

    except Exception as e:
        st.error(f"Error: {str(e)}")

# Chat Interface
for msg in st.session_state.messages:
    role = msg["role"] if isinstance(msg, dict) else msg.role
    content = msg["content"] if isinstance(msg, dict) else msg.content
    
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
    asyncio.run(run_agent_turn(prompt))
    st.rerun()

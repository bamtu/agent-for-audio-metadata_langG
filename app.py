import streamlit as st
from dotenv import load_dotenv
import os

from utils.utils import init_vector_store
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from nodes import (
    get_llm,
    retrieve_node,
    tool_node,
    tool_executor,
    route_after_tool_choice
)

# Page config
st.set_page_config(
    page_title="Audio Metadata Agent",
    page_icon="🎵",
    layout="wide"
)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.app = None
    st.session_state.thread_config = {"configurable": {"thread_id": "1"}}
    st.session_state.messages = []
    st.session_state.pending_approval = False
    st.session_state.pending_tool_calls = None

def initialize_app():
    """Initialize the LangGraph app"""
    load_dotenv()

    # Set folder path
    folder_path = "C:/music_files"

    # Initialize LLM
    llm = get_llm()

    # Initialize vector store with retriever
    with st.spinner("Initializing vector store..."):
        init_vector_store(folder_path=folder_path, llm=llm)

    # Create LangGraph workflow
    flow = StateGraph(MessagesState)

    # Add nodes
    flow.add_node("retrieve", retrieve_node)
    flow.add_node("tool", tool_node)
    flow.add_node("tool_executor", tool_executor)

    # Set entry point
    flow.set_entry_point("retrieve")

    # Add edges
    flow.add_edge("retrieve", "tool")

    # Add conditional edge
    flow.add_conditional_edges(
        "tool",
        route_after_tool_choice,
        {
            "tool_executor": "tool_executor",
            "end": END
        }
    )

    flow.add_edge("tool_executor", END)

    # Compile without checkpointer
    memory = MemorySaver()
    app = flow.compile(
        checkpointer=memory,
        interrupt_before=["tool_executor"]
    )

    return app

# Title
st.title("🎵 Audio Metadata Agent")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("설정")

    if st.button("Initialize Agent (Vector store initialization)", type="primary"):

        try:
            st.session_state.app = initialize_app()
            st.session_state.initialized = True
            st.success("Agent initialized successfully!")
        except Exception as e:
            st.error(f"Error initializing agent: {e}")

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.pending_approval = False
        st.session_state.pending_tool_calls = None
        st.rerun()

# Main chat interface
if not st.session_state.initialized:
    st.info("👈 Please initialize the agent using the sidebar button.")
else:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Show pending approval if exists
    if st.session_state.pending_approval and st.session_state.pending_tool_calls:
        st.warning("⚠️ Tool execution pending approval")

        with st.expander("Tool Calls Details", expanded=True):
            for i, tool_call in enumerate(st.session_state.pending_tool_calls, 1):
                st.markdown(f"**{i}. Tool:** `{tool_call['name']}`")
                st.json(tool_call['args'])

        col1, col2 = st.columns(2)

        with col1:
            if st.button("✅ Approve", type="primary", use_container_width=True):
                try:
                    # Execute the tool
                    result = st.session_state.app.invoke(None, st.session_state.thread_config)

                    result = st.session_state.app.invoke(
                        {"messages": []}, 
                        st.session_state.thread_config
                    )

                    # 결과에서 마지막 메시지(ToolMessage)를 찾아서 표시
                    if result and "messages" in result:
                        tool_message = result["messages"][-1]
                        # tool_message.content에서 None이 아닌 경우만 표시
                        content = tool_message.content if tool_message.content else ""
                        # 간단히 결과만 표시
                        response_text = content
                    else:
                        response_text = "Tool executed successfully, but no result message was returned."

                    # Add to messages
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response_text
                    })

                    # Clear pending approval
                    st.session_state.pending_approval = False
                    st.session_state.pending_tool_calls = None

                    st.rerun()
                except Exception as e:
                    st.error(f"Error executing tool: {e}")

        with col2:
            if st.button("❌ Reject", use_container_width=True):
                try:
                    # Instead of rollback, add ToolMessage responses for each tool_call
                    # This prevents OpenAI API error about missing tool responses
                    from langchain_core.messages import ToolMessage

                    tool_messages = []
                    for tool_call in st.session_state.pending_tool_calls:
                        tool_messages.append(
                            ToolMessage(
                                content="Tool execution cancelled by user.",
                                tool_call_id=tool_call['id']
                            )
                        )

                    # Update state with tool cancellation messages
                    st.session_state.app.update_state(
                        st.session_state.thread_config,
                        {"messages": tool_messages}
                    )

                    # Add cancellation message
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "❌ Tool execution cancelled. Please provide a new query or modify your request."
                    })

                    # Clear pending approval
                    st.session_state.pending_approval = False
                    st.session_state.pending_tool_calls = None

                    st.rerun()
                except Exception as e:
                    st.error(f"Error during rejection: {e}")
                    # Still clear pending state even if rollback fails
                    st.session_state.pending_approval = False
                    st.session_state.pending_tool_calls = None
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "Tool execution cancelled."
                    })
                    st.rerun()

    # Chat input
    if prompt := st.chat_input("Enter your query...", disabled=st.session_state.pending_approval):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        # Get agent response
        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                try:
                    result = st.session_state.app.invoke(
                        {"messages": [{"role": "user", "content": prompt}]},
                        st.session_state.thread_config
                    )

                    if result and "messages" in result:
                        last_message = result["messages"][-1]

                        # Check if there are pending tool calls
                        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                            st.session_state.pending_approval = True
                            st.session_state.pending_tool_calls = last_message.tool_calls

                            response_text = "🔧 Tool calls are pending approval. Please review and approve/reject above."
                        else:
                            response_text = last_message.content

                        st.markdown(response_text)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response_text
                        })

                    if st.session_state.pending_approval:
                        st.rerun()

                except Exception as e:
                    error_msg = f"Error: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })

# Footer
st.markdown("---")
st.markdown("Made with Streamlit and LangGraph")

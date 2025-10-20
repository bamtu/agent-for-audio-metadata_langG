import os
from typing import Literal
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import MessagesState

from utils.audio_tools import (
    get_filepaths_by_query_with_retriever_tool,
    batch_update_artist_tool,
    batch_update_to_same_artist_tool,
    update_title_tool,
    batch_update_album_tool,
    batch_update_to_same_album_tool,
    batch_update_genre_tool,
    batch_update_to_same_genre_tool,
    batch_update_year_tool,
    batch_update_to_same_year_tool,
    update_track_tool,
    update_comment_tool,
    batch_update_comment_tool,
    batch_update_album_artist_tool,
    batch_update_to_same_album_artist_tool
)


FOLDER_PATH = "C:/music_files"

# Metadata update tools only (excluding retrieval tool)
metadata_update_tools = [
    batch_update_artist_tool,
    batch_update_to_same_artist_tool,
    update_title_tool,
    batch_update_album_tool,
    batch_update_to_same_album_tool,
    batch_update_genre_tool,
    batch_update_to_same_genre_tool,
    batch_update_year_tool,
    batch_update_to_same_year_tool,
    update_track_tool,
    update_comment_tool,
    batch_update_comment_tool,
    batch_update_album_artist_tool,
    batch_update_to_same_album_artist_tool
]

SYSTEM_MESSAGE = f"""You are a metadata editing agent for music files.
Your job is to update metadata of audio files based on user requests.
Files are located in: {FOLDER_PATH}

Available metadata update tools:
- batch_update_artist_tool: Update different artists for multiple files
- batch_update_to_same_artist_tool: Update same artist for multiple files
- update_title_tool: Update title of a single file
- batch_update_album_tool: Update different albums for multiple files
- batch_update_to_same_album_tool: Update same album for multiple files
- batch_update_genre_tool: Update different genres for multiple files
- batch_update_to_same_genre_tool: Update same genre for multiple files
- batch_update_year_tool: Update different years for multiple files
- batch_update_to_same_year_tool: Update same year for multiple files
- update_track_tool: Update track number of a single file
- update_comment_tool: Update comment of a single file
- batch_update_comment_tool: Update different comments for multiple files
- batch_update_album_artist_tool: Update different album artists for multiple files
- batch_update_to_same_album_artist_tool: Update same album artist for multiple files

Use these tools only when user explicitly asks to update metadata.
If retriever can't retrieve any files, inform the user that no files were found.
"""


def get_llm():
    """Initialize Azure OpenAI LLM"""
    llm = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        temperature=1
    )
    return llm


def retrieve_node(state: MessagesState):
    """
    Retrieve node that searches for relevant audio files based on user query.
    """
    messages = state["messages"]
    last_message = messages[-1]

    try:
        # Use retrieval tool to find relevant files
        filepaths = get_filepaths_by_query_with_retriever_tool.invoke({"query": last_message.content})

        # Format results with instruction for next step
        if filepaths:
            result_message = (
                f"검색된 파일들 ({len(filepaths)}개):\n" +
                "\n".join([f"- {fp}" for fp in filepaths]) +
                f"\n\n이 파일들의 메타데이터를 업데이트하려면 어떤 작업을 하시겠습니까?"
            )
        else:
            result_message = "검색된 파일이 없습니다."

        return {"messages": [AIMessage(content=result_message)]}
    except Exception as e:
        return {"messages": [AIMessage(content=f"검색 중 오류 발생: {str(e)}")]}


def tool_node(state: MessagesState):
    """
    Tool node that decides which metadata update tool to call.
    LLM analyzes user request and previous messages to select appropriate tool.
    """
    llm = get_llm()
    llm_with_tools = llm.bind_tools(metadata_update_tools)

    messages = state["messages"]

    # Add system message for context
    messages_with_system = [{"role": "system", "content": SYSTEM_MESSAGE}] + messages

    # Call LLM to decide which tool to use
    response = llm_with_tools.invoke(messages_with_system)

    return {"messages": [response]}


def should_continue_to_review(state: MessagesState) -> Literal["human_review", "end"]:
    """
    Router: Check if tool calls exist to determine next step.
    """
    messages = state["messages"]
    last_message = messages[-1]

    # If there are tool calls, go to human review
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "human_review"

    return "end"


def human_review(state: MessagesState):
    """
    Human review node - passes through state for human inspection.
    This node is interrupted before execution for human approval.
    """
    # Simply pass through the state
    # The actual approval happens via interrupt mechanism
    return state


def should_execute_tool(state: MessagesState) -> Literal["tool_executor", "end"]:
    """
    Router after human review: determines whether to execute tool or end.
    This is called after human approval/rejection.
    """
    messages = state["messages"]
    last_message = messages[-1]

    # If there are tool calls, execute them
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tool_executor"

    return "end"


# Create tool execution node
tool_executor = ToolNode(metadata_update_tools)

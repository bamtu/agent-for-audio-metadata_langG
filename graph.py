from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph

from nodes import (
    retrieve_node,
    route_after_tool_choice,
    tool_executor,
    tool_node,
)


def build_graph(memory: MemorySaver):
    flow = StateGraph(MessagesState)

    flow.add_node("retriever", retrieve_node)
    flow.add_node("update_tool", tool_node)
    flow.add_node("tool_executor", tool_executor)

    flow.set_entry_point("retriever")
    flow.add_edge("retriever", "update_tool")

    flow.add_conditional_edges(
        "update_tool",
        route_after_tool_choice,
        {
            "tool_executor": "tool_executor",
            "end": END,
        },
    )

    flow.add_edge("tool_executor", END)

    return flow.compile(checkpointer=memory, interrupt_before=["tool_executor"])

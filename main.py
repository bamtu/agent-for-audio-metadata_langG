from dotenv import load_dotenv

from utils.utils import *
from utils.audio_tools import *
from utils.audio_tag_editor import *

from langgraph.graph import END, MessagesState, StateGraph

from nodes import (
    get_llm,
    retrieve_node,
    tool_node,
    route_after_tool_choice,
    tool_executor
)

if __name__ == "__main__":

    load_dotenv()

    # Set folder path
    folder_path = "C:/music_files"

    # Initialize LLM
    llm = get_llm()

    # Initialize vector store with retriever
    print("Initializing vector store...")
    init_vector_store(folder_path=folder_path, llm=llm)
    print("Vector store initialized successfully!")

    # Create LangGraph workflow
    flow = StateGraph(MessagesState)

    # Add nodes
    flow.add_node("retriever", retrieve_node)  # Start: search for files
    flow.add_node("update_tool", tool_node)    # Decide which metadata update tool to use
    flow.add_node("tool_executor", tool_executor)  # Execute tools after human approval

    # Set entry point to retriever
    flow.set_entry_point("retriever")

    # retriever -> update_tool (always go to update_tool after retrieval)
    flow.add_edge("retriever", "update_tool")

    # Conditional routing from human_review
    # After human approval, either execute tool or end
    flow.add_conditional_edges(
        "update_tool",
        route_after_tool_choice,
        {
            "tool_executor": "tool_executor",
            "end": END
        }
    )

    # After tool execution, end the flow
    flow.add_edge("tool_executor", END)

    # Compile the graph with interrupt before human_review for approval
    app = flow.compile(interrupt_before=["tool_executor"])

    # Save graph visualization
    app.get_graph().draw_mermaid_png(output_file_path="graph.png")
    print("Graph saved to graph.png")

    # Interactive loop
    print("\n" + "="*50)
    print("Audio Metadata Agent is ready!")
    print("="*50)
    print("Enter your queries (type 'quit' or 'exit' to stop):\n")

    config = {"configurable": {"thread_id": "1"}}

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        if not user_input:
            continue

        # Invoke the agent
        try:
            # Invoke the workflow
            result = app.invoke({"messages": [{"role": "user", "content": user_input}]}, config)

            # Display all messages
            if result and "messages" in result:
                for msg in result["messages"][1:]:  # Skip user message
                    if hasattr(msg, 'content') and msg.content:
                        print(f"\nAgent: {msg.content}")

                    # Check if workflow was interrupted (tool calls need approval)
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        print("\n" + "="*50)
                        print("Tool calls detected - Human Review Required:")
                        print("="*50)
                        for tool_call in msg.tool_calls:
                            print(f"\nTool: {tool_call['name']}")
                            print(f"Arguments: {tool_call['args']}")

                        approval = input("\nApprove these tool calls? (yes/no): ").strip().lower()

                        if approval in ['yes', 'y']:
                            # Continue the workflow after approval (resume from interrupt)
                            print("\nExecuting tools...")
                            # Resume from interrupt by passing the previous result
                            continue_result = app.invoke(None, config)

                            # Display results after execution
                            if continue_result and "messages" in continue_result:
                                print("\n✓ Tools executed successfully.")
                                for msg in continue_result["messages"]:
                                    if msg.role == 'tool':
                                        print(f"  - Result for [{msg.name}]: {msg.content}")
                        else:
                            print("\nTool execution cancelled.")
                            app.update_state(
                                config,
                                {"messages": [("human", "Tool execution cancelled by user.")]}
                            )

                print()  # Empty line for readability
            else:
                print("\nAgent: No response generated.\n")

        except Exception as e:
            print(f"\nError: {e}\n")
            import traceback
            traceback.print_exc()

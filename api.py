import os
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

from graph import build_graph
from nodes import get_llm
from utils.utils import init_vector_store


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None  # 없으면 서버가 새로 생성


class ChatResponse(BaseModel):
    thread_id: str
    status: str  # "pending" | "done"
    message: Optional[str] = None       # 에이전트 응답 텍스트 (done일 때)
    tool_calls: Optional[list] = None   # 승인 대기 중인 툴 목록 (pending일 때)


class ApproveRequest(BaseModel):
    thread_id: str
    approved: bool


class ApproveResponse(BaseModel):
    status: str   # "done" | "cancelled"
    message: str


# ---------------------------------------------------------------------------
# App state (서버 수명 동안 유지)
# ---------------------------------------------------------------------------

app_state: dict = {}


# ---------------------------------------------------------------------------
# Lifespan: 서버 시작 시 한 번만 초기화
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()

    folder_path = os.getenv("MUSIC_FOLDER_PATH", "C:/music_files")
    llm = get_llm()

    print(f"Initializing vector store from: {folder_path}")
    init_vector_store(folder_path=folder_path, llm=llm)
    print("Vector store initialized.")

    memory = MemorySaver()
    app_state["graph"] = build_graph(memory)
    print("LangGraph compiled. API is ready.")

    yield

    app_state.clear()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

api = FastAPI(title="Audio Metadata Agent API", lifespan=lifespan)

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 운영 환경에서는 실제 프론트엔드 origin으로 좁히세요
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@api.get("/health")
async def health():
    """서버 및 초기화 상태 확인"""
    return {"status": "ok", "initialized": "graph" in app_state}


@api.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    사용자 메시지를 받아 에이전트를 실행합니다.

    - 툴 실행이 필요하면 status="pending" + tool_calls 반환
    - 툴 없이 응답이 완료되면 status="done" + message 반환
    """
    graph = app_state.get("graph")
    if graph is None:
        raise HTTPException(status_code=503, detail="Agent not initialized.")

    thread_id = request.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    result = await graph.ainvoke(
        {"messages": [{"role": "user", "content": request.message}]},
        config,
    )

    if not result or "messages" not in result:
        raise HTTPException(status_code=500, detail="Agent returned no response.")

    last_message = result["messages"][-1]

    # 툴 실행 대기 중 (interrupt_before=["tool_executor"] 에 걸린 상태)
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        serialized_tool_calls = [
            {"id": tc["id"], "name": tc["name"], "args": tc["args"]}
            for tc in last_message.tool_calls
        ]
        return ChatResponse(
            thread_id=thread_id,
            status="pending",
            tool_calls=serialized_tool_calls,
        )

    return ChatResponse(
        thread_id=thread_id,
        status="done",
        message=last_message.content,
    )


@api.post("/approve", response_model=ApproveResponse)
async def approve(request: ApproveRequest):
    """
    툴 실행을 승인하거나 거절합니다.

    - approved=true  → 중단된 지점에서 재개
    - approved=false → 취소 ToolMessage를 삽입하고 종료
    """
    graph = app_state.get("graph")
    if graph is None:
        raise HTTPException(status_code=503, detail="Agent not initialized.")

    config = {"configurable": {"thread_id": request.thread_id}}

    if request.approved:
        result = await graph.ainvoke(None, config)

        if result and "messages" in result:
            last_message = result["messages"][-1]
            result_text = last_message.content if last_message.content else "완료"
        else:
            result_text = "툴 실행 완료"

        return ApproveResponse(status="done", message=result_text)

    else:
        # 현재 중단된 상태의 tool_calls를 가져와서 각각 취소 ToolMessage 삽입
        current_state = graph.get_state(config)
        messages = current_state.values.get("messages", [])

        tool_messages = []
        for msg in reversed(messages):
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_messages.append(
                        ToolMessage(
                            content="Tool execution cancelled by user.",
                            tool_call_id=tc["id"],
                        )
                    )
                break

        if tool_messages:
            graph.update_state(config, {"messages": tool_messages})

        return ApproveResponse(status="cancelled", message="툴 실행이 취소되었습니다.")

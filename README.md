# agent-for-audio-metadata_langG
audio metadata agent implementation with LangGraph

### 파일 구조

```
.
├── main.py                      # CLI 진입점. 벡터 스토어 초기화 후 터미널에서 에이전트와 대화
├── app.py                       # Streamlit UI. 웹 브라우저에서 에이전트와 대화
├── api.py                       # FastAPI 서버. REST API로 에이전트 기능 제공 (/chat, /approve)
├── graph.py                     # LangGraph 워크플로우 정의. 노드와 엣지를 연결해 그래프 빌드
├── nodes.py                     # LangGraph 노드 구현. LLM 초기화, 검색/툴 선택/실행 노드 정의
├── requirements.txt             # 패키지 의존성 목록
└── utils/
    ├── audio_tag_editor.py      # mutagen을 사용해 MP3/M4A 파일의 메타데이터를 읽고 수정
    ├── audio_tools.py           # LangChain Tool로 래핑된 메타데이터 일괄 수정 함수 모음
    └── utils.py                 # Chroma 벡터 스토어 초기화 및 SelfQueryRetriever 설정
```

### 필요 조건

ollama embedding -> bona/bge-m3-korean:latest가 local에서 구동되고 있어야함.


### Streamlit을 사용하려면
```bash
streamlit run app.py
```

### FastAPI 서버를 띄우려면
```bash
uvicorn api:api
```


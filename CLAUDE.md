# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Audio metadata agent implementation using LangGraph. The system processes audio files (MP3, M4A) to read, manage, and edit their metadata using vector stores and AI-powered retrieval.

## Core Architecture

### Main Components

- **main.py**: Entry point - initializes vector store and sets up LangGraph workflow (incomplete: missing LLM parameter at line 14)
- **nodes.py**: Empty file for LangGraph node definitions - needs implementation
- **utils/**:
  - **audio_tag_editor.py**: Audio metadata read/write using mutagen (EasyID3, EasyMP4)
  - **audio_tools.py**: LangChain tools for batch metadata operations
  - **utils.py**: Vector store initialization and self-query retriever setup

### LangGraph Workflow

Uses StateGraph with MessagesState:
- Entry point: VECTOR_STORE_INIT node (not yet defined)
- Agent reasoning: AGENT_REASONING node (imports from empty nodes.py)
- Compiles to graph visualization (graph.png)

### Vector Store

Two approaches available:
1. **Metadata-based** (`init_vector_store`): Individual documents per audio file
2. **Content-based** (`init_vector_store_as_content`): Chunked documents (100 files/chunk)

Embeddings:
- Ollama: bona/bge-m3-korean (metadata), qllama/multilingual-e5-large-instruct (content)
- Azure OpenAI: text-embedding-3-large (commented out in code)

Uses Chroma as vector database with self-query retriever for structured metadata filtering.

### Metadata Fields

Supports MP3/M4A with fields: filepath, title, album, artist, genre, year, track, comment, album_artist

## Key Functions

### Initialization
- `init_vector_store(folder_path, llm)`: Creates self-query retriever with metadata field info
- `store_metadata_in_vector_store()`: Processes folder and stores in Chroma

### Updates
- `update_*()` functions in audio_tag_editor.py: Updates both file tags and vector store
- Returns vector store object (inconsistent return types - some return strings, update_artist returns vector_store)

### Tools (audio_tools.py)
- `get_filepaths_by_query_with_retriever_tool`: Natural language file queries
- `batch_update_*_tool`: Batch operations for each metadata field
- `update_title_tool`, `update_track_tool`, `update_comment_tool`: Individual updates
- Korean language responses ("개 성공")

## Current Issues

1. **main.py:14**: Missing LLM parameter in init_vector_store call
2. **main.py:17**: VECTOR_STORE_INIT constant undefined
3. **nodes.py**: Empty - agent_reasoning function not implemented
4. **Inconsistent returns**: update_artist returns vector_store, others return strings

## Development Commands

No test, build, or lint commands found in codebase. Project appears to be in early development stage.

## Environment Setup

Requires `.env` file with Azure OpenAI credentials if using Azure embeddings (currently commented out).
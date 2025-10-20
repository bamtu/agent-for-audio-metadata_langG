from pathlib import Path
from mutagen.easyid3 import EasyID3
from uuid import uuid4
from langchain_core.documents import Document
from langchain_chroma import Chroma
from mutagen.easymp4 import EasyMP4


def return_metadata_from_folder(folder_path: str) -> list[dict]:
    result = []
    for file in Path(folder_path).iterdir():
        if file.is_file():
            filepath = str(file.resolve())
            ext = file.suffix.lower()

            # 기본값은 None
            metadata = {
                "filepath": filepath,
                "title": None,
                "album": None, 
                "artist": None,
                "genre": None,
                "year": None,
                "track": None,
                "comment": None,
                "album_artist": None,
            }

            try:
                if ext == ".mp3":
                    tag = EasyID3(filepath)
                    metadata["title"] = tag.get("title", [None])[0]
                    metadata["album"] = tag.get("album", [None])[0]
                    metadata["artist"] = tag.get("artist", [None])[0]
                    metadata["genre"] = tag.get("genre", [None])[0]
                    metadata["year"] = tag.get("date", [None])[0]
                    metadata["track"] = tag.get("tracknumber", [None])[0]
                    metadata["comment"] = tag.get("comment", [None])[0]
                    metadata["album_artist"] = tag.get("albumartist", [None])[0]

                elif ext == ".m4a":
                    tag = EasyMP4(filepath)
                    metadata["title"] = tag.get("title", [None])[0]
                    metadata["album"] = tag.get("album", [None])[0]
                    metadata["artist"] = tag.get("artist", [None])[0]
                    metadata["genre"] = tag.get("genre", [None])[0]
                    metadata["year"] = tag.get("date", [None])[0]
                    metadata["track"] = tag.get("tracknumber", [None])[0]
                    metadata["comment"] = tag.get("comment", [None])[0]
                    metadata["album_artist"] = tag.get("albumartist", [None])[0]

            except Exception as e:
                # 오류 발생 시 기본값 그대로 유지
                print(f"[오류] {filepath}: {e}")

            result.append(metadata)
        
    return result

def store_metadata_in_vector_store(folder_path: str, embeddings) -> Chroma:
    metadata_list = return_metadata_from_folder(folder_path)
    documents = []

    for metadata in metadata_list:
        file_path = metadata["filepath"]
        content = (
            f"Audio file metadata for: {file_path}"
        )
        
        document = Document(page_content=content, metadata=metadata, id=f"{file_path}")
        documents.append(document)
        print(document)

    vector_store = Chroma.from_documents(documents=documents, embedding=embeddings)
    print(f"메타데이터를 벡터 스토어에 저장했습니다. 문서 수: {len(documents)}")
    return vector_store

def store_page_content_in_vector_store(folder_path: str, embeddings) -> Chroma:
    metadata_list = return_metadata_from_folder(folder_path)

    documents = []
    chunk_size = 100  # 10개씩 묶음
    for i in range(0, len(metadata_list), chunk_size):
        chunk = metadata_list[i:i + chunk_size]

        chunk_texts = []
        for metadata in chunk:
            page = "\n".join([
                f"File: {metadata['filepath']}",
                f"Title: {metadata.get('title', '')}",
                f"Album: {metadata.get('album', '')}",
                f"Artist: {metadata.get('artist', '')}",
                f"Genre: {metadata.get('genre', '')}",
                f"Year: {metadata.get('year', '')}",
                f"Album Artist: {metadata.get('album_artist', '')}",
                f"Comment: {metadata.get('comment', '')}",
                ", "  # 문서 간 구분
            ])
            chunk_texts.append(page)

        full_chunk_text = "\n".join(chunk_texts)

        document = Document(
            page_content=full_chunk_text,
            metadata={"source": f"{folder_path}_chunk_{i//chunk_size}"}
        )
        documents.append(document)

    vector_store = Chroma.from_documents(documents, embedding=embeddings)
    print(f"✅ {len(documents)}개의 도큐먼트를 벡터 스토어에 저장했습니다. (총 {len(metadata_list)}개의 파일)")

    return vector_store, documents

def update_title(vector_store, filepath: str, title: str) -> str:
    path = Path(filepath)
    ext = path.suffix.lower()
    try:
        if ext == ".mp3":
            tag = EasyID3(filepath)
            tag['title'] = title
            tag.save(filepath)
            vector_store.update_document(
                document=Document(page_content="page_content", 
                                  metadata={"title": title})
                ,document_id=f"{filepath}")
        elif ext == ".m4a":
            tag = EasyMP4(filepath)
            tag['title'] = title
            tag.save(filepath)
            vector_store.update_document(
                document=Document(page_content="page_content", 
                                  metadata={"title": title})
                ,document_id=f"{filepath}")
        else:
            return f"지원하지 않는 파일 형식입니다: {filepath}"
        return f"제목을 '{title}'로 업데이트했습니다: {filepath}"
    except Exception as e:
        return f"[오류] 제목 업데이트 실패 - {filepath}: {e}"    

def update_album(vector_store, filepath: str, album: str) -> str:
    path = Path(filepath)
    ext = path.suffix.lower()
    try:
        if ext == ".mp3":
            tag = EasyID3(filepath)
            tag['album'] = album
            tag.save(filepath)
            vector_store.update_document(
                document=Document(page_content="page_content", 
                                  metadata={"album": album})
                ,document_id=f"{filepath}")
        elif ext == ".m4a":
            tag = EasyMP4(filepath)
            tag['album'] = album
            tag.save(filepath)
            vector_store.update_document(
                document=Document(page_content="page_content", 
                                  metadata={"album": album})
                ,document_id=f"{filepath}")
        else:
            return f"지원하지 않는 파일 형식입니다: {filepath}"
        return f"제목을 '{album}'로 업데이트했습니다: {filepath}"
    except Exception as e:
        return f"[오류] 제목 업데이트 실패 - {filepath}: {e}"
    
def update_artist(vector_store, filepath: str, artist: str) -> str:
    path = Path(filepath)
    ext = path.suffix.lower()
    try:
        if ext == ".mp3":
            tag = EasyID3(filepath)
            tag['artist'] = artist
            tag.save(filepath)
            vector_store.update_document(
                document=Document(page_content="page_content", 
                                  metadata={"artist": artist})
                ,document_id=f"{filepath}")
        elif ext == ".m4a":
            tag = EasyMP4(filepath)
            tag['artist'] = artist
            tag.save(filepath)
            vector_store.update_document(
                document=Document(page_content="page_content", 
                                  metadata={"artist": artist})
                ,document_id=f"{filepath}")
        else:
            return f"지원하지 않는 파일 형식입니다: {filepath}"
        return vector_store
    except Exception as e:
        return f"[오류] 아티스트 업데이트 실패 - {filepath}: {e}"

def update_genre(vector_store, filepath: str, genre: str) -> str:
    path = Path(filepath)
    ext = path.suffix.lower()
    try:
        if ext == ".mp3":
            tag = EasyID3(filepath)
            tag['genre'] = genre
            tag.save(filepath)
            vector_store.update_document(
                document=Document(page_content="page_content", 
                                  metadata={"genre": genre})
                ,document_id=f"{filepath}")
        elif ext == ".m4a":
            tag = EasyMP4(filepath)
            tag['genre'] = genre
            tag.save(filepath)
            vector_store.update_document(
                document=Document(page_content="page_content", 
                                  metadata={"genre": genre})
                ,document_id=f"{filepath}")
        else:
            return f"지원하지 않는 파일 형식입니다: {filepath}"
        return f"장르를 '{genre}'로 업데이트했습니다: {filepath}"
    except Exception as e:
        return f"[오류] 장르 업데이트 실패 - {filepath}: {e}"

def update_year(vector_store, filepath: str, year: str) -> str:
    path = Path(filepath)
    ext = path.suffix.lower()
    try:
        if ext == ".mp3":
            tag = EasyID3(filepath)
            tag['date'] = year
            tag.save(filepath)
            vector_store.update_document(
                document=Document(page_content="page_content", 
                                  metadata={"year": year})
                ,document_id=f"{filepath}")
        elif ext == ".m4a":
            tag = EasyMP4(filepath)
            tag['date'] = year
            tag.save(filepath)
            vector_store.update_document(
                document=Document(page_content="page_content", 
                                  metadata={"year": year})
                ,document_id=f"{filepath}")
        else:
            return f"지원하지 않는 파일 형식입니다: {filepath}"
        return f"연도를 '{year}'로 업데이트했습니다: {filepath}"
    except Exception as e:
        return f"[오류] 연도 업데이트 실패 - {filepath}: {e}"

def update_track(vector_store, filepath: str, track: str) -> str:
    path = Path(filepath)
    ext = path.suffix.lower()
    try:
        if ext == ".mp3":
            tag = EasyID3(filepath)
            tag['tracknumber'] = track
            tag.save(filepath)
            vector_store.update_document(
                document=Document(page_content="page_content", 
                                  metadata={"track": track})
                ,document_id=f"{filepath}")
        elif ext == ".m4a":
            tag = EasyMP4(filepath)
            tag['tracknumber'] = track  # 예: 3 -> (3, 0)
            tag.save(filepath)
            vector_store.update_document(
                document=Document(page_content="page_content", 
                                  metadata={"track": track})
                ,document_id=f"{filepath}")
        else:
            return f"지원하지 않는 파일 형식입니다: {filepath}"
        return f"트랙 번호를 '{track}'으로 업데이트했습니다: {filepath}"
    except Exception as e:
        return f"[오류] 트랙 번호 업데이트 실패 - {filepath}: {e}"

def update_comment(vector_store, filepath: str, comment: str) -> str:
    path = Path(filepath)
    ext = path.suffix.lower()
    try:
        if ext == ".mp3":
            tag = EasyID3(filepath)
            tag['comment'] = comment
            tag.save(filepath)
            vector_store.update_document(
                document=Document(page_content="page_content", 
                                  metadata={"comment": comment})
                ,document_id=f"{filepath}")
        elif ext == ".m4a":
            tag = EasyMP4(filepath)
            tag['comment'] = comment
            tag.save(filepath)
            vector_store.update_document(
                document=Document(page_content="page_content", 
                                  metadata={"comment": comment})
                ,document_id=f"{filepath}")
        else:
            return f"지원하지 않는 파일 형식입니다: {filepath}"
        return f"코멘트를 '{comment}'로 업데이트했습니다: {filepath}"
    except Exception as e:
        return f"[오류] 코멘트 업데이트 실패 - {filepath}: {e}"

def update_album_artist(vector_store, filepath: str, album_artist: str) -> str:
    path = Path(filepath)
    ext = path.suffix.lower()
    try:
        if ext == ".mp3":
            tag = EasyID3(filepath)
            tag['albumartist'] = album_artist
            tag.save(filepath)
            vector_store.update_document(
                document=Document(page_content="page_content", 
                                  metadata={"album_artist": album_artist})
                ,document_id=f"{filepath}")
        elif ext == ".m4a":
            tag = EasyMP4(filepath)
            tag['albumartist'] = album_artist
            tag.save(filepath)
            vector_store.update_document(
                document=Document(page_content="page_content", 
                                  metadata={"album_artist": album_artist})
                ,document_id=f"{filepath}")
        else:
            return f"지원하지 않는 파일 형식입니다: {filepath}"
        return f"앨범 아티스트를 '{album_artist}'로 업데이트했습니다: {filepath}"
    except Exception as e:
        return f"[오류] 앨범 아티스트 업데이트 실패 - {filepath}: {e}"
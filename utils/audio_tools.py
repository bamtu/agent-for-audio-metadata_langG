from utils.audio_tag_editor import *
from utils.utils import *

from langchain_core.tools import tool
from typing import List




@tool
def get_filepaths_by_query_with_retriever_tool(query: str) -> list[str]:
    """
    Returns a list of filepaths of music files that correspond to a given query message.
    Example: “Music files with the genre Pop”
    """
    retriever = get_retriever()
    docs = retriever.invoke(query)
    filepaths = [doc.metadata.get("filepath", "") for doc in docs if "filepath" in doc.metadata]
    return filepaths

@tool
def batch_update_artist_tool(filepaths: List[str], artists: List[str]):
    """
    Update the artists of the given audio files.
    filepaths and artists should be of the same length, and each file will be updated with the corresponding artist.
    Args:
        filepaths: List of file paths
        artists: List of artists
    """
    if len(filepaths) != len(artists):
        return "filepaths와 artists의 길이가 같아야 합니다."

    success_count = 0
    vector_store = get_vector_store()
    for path, artist in zip(filepaths, artists):
        vector_store = update_artist(vector_store, path, artist)
        success_count += 1
    return f"{success_count}개 성공"

@tool
def batch_update_to_same_artist_tool(filepaths: List[str], artist: str):
    """
    Update to same artist of the given audio files.
    Args:
        filepaths: List of file paths
        artist: artist
    """
    success_count = 0
    vector_store = get_vector_store()
    for path in filepaths:
        vector_store = update_artist(vector_store, path, artist)
        success_count += 1
    return f"{success_count}개 성공"

@tool
def update_title_tool(filepath: str, title: str) -> str:
    """Update the title of the given audio file.
    Args: filepath:filepath, title:title
    """
    vector_store = get_vector_store()
    vector_store = update_title(vector_store, filepath, title)
    return 'title updated successfully'

@tool
def batch_update_album_tool(filepaths: List[str], albums: List[str]) -> str:
    """
    Update the different albums of the given audio files.
    filepaths and albums should be of the same length, and each file will be updated with the corresponding album.
    Args:
        filepaths: List of file paths
        albums: List of albums
    """
    if len(filepaths) != len(albums):
        return "filepaths와 albums의 길이가 같아야 합니다."

    success_count = 0
    vector_store = get_vector_store()
    for path, album in zip(filepaths, albums):
        vector_store = update_album(vector_store, path, album)
        success_count += 1
    return f"{success_count}개 성공"

@tool
def batch_update_to_same_album_tool(filepaths: List[str], album: str) -> str:
    """
    Update to same album of the given audio files.
    Args:
        filepaths: List of file paths
        album: album
    """
    success_count = 0
    vector_store = get_vector_store()
    for path in filepaths:
        vector_store = update_album(vector_store, path, album)
        success_count += 1
    return f"{success_count}개 성공"


@tool
def batch_update_genre_tool(filepaths: List[str], genres: List[str]) -> str:
    """
    Update the different genres of the given audio files.
    filepaths and genres should be of the same length, and each file will be updated with the corresponding genre.
    Args: filepaths: List of file paths, genres: List of genres
    """
    if len(filepaths) != len(genres):
        return ["filepaths와 genres의 길이가 같아야 합니다."]
    
    success_count = 0
    vector_store = get_vector_store()
    for path, genre in zip(filepaths, genres):
        vector_store = update_genre(vector_store, path, genre)
        success_count += 1
        
    return f"{success_count}개 성공"

@tool
def batch_update_to_same_genre_tool(filepaths: List[str], genre: str) -> str:
    """
    Update to same genres of the given audio files.
    filepaths and genres should be of the same length, and each file will be updated with the corresponding genre.
    Args: filepaths: List of file paths, genres: genre
    """
    success_count = 0
    vector_store = get_vector_store()
    for path in filepaths:
        vector_store = update_genre(vector_store, path, genre)
        success_count += 1
        
    return f"{success_count}개 성공"

@tool
def batch_update_year_tool(filepaths: List[str], years: List[str]) -> str:
    """
    Update the different years of the given audio files.
    filepaths and years should be of the same length, and each file will be updated with the corresponding year.
    Args:
        filepaths: List of file paths
        years: List of years
    """
    if len(filepaths) != len(years):
        return "filepaths와 years의 길이가 같아야 합니다."

    success_count = 0
    vector_store = get_vector_store()
    for path, year in zip(filepaths, years):
        vector_store = update_year(vector_store, path, year)
        success_count += 1

    return f"{success_count}개 성공"

@tool
def batch_update_to_same_year_tool(filepaths: List[str], year: str) -> str:
    """
    Update to same year of the given audio files.
    Args:
        filepaths: List of file paths
        year: year
    """
    success_count = 0
    vector_store = get_vector_store()
    for path in filepaths:
        vector_store = update_year(vector_store, path, year)
        success_count += 1

    return f"{success_count}개 성공"

@tool
def update_track_tool(filepath: str, track: str) -> str:
    """Update the track number of the given audio file.
    Args: filepath:filepath, track:track
    """
    vector_store = get_vector_store()
    vector_store = update_track(vector_store, filepath, track)
    return "track updated successfully"

@tool
def update_comment_tool(filepath: str, comment: str) -> str:
    """Update the comment of the given audio file.
    Args: filepath:filepath, comment:comment
    """
    vector_store = get_vector_store()
    vector_store = update_comment(vector_store, filepath, comment)
    return "comment updated successfully"

@tool
def batch_update_comment_tool(filepaths: List[str], comments: List[str]) -> str:
    """
    Update the different comments of the given audio files.
    filepaths and comments should be of the same length, and each file will be updated with the corresponding comment.
    Args:
        filepaths: List of file paths
        comments: List of comments
    """
    if len(filepaths) != len(comments):
        return "filepaths와 comments의 길이가 같아야 합니다."

    success_count = 0
    vector_store = get_vector_store()
    for path, comment in zip(filepaths, comments):
        vector_store = update_comment(vector_store, path, comment)
        success_count += 1

    return f"{success_count}개 성공"

@tool
def batch_update_album_artist_tool(filepaths: List[str], album_artists: List[str]) -> str:
    """
    Update the different album artists of the given audio files.
    filepaths and album_artists should be of the same length, and each file will be updated with the corresponding album artist.
    Args:
        filepaths: List of file paths
        album_artists: List of album artists
    """
    if len(filepaths) != len(album_artists):
        return "filepaths와 album_artists의 길이가 같아야 합니다."

    success_count = 0
    vector_store = get_vector_store()
    for path, album_artist in zip(filepaths, album_artists):
        vector_store = update_album_artist(vector_store, path, album_artist)
        success_count += 1
    return f"{success_count}개 성공"

@tool
def batch_update_to_same_album_artist_tool(filepaths: List[str], album_artist: str) -> str:
    """
    Update to same album artist of the given audio files.
    Args:
        filepaths: List of file paths
        album_artist: album artist
    """
    success_count = 0
    vector_store = get_vector_store()
    for path in filepaths:
        vector_store = update_album_artist(vector_store, path, album_artist)
        success_count += 1
    return f"{success_count}개 성공"

# @tool
# def batch_update_composer_tool(filepaths: List[str], composers: List[str]) -> str:
#     """
#     Update the composers of the given audio files.
#     filepaths and composers should be of the same length, and each file will be updated with the corresponding composer.
#     Args:
#         filepaths: List of file paths
#         composers: List of composers
#     """
#     if len(filepaths) != len(composers):
#         return "filepaths와 composers의 길이가 같아야 합니다."

#     success_count = 0
#     for path, composer in zip(filepaths, composers):
#         result = update_composer(path, composer)
#         success_count += 1

#     return f"{success_count}개 성공"

# @tool
# def batch_update_to_same_composer_tool(filepaths: List[str], composer: str) -> str:
#     """
#     Update to same composer of the given audio files.
#     Args:
#         filepaths: List of file paths
#         composer: composer
#     """
#     success_count = 0
#     for path in filepaths:
#         result = update_composer(path, composer)
#         success_count += 1

#     return f"{success_count}개 성공"

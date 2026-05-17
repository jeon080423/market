import streamlit as st
import requests
import time
from datetime import datetime, timezone
import concurrent.futures
from youtube_transcript_api import YouTubeTranscriptApi

YOUTUBE_BASE_URL = "https://www.googleapis.com/youtube/v3"

def get_youtube_api_key():
    """
    Case-insensitive check for YouTube API Key in Streamlit secrets.
    """
    if "YOUTUBE_API_KEY" in st.secrets:
        return st.secrets["YOUTUBE_API_KEY"]
    if "youtube_api_key" in st.secrets:
        return st.secrets["youtube_api_key"]
    
    # Nested check: [youtube] api_key = "..."
    for key in st.secrets.keys():
        if key.lower() == "youtube":
            section = st.secrets[key]
            if hasattr(section, "get"):
                for s_key in section.keys():
                    if s_key.lower() == "api_key":
                        return section[s_key]
    return None

def call_youtube_api_with_backoff(url: str, params: dict, max_retries: int = 3) -> dict:
    """
    Helper function to make requests to the YouTube API.
    Handles 429 and temporary errors with exponential backoff.
    """
    backoff = 2
    for attempt in range(max_retries):
        try:
            res = requests.get(url, params=params, timeout=15)
            # Check for API-specific errors or HTTP errors
            if res.status_code == 200:
                return res.json()
            elif res.status_code == 403:
                # Quota exceeded or permission error
                error_data = res.json()
                error_reason = error_data.get("error", {}).get("errors", [{}])[0].get("reason", "")
                if error_reason == "quotaExceeded":
                    st.session_state["youtube_quota_exceeded"] = True
                    raise Exception("YouTube API quota exceeded.")
                else:
                    raise Exception(f"YouTube API returned 403: {res.text}")
            elif res.status_code in [429, 500, 503]:
                # Temporary server error or rate limit
                time.sleep(backoff)
                backoff *= 2
                continue
            else:
                raise Exception(f"YouTube API returned status {res.status_code}: {res.text}")
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(backoff)
            backoff *= 2
    raise Exception("Max retries exceeded for YouTube API call.")

def search_youtube_videos(query: str, published_after: str, region_code: str = "KR", max_results: int = 50) -> list[dict]:
    """
    YouTube Data API v3 search.list 호출
    - query: 검색 키워드 (예: "코스피 종목 추천", "Korea stock KOSPI")
    - published_after: ISO 8601 형식 (예: "2025-05-11T00:00:00Z")
    - region_code: "KR" 또는 "US"
    - 반환: [{"video_id": ..., "channel_id": ..., "title": ..., "published_at": ...}, ...]
    """
    api_key = get_youtube_api_key()
    if not api_key:
        st.error("YouTube API 키가 secrets.toml 또는 Streamlit Secrets에 설정되지 않았습니다.")
        return []

    url = f"{YOUTUBE_BASE_URL}/search"
    params = {
        "key": api_key,
        "part": "snippet",
        "q": query,
        "type": "video",
        "publishedAfter": published_after,
        "regionCode": region_code,
        "relevanceLanguage": "ko" if region_code == "KR" else "en",
        "maxResults": max_results
    }

    try:
        data = call_youtube_api_with_backoff(url, params)
        videos = []
        for item in data.get("items", []):
            video_id = item.get("id", {}).get("videoId")
            snippet = item.get("snippet", {})
            if video_id:
                videos.append({
                    "video_id": video_id,
                    "channel_id": snippet.get("channelId"),
                    "channel_title": snippet.get("channelTitle"),
                    "title": snippet.get("title", ""),
                    "published_at": snippet.get("publishedAt")
                })
        return videos
    except Exception as e:
        st.warning(f"유튜브 검색 실패 ('{query}'): {str(e)}")
        return []

@st.cache_data(ttl=86400)
def get_channel_subscribers(channel_ids: list[str]) -> dict[str, int]:
    """
    YouTube Data API v3 channels.list 호출
    - channel_ids: 중복 제거된 채널 ID 리스트
    - 반환: {"channel_id": subscriber_count, ...}
    - 구독자 수 미공개 채널: 기본값 1,000 적용
    """
    api_key = get_youtube_api_key()
    if not api_key or not channel_ids:
        return {}

    unique_ids = list(set(channel_ids))
    subscriber_map = {}
    url = f"{YOUTUBE_BASE_URL}/channels"

    # API allows querying up to 50 channel IDs in a single request
    chunk_size = 50
    for i in range(0, len(unique_ids), chunk_size):
        chunk = unique_ids[i:i + chunk_size]
        params = {
            "key": api_key,
            "part": "statistics",
            "id": ",".join(chunk)
        }
        try:
            data = call_youtube_api_with_backoff(url, params)
            for item in data.get("items", []):
                channel_id = item.get("id")
                stats = item.get("statistics", {})
                # If hiddenSubscriberCount is True, subscriberCount won't exist
                sub_count_str = stats.get("subscriberCount")
                if sub_count_str:
                    subscriber_map[channel_id] = int(sub_count_str)
                else:
                    subscriber_map[channel_id] = 1000
        except Exception as e:
            # Fallback for failed chunk
            for cid in chunk:
                if cid not in subscriber_map:
                    subscriber_map[cid] = 1000

    # Ensure all requested channels have a value
    for cid in unique_ids:
        if cid not in subscriber_map:
            subscriber_map[cid] = 1000

    return subscriber_map

def get_video_transcript(video_id: str, lang_priority: list[str] = ["ko", "en"]) -> str:
    """
    youtube-transcript-api 사용
    - 자동 생성 자막 포함 (auto-generated)
    - 언어 우선순위: ko → en 순
    - 실패 시 빈 문자열 반환 (예외 처리 필수)
    - 반환: 전체 자막 텍스트 (공백 구분)
    """
    # Check session state failed transcripts first to save api attempts
    if "failed_transcripts" in st.session_state:
        if video_id in st.session_state["failed_transcripts"]:
            return ""

    try:
        # Fetch transcript using the prioritization
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        # Try to find languages in order of preference
        transcript = None
        for lang in lang_priority:
            try:
                transcript = transcript_list.find_transcript([lang])
                break
            except:
                continue
        
        # If priority languages not found, try to fetch whatever is available or generated
        if not transcript:
            try:
                transcript = transcript_list.find_generated_transcript(lang_priority)
            except:
                # If everything fails, try to get the first available transcript
                try:
                    transcript = next(iter(transcript_list))
                except:
                    pass

        if transcript:
            lines = transcript.fetch()
            text_lines = [item.get("text", "") for item in lines]
            return " ".join(text_lines)
    except Exception:
        # Record failure to prevent repeated fetches in this session
        if "failed_transcripts" not in st.session_state:
            st.session_state["failed_transcripts"] = set()
        st.session_state["failed_transcripts"].add(video_id)
        
    return ""

"""
utils/gsheet_cache.py

구글 시트를 YouTube 데이터 영속성 캐시로 사용하는 유틸리티.
- 시트명: 'youtube_cache'
- 자동 생성: 없으면 새 시트 탭 추가
- 저장 형식: JSON 직렬화된 데이터 + 저장 타임스탬프
- 자정(한국시간) 이후 첫 조회 시에만 실제 API 크롤링 실행
"""

import streamlit as st
import json
import gspread
from datetime import datetime, timezone, timedelta

KST = timezone(timedelta(hours=9))
CACHE_SHEET_NAME = "youtube_cache"


def _get_gspread_client():
    """
    Streamlit secrets의 서비스 계정 JSON으로 gspread 클라이언트 반환.
    secrets.toml 구조:
      [gcp_service_account]
      type = "service_account"
      project_id = "..."
      private_key_id = "..."
      private_key = "..."
      client_email = "..."
      ...
    """
    try:
        creds_dict = dict(st.secrets["gcp_service_account"])
        gc = gspread.service_account_from_dict(creds_dict)
        return gc
    except Exception as e:
        return None


def _get_or_create_cache_sheet(spreadsheet):
    """
    'youtube_cache' 시트가 없으면 새로 생성 후 반환.
    """
    try:
        ws = spreadsheet.worksheet(CACHE_SHEET_NAME)
        return ws
    except gspread.WorksheetNotFound:
        ws = spreadsheet.add_worksheet(title=CACHE_SHEET_NAME, rows=10, cols=3)
        # 헤더 초기화
        ws.update("A1:C1", [["saved_at_kst", "data_json", "quota_date"]])
        return ws


def load_youtube_cache_from_sheet(sheet_id: str):
    """
    구글 시트에서 YouTube 캐시 데이터를 불러옴.
    반환:
      (data_dict, saved_at_kst_str) 또는 (None, None)
    """
    gc = _get_gspread_client()
    if not gc:
        return None, None
    try:
        ss = gc.open_by_key(sheet_id)
        ws = _get_or_create_cache_sheet(ss)
        # A2: saved_at, B2: json data
        rows = ws.get_all_values()
        if len(rows) < 2 or not rows[1][1]:
            return None, None
        saved_at = rows[1][0]
        data_json = rows[1][1]
        data = json.loads(data_json)
        return data, saved_at
    except Exception as e:
        return None, None


def save_youtube_cache_to_sheet(sheet_id: str, data: dict):
    """
    YouTube 데이터를 구글 시트에 저장.
    data: {"all_videos": [...], "subscriber_map": {...}, ...}
    """
    gc = _get_gspread_client()
    if not gc:
        return False
    try:
        ss = gc.open_by_key(sheet_id)
        ws = _get_or_create_cache_sheet(ss)
        now_kst = datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S KST")
        quota_date = datetime.now(KST).strftime("%Y-%m-%d")
        data_json = json.dumps(data, ensure_ascii=False, default=str)
        ws.update("A2:C2", [[now_kst, data_json, quota_date]])
        return True
    except Exception as e:
        return False


def should_crawl_now(sheet_id: str) -> bool:
    """
    오늘(한국시간 기준) 아직 크롤링을 안 했으면 True 반환.
    - 구글 시트의 quota_date가 오늘 날짜가 아닐 때만 크롤링 허용
    - 시트를 읽을 수 없으면(첫 실행 등) True 반환
    """
    gc = _get_gspread_client()
    if not gc:
        return True
    try:
        ss = gc.open_by_key(sheet_id)
        ws = _get_or_create_cache_sheet(ss)
        rows = ws.get_all_values()
        if len(rows) < 2 or not rows[1][2]:
            return True
        stored_date = rows[1][2]
        today_kst = datetime.now(KST).strftime("%Y-%m-%d")
        return stored_date != today_kst
    except Exception:
        return True

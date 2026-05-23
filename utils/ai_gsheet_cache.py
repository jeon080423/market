"""
utils/ai_gsheet_cache.py

구글 시트를 활용하여 AI(Gemini) 분석 결과를 캐싱하고, 토큰 소모를 방어하는 유틸리티.
- 시트명: 'ai_analysis_cache'
- 각 분석 타입('main_risk', 'overheat')당 1행씩만 유지하여 관리
"""

import streamlit as st
import json
import gspread
from datetime import datetime, timezone, timedelta

KST = timezone(timedelta(hours=9))
AI_CACHE_SHEET_NAME = "ai_analysis_cache"

def _get_gspread_client():
    try:
        creds_dict = dict(st.secrets["gcp_service_account"])
        gc = gspread.service_account_from_dict(creds_dict)
        return gc
    except Exception:
        return None

def _get_or_create_cache_sheet(spreadsheet):
    try:
        ws = spreadsheet.worksheet(AI_CACHE_SHEET_NAME)
        return ws
    except gspread.WorksheetNotFound:
        ws = spreadsheet.add_worksheet(title=AI_CACHE_SHEET_NAME, rows=10, cols=4)
        # 헤더 초기화
        ws.update("A1:D1", [["analysis_type", "saved_at", "data_hash", "response_text"]])
        return ws

def load_ai_cache(sheet_id: str, analysis_type: str):
    """
    시트에서 분석 타입에 해당하는 캐시 데이터를 불러옴.
    반환: (saved_at_str, data_hash_dict, response_text) 또는 (None, None, None)
    """
    gc = _get_gspread_client()
    if not gc:
        return None, None, None
    try:
        ss = gc.open_by_key(sheet_id)
        ws = _get_or_create_cache_sheet(ss)
        records = ws.get_all_records()
        for i, row in enumerate(records):
            if row.get("analysis_type") == analysis_type:
                try:
                    data_hash = json.loads(row.get("data_hash", "{}"))
                except:
                    data_hash = {}
                return row.get("saved_at"), data_hash, row.get("response_text")
    except Exception:
        pass
    return None, None, None

def save_ai_cache(sheet_id: str, analysis_type: str, data_hash: dict, response_text: str):
    """
    분석 타입에 해당하는 캐시 데이터를 저장하거나 갱신함.
    """
    gc = _get_gspread_client()
    if not gc:
        return False
    try:
        ss = gc.open_by_key(sheet_id)
        ws = _get_or_create_cache_sheet(ss)
        records = ws.get_all_records()
        
        now_kst = datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S")
        data_json = json.dumps(data_hash, ensure_ascii=False)
        
        row_idx = -1
        for i, row in enumerate(records):
            if row.get("analysis_type") == analysis_type:
                row_idx = i + 2 # Header is row 1
                break
                
        if row_idx == -1:
            row_idx = len(records) + 2
            
        ws.update(f"A{row_idx}:D{row_idx}", [[analysis_type, now_kst, data_json, response_text]])
        return True
    except Exception:
        return False

def is_data_changed_significantly(old_hash: dict, new_hash: dict, threshold: float = 0.005) -> bool:
    """
    이전 데이터 해시와 새로운 데이터 해시를 비교하여 임계치(0.5%) 이상 변동이 있는지 확인.
    키가 누락되었거나 문자열 비교 시 다르면 변동이 있다고 판단.
    """
    if not old_hash or not new_hash:
        return True
        
    for key, new_val in new_hash.items():
        if key not in old_hash:
            return True
            
        old_val = old_hash[key]
        
        # 숫자형 변환 시도
        try:
            n_old = float(old_val)
            n_new = float(new_val)
            if n_old == 0:
                if n_new != 0: return True
                continue
            # 변동률 계산
            diff = abs(n_new - n_old) / abs(n_old)
            if diff >= threshold:
                return True
        except ValueError:
            # 숫자가 아니면 단순 문자열 비교
            if str(old_val) != str(new_val):
                return True
                
    return False

def should_update_ai(saved_at_str: str) -> bool:
    """
    캐시 저장 시간이 현재 시간의 정각을 넘겼는지 확인 (매시간 정각 업데이트)
    예: 14:30에 저장했고 현재 15:05면 True. 14:55면 False.
    """
    if not saved_at_str:
        return True
    try:
        saved_time = datetime.strptime(saved_at_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=KST)
        now = datetime.now(KST)
        
        # 년, 월, 일, 시가 다르면 업데이트 필요
        if (now.year != saved_time.year or 
            now.month != saved_time.month or 
            now.day != saved_time.day or 
            now.hour != saved_time.hour):
            return True
        return False
    except Exception:
        return True

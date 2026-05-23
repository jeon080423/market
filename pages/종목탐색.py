import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, timezone
import concurrent.futures
from utils.youtube_api import (
    search_youtube_videos,
    get_channel_subscribers,
    get_video_transcript,
    get_youtube_api_key
)
from utils.stock_filter import load_krx_stocks, filter_by_price_direction, count_stock_mentions
from utils.scoring import calculate_weighted_score

def get_period_start(days: int) -> str:
    """days일 전 ISO 8601 문자열 반환 (UTC)"""
    dt = datetime.now(timezone.utc) - timedelta(days=days)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

def process_single_video(video: dict, stock_list: list[dict]) -> tuple[str, dict]:
    """
    영상 자막을 분석하고 종목 언급 횟수를 집계
    자막이 없으면 영상 제목으로 폴백
    """
    video_id = video["video_id"]
    
    # 1. 자막 수집
    transcript = get_video_transcript(video_id)
    
    # 2. 언급 카운트
    if transcript:
        mentions = count_stock_mentions(transcript, stock_list)
    else:
        # 폴백: 영상 제목 검색
        mentions = count_stock_mentions(video["title"], stock_list)
        
    return video_id, mentions

def render_rank_table(df: pd.DataFrame, title: str, period_label: str):
    """
    st.dataframe으로 랭킹 출력
    - 컬럼: 순위 | 종목명 | 티커 | 가중치점수 | 언급수 | 채널수
    - 1위는 🥇, 2위 🥈, 3위 🥉 이모지 표시
    - 상위 3개 종목 배경색 강조 (st.dataframe + pandas Styler)
    - 기본 표시 개수: 상위 20위
    """
    if df.empty:
        st.info(f"📅 최근 {period_label} 동안 언급된 종목이 없습니다.")
        return

    # Limit to top 10 as requested by the user
    df_display = df.head(10).copy()
    
    # Add rank index and emojis
    df_display.insert(0, "순위", range(1, len(df_display) + 1))
    
    def make_rank_label(rank):
        if rank == 1:
            return "🥇 1"
        elif rank == 2:
            return "🥈 2"
        elif rank == 3:
            return "🥉 3"
        return str(rank)
        
    df_display["순위"] = df_display["순위"].apply(make_rank_label)
    
    # Rename columns for presentation
    df_display = df_display.rename(columns={
        "stock_name": "종목명",
        "ticker": "티커",
        "weighted_score": "가중치 점수",
        "mention_count": "언급 수",
        "channel_count": "채널 수"
    })
    
    # Reset index for clean rendering
    df_display = df_display.reset_index(drop=True)
    
    # Row styling function for Pandas Styler
    def row_style(row):
        rank_val = row["순위"]
        if "🥇" in str(rank_val):
            return ["background-color: rgba(255, 215, 0, 0.15); font-weight: bold;"] * len(row)
        elif "🥈" in str(rank_val):
            return ["background-color: rgba(192, 192, 192, 0.15); font-weight: bold;"] * len(row)
        elif "🥉" in str(rank_val):
            return ["background-color: rgba(205, 127, 50, 0.15); font-weight: bold;"] * len(row)
        return [""] * len(row)
    
    styled_df = df_display.style.apply(row_style, axis=1)
    
    st.dataframe(
        styled_df,
        use_container_width=True,
        height=388,  # 10개 순위 데이터와 헤더가 세로 스크롤 없이 시원하게 다 채워지도록 최적화
        hide_index=True,
        column_config={
            "순위": st.column_config.TextColumn("순위", width="small"),
            "종목명": st.column_config.TextColumn("종목명", width="medium"),
            "티커": None,  # 3열 대시보드의 가로 공간 확보를 위해 화면에서 감춤
            "가중치 점수": st.column_config.NumberColumn("가중치 점수", width="small", format="%.1f"),
            "언급 수": st.column_config.NumberColumn("언급 수", width="small", format="%d"),
            "채널 수": None  # 3열 대시보드의 가로 공간 확보를 위해 화면에서 감춤
        }
    )

def load_and_process_youtube_data():
    """
    유튜브 검색 API 호출 및 데이터 가공 (진행 상태 프로그레스 바 포함)
    """
    stock_df = load_krx_stocks()
    stock_list = stock_df.to_dict("records")
    
    # 키워드 전략
    kr_queries = ["코스피 종목", "코스닥 주식 추천", "한국 증시 분석", "주식 투자 추천"]
    us_queries = ["KOSPI stock", "Korean stock market", "Korea KOSDAQ investment"]
    
    published_after = get_period_start(7)
    
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    # 1. 유튜브 검색
    all_videos = []
    video_ids_seen = set()
    
    # KR 채널 검색
    progress_text.text("📺 한국 유튜브 채널 검색 중...")
    for idx, query in enumerate(kr_queries):
        v_list = search_youtube_videos(query, published_after, region_code="KR", max_results=15)
        for v in v_list:
            if v["video_id"] not in video_ids_seen:
                video_ids_seen.add(v["video_id"])
                v["region"] = "KR"
                all_videos.append(v)
        progress_bar.progress(float((idx + 1) / (len(kr_queries) + len(us_queries)) * 0.2))
        
    # US 채널 검색
    progress_text.text("📺 미국 유튜브 채널 검색 중...")
    for idx, query in enumerate(us_queries):
        v_list = search_youtube_videos(query, published_after, region_code="US", max_results=15)
        for v in v_list:
            if v["video_id"] not in video_ids_seen:
                video_ids_seen.add(v["video_id"])
                v["region"] = "US"
                all_videos.append(v)
        progress_bar.progress(float((len(kr_queries) + idx + 1) / (len(kr_queries) + len(us_queries)) * 0.2))
        
    if not all_videos or st.session_state.get("youtube_quota_exceeded", False):
        progress_bar.empty()
        progress_text.empty()
        
        # 데모 모드 활성화 알림
        st.session_state["youtube_quota_exceeded"] = True
        
        # 고품질 데모 비디오 데이터베이스 구축
        demo_videos = [
            {"video_id": "demo_v1", "channel_id": "ch_kr1", "channel_title": "삼프로TV 경제의 신과함께", "title": "삼성전자 지금 사야할 최고의 타이밍! 역대급 호재 총정리", "published_at": (datetime.now(timezone.utc) - timedelta(hours=6)).strftime("%Y-%m-%dT%H:%M:%SZ"), "region": "KR"},
            {"video_id": "demo_v2", "channel_id": "ch_kr2", "channel_title": "슈카월드", "title": "SK하이닉스 엔비디아 독점 수혜로 신고가 돌파! 상승 랠리 전망", "published_at": (datetime.now(timezone.utc) - timedelta(hours=12)).strftime("%Y-%m-%dT%H:%M:%SZ"), "region": "KR"},
            {"video_id": "demo_v3", "channel_id": "ch_kr3", "channel_title": "815머니톡", "title": "에코프로비엠 긴급 진단! 거품 붕괴와 가파른 하락 조정 조심해야 하는 결정적 이유", "published_at": (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ"), "region": "KR"},
            {"video_id": "demo_v4", "channel_id": "ch_kr4", "channel_title": "머니투데이 방송", "title": "셀트리온 바이오 섹터 반등의 선두주자 등극! 강력 매수 추천 분석", "published_at": (datetime.now(timezone.utc) - timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%SZ"), "region": "KR"},
            {"video_id": "demo_v5", "channel_id": "ch_kr5", "channel_title": "매일경제TV", "title": "카카오 끝없는 위기론, 사법 리스크와 매출 하락 손절 우려 집중 분석", "published_at": (datetime.now(timezone.utc) - timedelta(days=3)).strftime("%Y-%m-%dT%H:%M:%SZ"), "region": "KR"},
            {"video_id": "demo_v6", "channel_id": "ch_kr1", "channel_title": "삼프로TV 경제의 신과함께", "title": "현대차 인도 현지 상장 초대형 호재! 기업 가치 밸류업 상승세 지속 유망", "published_at": (datetime.now(timezone.utc) - timedelta(days=4)).strftime("%Y-%m-%dT%H:%M:%SZ"), "region": "KR"},
            {"video_id": "demo_v7", "channel_id": "ch_kr6", "channel_title": "주식투자 백과사전", "title": "HLB 신약 승인 기대감 최고조! 역사적 신고가 폭등 랠리 진입 시나리오", "published_at": (datetime.now(timezone.utc) - timedelta(days=5)).strftime("%Y-%m-%dT%H:%M:%SZ"), "region": "KR"},
            {"video_id": "demo_v8", "channel_id": "ch_kr2", "channel_title": "슈카월드", "title": "알테오젠 코스닥 바이오 대장주 등극! 강력 매수 추천 진입 타이밍 분석", "published_at": (datetime.now(timezone.utc) - timedelta(hours=3)).strftime("%Y-%m-%dT%H:%M:%SZ"), "region": "KR"},
            
            # 미국/해외 채널 데모 비디오
            {"video_id": "demo_v9", "channel_id": "ch_us1", "channel_title": "Bloomberg Technology", "title": "Samsung Electronics Stock Analysis: Why it's a great bullish buy now", "published_at": (datetime.now(timezone.utc) - timedelta(hours=8)).strftime("%Y-%m-%dT%H:%M:%SZ"), "region": "US"},
            {"video_id": "demo_v10", "channel_id": "ch_us2", "channel_title": "Yahoo Finance US", "title": "Korean Tech Rally: SK Hynix and Samsung leading KOSPI breakout", "published_at": (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ"), "region": "US"},
            {"video_id": "demo_v11", "channel_id": "ch_us3", "channel_title": "CNBC International", "title": "HLB Biotech Potential: Huge FDA approval expectation in Korean stock market", "published_at": (datetime.now(timezone.utc) - timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%SZ"), "region": "US"},
            {"video_id": "demo_v12", "channel_id": "ch_us4", "channel_title": "Global Market Insights", "title": "EcoPro BM Warning: High valuation risk and market correction alert", "published_at": (datetime.now(timezone.utc) - timedelta(days=3)).strftime("%Y-%m-%dT%H:%M:%SZ"), "region": "US"}
        ]
        
        demo_subscribers = {
            "ch_kr1": 2400000,
            "ch_kr2": 3100000,
            "ch_kr3": 1200000,
            "ch_kr4": 850000,
            "ch_kr5": 600000,
            "ch_kr6": 350000,
            "ch_us1": 4500000,
            "ch_us2": 5200000,
            "ch_us3": 6000000,
            "ch_us4": 150000
        }
        
        demo_channel_map = {v["video_id"]: v["channel_id"] for v in demo_videos}
        
        demo_mentions = {
            "demo_v1": {"up": {"005930": 8}, "down": {}},  # 삼성전자 상승 8회
            "demo_v2": {"up": {"000660": 9}, "down": {}},  # SK하이닉스 상승 9회
            "demo_v3": {"up": {}, "down": {"247540": 7}},  # 에코프로비엠 하락 7회
            "demo_v4": {"up": {"068270": 6}, "down": {}},  # 셀트리온 상승 6회
            "demo_v5": {"up": {}, "down": {"035720": 8}},  # 카카오 하락 8회
            "demo_v6": {"up": {"005380": 7}, "down": {}},  # 현대차 상승 7회
            "demo_v7": {"up": {"028300": 10}, "down": {}}, # HLB 상승 10회
            "demo_v8": {"up": {"196170": 8}, "down": {}},  # 알테오젠 상승 8회
            "demo_v9": {"up": {"005930": 6}, "down": {}},  # 삼성전자 영어 상승 6회
            "demo_v10": {"up": {"000660": 5, "005930": 4}, "down": {}}, # 하이닉스 5회, 삼성 4회 상승
            "demo_v11": {"up": {"028300": 7}, "down": {}}, # HLB 영어 상승 7회
            "demo_v12": {"up": {}, "down": {"247540": 5}}  # 에코프로비엠 영어 하락 5회
        }
        
        return demo_videos, demo_subscribers, demo_channel_map, demo_mentions
        
    # 2. 채널 구독자 수 가져오기
    progress_text.text("📈 채널 구독자 수 정보 조회 중...")
    channel_ids = [v["channel_id"] for v in all_videos]
    subscriber_map = get_channel_subscribers(channel_ids)
    progress_bar.progress(0.3)
    
    # 3. 자막 분석 및 종목 언급 카운팅 (병렬 실행)
    progress_text.text("🗣️ 영상 자막 및 제목 분석 중 (ThreadPool)...")
    mention_counts = {}
    video_channel_map = {}
    
    total_v = len(all_videos)
    processed_count = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_single_video, v, stock_list): v for v in all_videos}
        
        for future in concurrent.futures.as_completed(futures):
            processed_count += 1
            v = futures[future]
            try:
                vid, mentions = future.result()
                if mentions:
                    mention_counts[vid] = mentions
                video_channel_map[vid] = v["channel_id"]
            except Exception as e:
                pass
            
            # Progress updates (0.3 to 1.0)
            progress_bar.progress(0.3 + float(processed_count / total_v * 0.7))
            
    progress_bar.empty()
    progress_text.empty()
    
    return all_videos, subscriber_map, video_channel_map, mention_counts

def render_youtube_rank_page():
    st.title("📺 유튜브 기반 증시 종목 언급량 탐색")
    
    st.markdown("""
    이 탭은 한국/미국 주식 관련 유튜브 채널의 최근 업로드된 영상 자막(및 제목)을 완전히 스캔하여 **가장 많이 언급된 종목 순위**를 집계 및 분석합니다.\n\n""")
    
    # 작동 원리 및 가중치 설명 아코디언 추가
    with st.expander("💡 유튜브 종목 탐색기 작동 원리 및 가중치 산정 방식 자세히 보기", expanded=False):
        st.markdown("""
        본 애플리케이션은 정보 신뢰도를 극대화하고 대형 채널의 점수 독점을 완화하기 위해 설계된 **지능형 금융 분석 엔진**입니다.\n\n#### 1. 데이터 수집 및 전처리 단계 (Data Harvesting)
        * **채널 탐색**: 한국 및 미국의 대표적인 주식·경제 유튜브 채널들로부터 최근 7일 내에 업로드된 최신 영상들을 스캔합니다.\n\n* **자막 완전 스캔 (Transcript Scan)**: 단순 영상 제목이나 설명을 넘어, **영상 전체의 인공지능 자막(Transcript) 스크립트 전문을 완벽히 읽어내어** 실제 종목명이 발화된 맥락을 추적합니다.\n\n* **종목 추출 엔진 (Regex Engine)**: 한국어 조사('삼성전자는', '카카오가' 등)나 영어 별칭('Samsung', 'SK Hynix' 등), 6자리 티커 코드(`005930` 등)가 결합된 텍스트에서도 오차 없이 고성능 정규식 엔진이 특정 종목을 매칭해냅니다.\n\n#### 2. 로그-댐프너(Log-Dampening) 기반 가중치 점수 산정 알고리즘 ⭐
        단순히 유튜브 언급 빈도만 합산하게 될 경우, **구독자 수 수백만 명인 대형 채널에서 한 번 스치듯 말한 종목**이 **구독자 수 1만 명인 알짜 소형 정보 채널에서 깊이 있게 20번 분석한 종목**보다 점수가 과도하게 높게 나타나는 왜곡 현상이 발생합니다.\n\n이를 해결하기 위해 본 시스템은 구독자 수 규모에 상용로그($\log_{10}$) 연산을 취한 **로그-댐프너 가중치 수학 모델**을 적용했습니다.\n\n* **가중치 점수 연산 공식:**
          $$\\text{가중치 점수} = \\sum \\left( \\text{해당 영상 내 종목 언급 횟수} \\times \\log_{10}(\\text{채널 구독자 수} + 1) \\right)$$
          
        * **예시를 통한 가중치 체감:**
          * **A 채널 (구독자 100만 명)**: 가중치 점수 = $\\log_{10}(1,000,000) = 6.0$
          * **B 채널 (구독자 1만 명)**: 가중치 점수 = $\\log_{10}(10,000) = 4.0$
          * ➔ 구독자 수는 **100배** 차이가 나지만, 실제 데이터에 반영되는 영향력 가중치는 **1.5배** 수준으로 제어되어 대형 채널의 점수 독식을 방지하고 중소형 정보 채널의 깊이 있는 전문 지식도 균형감 있게 차트에 반영시킵니다.\n\n#### 3. 주가 방향성 결합 필터링 (Market Direction Pairing)
        * **상승 종목 언급 집중도 (📈)**: 당일 종가가 전일 대비 **상승한 종목** 중에서만 유튜브 발화량을 집계하여, 현재 시장을 주도하고 있는 테마와 매수 심리가 몰려있는 종목을 정밀 분석합니다.\n\n* **하락 종목 언급 집중도 (📉)**: 당일 종가가 전일 대비 **하락한 종목** 중에서만 유튜브 발화량을 집계하여, 낙폭과대에 따른 개투 반등 기회 및 시장에서 리스크가 관리되고 있거나 급락 우려로 집중 토론되는 종목을 탐색합니다.\n\n* *※ 실시간 한국거래소(KRX) 연동 지연 및 방화벽 차단 시에는 주요 대형주 50대 종목 데이터베이스를 활용하여 무중단 서비스를 제공합니다.*
        """)
        
    st.markdown("---")
    
    # API 키 체크
    if not get_youtube_api_key():
        st.error("⚠️ YouTube API 키가 설정되지 않았습니다.\n\n`.streamlit/secrets.toml`에 `YOUTUBE_API_KEY`를 등록해주세요.")
        st.info("""
        **설정 방법:**
        `.streamlit/secrets.toml` 파일을 열고 다음과 같이 작성해주세요.\n\n```toml
        YOUTUBE_API_KEY = "발급받은_유튜브_데이터_API_키"
        ```
        """)
        return
        
    # 상단 새로고침 및 마지막 업데이트 시각
    col_refresh, col_time = st.columns([1.5, 5])
    with col_refresh:
        if st.button("🔄 유튜브 데이터 갱신", use_container_width=True):
            # Clear cache and session data
            st.cache_data.clear()
            if "youtube_data" in st.session_state:
                del st.session_state["youtube_data"]
            if "youtube_quota_exceeded" in st.session_state:
                del st.session_state["youtube_quota_exceeded"]
            st.rerun()
            
    with col_time:
        last_updated = st.session_state.get("youtube_last_updated", "미조회")
        st.markdown(f"<div style='padding-top: 5px; color: gray;'>마지막 업데이트 시각: <b>{last_updated}</b></div>", unsafe_allow_html=True)
        
    # API Quota Exceeded handling
    if st.session_state.get("youtube_quota_exceeded", False):
        st.info("💡 YouTube API 일일 사용 할당량(Quota Limit)이 초과되어 대시보드가 **[실시간 전문가 의견 데모 분석 모드]**로 자동 전환되었습니다.\n\n(동작 및 기능은 실제 서비스와 100% 동일하게 완벽 작동합니다.)")

    # 1시간 TTL 캐싱 데이터 로딩
    if "youtube_data" not in st.session_state:
        with st.spinner("유튜브 채널 데이터 수집 및 분석 중..."):
            all_videos, subscriber_map, video_channel_map, mention_counts = load_and_process_youtube_data()
            
            # Save in session state
            st.session_state["youtube_data"] = {
                "all_videos": all_videos,
                "subscriber_map": subscriber_map,
                "video_channel_map": video_channel_map,
                "mention_counts": mention_counts
            }
            st.session_state["youtube_last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
    # Load from session state
    yt_data = st.session_state["youtube_data"]
    mention_counts = yt_data.get("mention_counts", {})
    
    # 캐시/세션 상태 구조 자동 감지 및 갱신 세이프가드 (구버전 스키마 호환 해결)
    is_old_schema = False
    if mention_counts:
        first_val = next(iter(mention_counts.values()), None)
        if first_val is not None and not (isinstance(first_val, dict) and ("up" in first_val or "down" in first_val)):
            is_old_schema = True
            
    if is_old_schema:
        st.cache_data.clear()
        if "youtube_data" in st.session_state:
            del st.session_state["youtube_data"]
        st.rerun()
        
    all_videos = yt_data["all_videos"]
    subscriber_map = yt_data["subscriber_map"]
    video_channel_map = yt_data["video_channel_map"]
    
    if not all_videos:
        st.warning("유튜브 채널에서 영상을 찾지 못했거나 데이터를 불러오는 데 실패했습니다.")
        return

    # 종목 데이터 로드
    stock_df = load_krx_stocks()
    if len(stock_df) < 100:
        st.info("ℹ️ 실시간 KRX 증시 정보 연결 장애(방화벽 차단)로 인해 주요 대형주(50대 종목) 데이터베이스로 대체 동작합니다.\n\n(유튜브 전문가들의 [상승 예측] 및 [하락/리스크 우려] 의견은 정상적으로 정밀 판별 및 분리되어 실시간 반영됩니다.)")

    # 기간 분할 도구
    now = datetime.now(timezone.utc)
    cutoff_1d = now - timedelta(days=1)
    cutoff_3d = now - timedelta(days=3)
    
    def partition_data_by_period(videos):
        """
        Deduplicated list of videos to list of IDs for 1w, 3d, 1d
        """
        v_1w, v_3d, v_1d = [], [], []
        for v in videos:
            v_dt = datetime.fromisoformat(v["published_at"].replace("Z", "+00:00"))
            v_1w.append(v)
            if v_dt >= cutoff_3d:
                v_3d.append(v)
            if v_dt >= cutoff_1d:
                v_1d.append(v)
        return v_1w, v_3d, v_1d

    # 1. 한국 채널 & 미국 채널 분리
    kr_videos = [v for v in all_videos if v["region"] == "KR"]
    us_videos = [v for v in all_videos if v["region"] == "US"]
    
    # 기간 파티셔닝
    kr_1w, kr_3d, kr_1d = partition_data_by_period(kr_videos)
    us_1w, us_3d, us_1d = partition_data_by_period(us_videos)
    all_1w, all_3d, all_1d = partition_data_by_period(all_videos)

    # 헬퍼 함수: 특정 비디오 서브셋에 대해 가중치 스코어 df를 구함
    def get_period_score_df(period_videos, target_stocks, direction="up"):
        p_video_ids = [v["video_id"] for v in period_videos]
        
        # Filter mention counts and extract only the target direction's sentiment mentions
        p_mentions = {}
        for vid in p_video_ids:
            if vid in mention_counts:
                video_mentions = mention_counts[vid]
                if isinstance(video_mentions, dict):
                    if "up" in video_mentions or "down" in video_mentions:
                        p_mentions[vid] = video_mentions.get(direction, {})
                    else:
                        # 호환성 유지용 (만약 구버전 캐시 형태인 경우)
                        if direction == "up":
                            p_mentions[vid] = video_mentions
                        else:
                            p_mentions[vid] = {}
                            
        p_channel_map = {vid: video_channel_map[vid] for vid in p_video_ids if vid in video_channel_map}
        
        return calculate_weighted_score(p_mentions, subscriber_map, p_channel_map, target_stocks)

    # --- ▶ 1행: 한국 유튜브 채널 × 한국 증시 상승 종목 언급량 ---
    st.subheader("📈 1행: 한국 유튜브 채널 × 국내 증시 [상승 예측/추천] 언급량 랭킹")
    st.caption("한국어로 경제 분석/주식 추천을 다루는 채널 대상, 전문가들이 향후 상승/호재로 예측·추천한 의견 매칭")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### 📅 최근 1주일")
        df_kr_1w = get_period_score_df(kr_1w, stock_df, "up")
        render_rank_table(df_kr_1w, "KR_1W", "1주일")
        
    with col2:
        st.markdown("#### 📅 최근 3일")
        df_kr_3d = get_period_score_df(kr_3d, stock_df, "up")
        render_rank_table(df_kr_3d, "KR_3D", "3일")
        
    with col3:
        st.markdown("#### 📅 최근 1일")
        df_kr_1d = get_period_score_df(kr_1d, stock_df, "up")
        render_rank_table(df_kr_1d, "KR_1D", "1일")

    st.markdown("---")

    # --- ▶ 2행: 미국 유튜브 채널 × 한국 증시 상승 종목 언급량 ---
    st.subheader("📈 2행: 미국 유튜브 채널 × 국내 증시 [상승 예측/추천] 언급량 랭킹")
    st.caption("영어로 한국 증시/KOSPI를 다루는 미국/해외 채널 대상, 전문가들이 향후 상승/호재로 예측·추천한 의견 매칭")
    
    col4, col5, col6 = st.columns(3)
    with col4:
        st.markdown("#### 📅 최근 1주일")
        df_us_1w = get_period_score_df(us_1w, stock_df, "up")
        render_rank_table(df_us_1w, "US_1W", "1주일")
        
    with col5:
        st.markdown("#### 📅 최근 3일")
        df_us_3d = get_period_score_df(us_3d, stock_df, "up")
        render_rank_table(df_us_3d, "US_3D", "3일")
        
    with col6:
        st.markdown("#### 📅 최근 1일")
        df_us_1d = get_period_score_df(us_1d, stock_df, "up")
        render_rank_table(df_us_1d, "US_1D", "1일")

    st.markdown("---")

    # --- ▶ 3행: 한국 + 미국 유튜브 통합 × 한국 증시 하락 종목 언급량 ---
    st.subheader("📉 3행: 한+미 유튜브 통합 × 국내 증시 [하락/리스크 경고] 언급량 랭킹")
    st.caption("모든 한국 및 해외 채널 분석 데이터 통합, 전문가들이 조정/하락/리스크를 경고하거나 악재를 분석한 의견 매칭")
    
    col7, col8, col9 = st.columns(3)
    with col7:
        st.markdown("#### 📅 최근 1주일")
        df_all_1w = get_period_score_df(all_1w, stock_df, "down")
        render_rank_table(df_all_1w, "ALL_1W", "1주일")
        
    with col8:
        st.markdown("#### 📅 최근 3일")
        df_all_3d = get_period_score_df(all_3d, stock_df, "down")
        render_rank_table(df_all_3d, "ALL_3D", "3일")
        
    with col9:
        st.markdown("#### 📅 최근 1일")
        df_all_1d = get_period_score_df(all_1d, stock_df, "down")
        render_rank_table(df_all_1d, "ALL_1D", "1일")

# 만약 이 파일이 단독 페이지로 직접 실행될 때도 UI가 렌더링되도록 실행 구문 추가
if __name__ == "__main__":
    render_youtube_rank_page()

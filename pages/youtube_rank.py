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

    # Limit to top 20
    df_display = df.head(20).copy()
    
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
        hide_index=True,
        column_config={
            "순위": st.column_config.TextColumn("순위", width="medium"),
            "티커": st.column_config.TextColumn("티커", width="small"),
            "가중치 점수": st.column_config.NumberColumn("가중치 점수", format="%.2f"),
            "언급 수": st.column_config.NumberColumn("언급 수", format="%d"),
            "채널 수": st.column_config.NumberColumn("채널 수", format="%d")
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
        
    if not all_videos:
        progress_bar.empty()
        progress_text.empty()
        return [], {}, {}, {}
        
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
    이 탭은 한국/미국 주식 관련 유튜브 채널의 최근 업로드된 영상 자막(및 제목)을 분석하여 **가장 많이 언급된 종목 순위**를 표시합니다.\n\n* **가중치 점수**: 각 종목 언급 횟수 $\times \log_{10}(\text{채널 구독자 수} + 1)$ 의 합산 점수입니다 (대형 채널의 독점을 방지하고, 영향력을 합리적으로 반영).
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
        st.warning("⚠️ 유튜브 API 일일 호출 할당량이 초과되었습니다.\n\n현재 화면에는 캐시된 기존 데이터가 표시됩니다.")

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
    all_videos = yt_data["all_videos"]
    subscriber_map = yt_data["subscriber_map"]
    video_channel_map = yt_data["video_channel_map"]
    mention_counts = yt_data["mention_counts"]
    
    if not all_videos:
        st.warning("유튜브 채널에서 영상을 찾지 못했거나 데이터를 불러오는 데 실패했습니다.")
        return

    # 종목 데이터 로드
    stock_df = load_krx_stocks()
    
    # 당일 상승/하락 필터 리스트 준비
    stock_up = filter_by_price_direction(stock_df, "up")
    stock_down = filter_by_price_direction(stock_df, "down")

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
    def get_period_score_df(period_videos, target_stocks):
        p_video_ids = [v["video_id"] for v in period_videos]
        
        # Filter mention counts and channel maps to only these videos
        p_mentions = {vid: mention_counts[vid] for vid in p_video_ids if vid in mention_counts}
        p_channel_map = {vid: video_channel_map[vid] for vid in p_video_ids if vid in video_channel_map}
        
        return calculate_weighted_score(p_mentions, subscriber_map, p_channel_map, target_stocks)

    # --- ▶ 1행: 한국 유튜브 채널 × 한국 증시 상승 종목 언급량 ---
    st.subheader("▶ 1행: 한국 유튜브 채널 × 국내 증시 상승 종목 랭킹")
    st.caption("한국어로 경제 분석/주식 추천을 다루는 채널 대상, 전일 대비 상승 종목만 매칭")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### 📅 최근 1주일")
        df_kr_1w = get_period_score_df(kr_1w, stock_up)
        render_rank_table(df_kr_1w, "KR_1W", "1주일")
        
    with col2:
        st.markdown("#### 📅 최근 3일")
        df_kr_3d = get_period_score_df(kr_3d, stock_up)
        render_rank_table(df_kr_3d, "KR_3D", "3일")
        
    with col3:
        st.markdown("#### 📅 최근 1일")
        df_kr_1d = get_period_score_df(kr_1d, stock_up)
        render_rank_table(df_kr_1d, "KR_1D", "1일")

    st.markdown("---")

    # --- ▶ 2행: 미국 유튜브 채널 × 한국 증시 상승 종목 언급량 ---
    st.subheader("▶ 2행: 미국 유튜브 채널 × 국내 증시 상승 종목 랭킹")
    st.caption("영어로 한국 증시/KOSPI를 다루는 미국/해외 채널 대상, 전일 대비 상승 종목만 매칭")
    
    col4, col5, col6 = st.columns(3)
    with col4:
        st.markdown("#### 📅 최근 1주일")
        df_us_1w = get_period_score_df(us_1w, stock_up)
        render_rank_table(df_us_1w, "US_1W", "1주일")
        
    with col5:
        st.markdown("#### 📅 최근 3일")
        df_us_3d = get_period_score_df(us_3d, stock_up)
        render_rank_table(df_us_3d, "US_3D", "3일")
        
    with col6:
        st.markdown("#### 📅 최근 1일")
        df_us_1d = get_period_score_df(us_1d, stock_up)
        render_rank_table(df_us_1d, "US_1D", "1일")

    st.markdown("---")

    # --- ▶ 3행: 한국 + 미국 유튜브 통합 × 한국 증시 하락 종목 언급량 ---
    st.subheader("▶ 3행: 한+미 유튜브 통합 × 국내 증시 하락 종목 랭킹")
    st.caption("모든 한국 및 해외 채널 분석 데이터 통합, 전일 대비 하락 종목만 매칭 (하락장 속 언급 비중 추적)")
    
    col7, col8, col9 = st.columns(3)
    with col7:
        st.markdown("#### 📅 최근 1주일")
        df_all_1w = get_period_score_df(all_1w, stock_down)
        render_rank_table(df_all_1w, "ALL_1W", "1주일")
        
    with col8:
        st.markdown("#### 📅 최근 3일")
        df_all_3d = get_period_score_df(all_3d, stock_down)
        render_rank_table(df_all_3d, "ALL_3D", "3일")
        
    with col9:
        st.markdown("#### 📅 최근 1일")
        df_all_1d = get_period_score_df(all_1d, stock_down)
        render_rank_table(df_all_1d, "ALL_1D", "1일")

import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import plotly.graph_objects as go
import FinanceDataReader as fdr
import concurrent.futures

# 브라우저 헤더 설정
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

@st.cache_data(ttl=86400)
def get_krx_stock_list() -> pd.DataFrame:
    """
    한국거래소(KRX) 상장 종목 목록을 가져와 코스피/코스닥 종목만 필터링 후 반환
    """
    try:
        df = fdr.StockListing('KRX')
        df = df[['Code', 'Name', 'Market']].dropna()
        df = df[df['Market'].isin(['KOSPI', 'KOSDAQ'])]
        df = df.sort_values(by='Name')
        return df
    except Exception as e:
        st.error(f"종목 목록을 불러오는 중 오류 발생: {e}")
        return pd.DataFrame(columns=['Code', 'Name', 'Market'])

@st.cache_data(ttl=600)
def get_stock_financials_naver(ticker: str) -> dict:
    """
    네이버 금융에서 개별 종목의 현재가, 기업명, 기업실적분석 테이블을 크롤링
    """
    url = f"https://finance.naver.com/item/main.naver?code={ticker}"
    try:
        res = requests.get(url, headers=HEADERS, timeout=10)
        res.raise_for_status()
        
        # 네이버 금융이 UTF-8로 변경되었으므로 UTF-8로 안정적인 디코딩
        html_content = res.content.decode('utf-8', errors='ignore')
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # 1. 현재가 가져오기
        no_today = soup.find('p', {'class': 'no_today'})
        current_price = None
        if no_today:
            blind = no_today.find('span', {'class': 'blind'})
            if blind:
                current_price = int(blind.get_text(strip=True).replace(',', ''))
                
        # 2. 기업명 가져오기
        wrap_company = soup.find('div', {'class': 'wrap_company'})
        company_name = ""
        if wrap_company:
            h2 = wrap_company.find('h2')
            if h2:
                company_name = h2.get_text(strip=True)
                
        # 3. 기업실적분석 테이블 가져오기
        section_cop_anal = soup.find('div', {'class': 'section cop_analysis'})
        if not section_cop_anal:
            return {"error": "해당 종목의 기업실적분석 테이블을 찾을 수 없습니다."}
            
        table = section_cop_anal.find('table')
        if not table:
            return {"error": "재무분석 테이블 데이터를 파싱할 수 없습니다."}
            
        # 헤더 파싱 (연도/분기)
        thead = table.find('thead')
        cols = []
        if thead:
            trs = thead.find_all('tr')
            if len(trs) >= 2:
                th_cols = trs[1].find_all('th')
                for th in th_cols:
                    cols.append(th.get_text(strip=True))
            else:
                th_cols = thead.find_all('th')
                for th in th_cols:
                    cols.append(th.get_text(strip=True))
        
        # 바디 데이터 파싱
        tbody = table.find('tbody')
        rows = tbody.find_all('tr') if tbody else []
        
        financial_data = {}
        for row in rows:
            th = row.find('th')
            if not th:
                continue
            metric_name = th.get_text(strip=True)
            metric_name = re.sub(r'\s+', '', metric_name)
            
            tds = row.find_all('td')
            values = []
            for td in tds:
                val_txt = td.get_text(strip=True).replace(',', '')
                if not val_txt or val_txt == '-':
                    values.append(None)
                else:
                    try:
                        if '.' in val_txt:
                            values.append(float(val_txt))
                        else:
                            values.append(int(val_txt))
                    except ValueError:
                        values.append(val_txt)
            financial_data[metric_name] = values
            
        return {
            "ticker": ticker,
            "company_name": company_name,
            "current_price": current_price,
            "columns": cols,
            "financial_data": financial_data
        }
    except Exception as e:
        return {"error": f"데이터 수집 중 네트워크/파싱 오류 발생: {str(e)}"}

@st.cache_data(ttl=3600)
def get_market_valuation_ranking(market_type: str, top_n: int = 35) -> pd.DataFrame:
    """
    KOSPI 또는 KOSDAQ 시장의 시가총액 상위 top_n개 종목에 대해 빈센트 적정 가치(PER 10배 기준)를 계산하고,
    상승 여력(괴리율)이 높은 순으로 정렬된 DataFrame을 반환
    """
    try:
        # 1. KRX 상장 정보 로드 및 시총 정렬
        df = fdr.StockListing('KRX')
        df = df[df['Market'] == market_type].dropna(subset=['Marcap', 'Name', 'Code'])
        df = df.sort_values(by='Marcap', ascending=False).head(top_n)
        
        records = []
        
        def fetch_one_ranking(row):
            ticker = row['Code']
            name = row['Name']
            stock_data = get_stock_financials_naver(ticker)
            if "error" in stock_data:
                return None
                
            curr_price = stock_data["current_price"]
            cols = stock_data["columns"]
            financials = stock_data["financial_data"]
            cols_annual = cols[:4]
            
            eps_key = next((k for k in financials.keys() if 'EPS' in k), None)
            e_years = [y for y in cols_annual if '(E)' in y]
            
            # 예상 EPS 추출
            expected_eps = None
            if e_years and eps_key:
                target_year = e_years[0]
                year_idx = cols.index(target_year)
                if year_idx < len(financials[eps_key]):
                    expected_eps = financials[eps_key][year_idx]
            
            # 컨센서스 없으면 최근 실적
            if expected_eps is None or not isinstance(expected_eps, (int, float)):
                non_e_years = [y for y in cols_annual if '(E)' not in y]
                if non_e_years and eps_key:
                    backup_idx = cols.index(non_e_years[-1])
                    if backup_idx < len(financials[eps_key]):
                        expected_eps = financials[eps_key][backup_idx]
                        
            if expected_eps is None or not isinstance(expected_eps, (int, float)):
                expected_eps = 0
                
            # 빈센트 적정 주가 (PER 10배 기준)
            vincent_fair_price = expected_eps * 10.0
            
            # 괴리율 (상승 여력)
            if curr_price and curr_price > 0 and vincent_fair_price > 0:
                divergence = ((vincent_fair_price - curr_price) / curr_price) * 100
            else:
                divergence = -100.0  # 산정 불가
                
            return {
                "종목명": name,
                "티커": ticker,
                "현재가": f"{curr_price:,.0f}원" if curr_price else "정보 없음",
                "예상 EPS(1년뒤)": f"{expected_eps:,.0f}원" if expected_eps else "정보 없음",
                "빈센트 적정가(10배)": f"{vincent_fair_price:,.0f}원" if vincent_fair_price else "정보 없음",
                "상승 여력(괴리율)": divergence,
                "_raw_div": divergence
            }

        # 멀티스레드 병렬 크롤링 (소켓 제한을 위해 max_workers=10 선호)
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            rows_list = [row for _, row in df.iterrows()]
            results = executor.map(fetch_one_ranking, rows_list)
            
        for res in results:
            if res is not None:
                records.append(res)
                
        if not records:
            return pd.DataFrame()
            
        # 데이터프레임으로 변환 후 정렬
        df_res = pd.DataFrame(records)
        df_res = df_res.sort_values(by='_raw_div', ascending=False)
        
        # 가독성 높은 텍스트 변환 및 정리
        df_res['상승 여력(괴리율)'] = df_res['상승 여력(괴리율)'].apply(
            lambda x: f"+{x:.1f}%" if x > 0 else (f"{x:.1f}%" if x != -100.0 else "산정 불가")
        )
        df_res = df_res.drop(columns=['_raw_div'])
        df_res.insert(0, '순위', range(1, len(df_res) + 1))
        
        return df_res
    except Exception as e:
        st.error(f"상승 여력 랭킹 계산 중 오류 발생: {e}")
        return pd.DataFrame()

def render_vincent_valuation_page():
    # 1. CSS 인젝션으로 화면 디자인 개선
    st.markdown("""
        <style>
        .vincent-card {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 25px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        .metric-card {
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 10px;
            padding: 15px;
            text-align: center;
            box-shadow: 0 2px 5px rgba(0,0,0,0.02);
        }
        .metric-title {
            font-size: 0.85rem;
            color: #6c757d;
            margin-bottom: 8px;
            font-weight: 600;
        }
        .metric-value {
            font-size: 1.6rem;
            font-weight: 700;
            color: #2b2b2b;
        }
        .metric-sub {
            font-size: 0.8rem;
            color: #495057;
            margin-top: 5px;
        }
        </style>
    """, unsafe_allow_html=True)

    # 헤더 및 가이드
    st.title("📈 종목 가치 평가 (빈센트의 주가 방정식)")
    st.markdown("""
    이 페이지는 빈센트 애널리스트가 제시한 핵심 주가 방정식을 개별 종목에 투사하여, 
    현재 종목의 주가가 미래 가치 대비 적정한 수준인지 동적으로 평가하고 시뮬레이션합니다.
    """)

    # 빈센트 주가 방정식 카드 설명
    st.markdown("""
        <div class="vincent-card">
            <h3 style="margin-top: 0; color: white;">🎯 빈센트의 주가 결정 방정식</h3>
            <p style="font-size: 1.1rem; font-weight: bold; margin-bottom: 12px; color: #f8f9fa;">
                주가 = PER(밸류에이션 멀티플) × EPS(미래 주당순이익)
            </p>
            <ul style="margin-bottom: 0; font-size: 0.95rem; line-height: 1.6; color: #e9ecef;">
                <li><b>EPS (주당순이익)</b>: 미래의 실제 이익 창출 능력이 핵심이며, 과거의 실적이 아닌 향후 1~2년 뒤의 컨센서스 추정치를 기초로 산정합니다.</li>
                <li><b>PER (밸류에이션)</b>: 시장의 평가 배수(멀티플)로, 코스피 지수의 최근 5년 평균인 <b>약 9.9 ~ 10배</b>를 적정 밸류에이션의 표준 분기점으로 삼습니다.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    # 메인 화면 탭 구성
    tab_indiv, tab_kospi, tab_kosdaq = st.tabs([
        "🔍 개별 종목 가치 진단", 
        "🏆 KOSPI 상승 여력 랭킹", 
        "🏆 KOSDAQ 상승 여력 랭킹"
    ])

    # --- 탭 1: 개별 종목 가치 진단 ---
    with tab_indiv:
        # 2. KRX 종목 로드 및 선택
        df_stocks = get_krx_stock_list()
        if df_stocks.empty:
            st.error("거래소 종목 목록을 불러오지 못했습니다. 잠시 후 다시 시도해 주세요.")
        else:
            # 시장 분리 선택 및 검색 필터링
            col_filter1, col_filter2 = st.columns([1, 2])
            with col_filter1:
                market_type = st.radio("🏢 시장 구분", ["KOSPI", "KOSDAQ"], horizontal=True, key="indiv_market")
            with col_filter2:
                search_query = st.text_input("🔍 종목명 검색 (예: 삼성, 에코프로 - 공백 입력 시 전체 목록)", value="", key="indiv_search")

            # 데이터 필터링
            df_filtered = df_stocks[df_stocks['Market'] == market_type]
            if search_query.strip():
                df_filtered = df_filtered[df_filtered['Name'].str.contains(search_query.strip(), case=False, na=False)]

            if df_filtered.empty:
                st.warning(f"⚠️ {market_type} 시장에서 '{search_query}'(이)가 포함된 종목 검색 결과가 없습니다. 다른 검색어를 입력해 주세요.")
            else:
                # 종목 선택 selectbox 구성
                stock_options = [f"{row['Name']} ({row['Code']})" for _, row in df_filtered.iterrows()]
                
                col_sel1, col_sel2 = st.columns([3, 1])
                with col_sel1:
                    selected_option = st.selectbox("📦 분석할 종목 선택", stock_options, index=0)
                    ticker = re.search(r'\(([0-9]{6})\)', selected_option).group(1)
                    company_name = selected_option.split(" (")[0]
                    
                with col_sel2:
                    st.write("")
                    st.write("")
                    if st.button("🔄 캐시 초기화", use_container_width=True, key="clear_cache_indiv"):
                        st.cache_data.clear()
                        st.rerun()

                # 3. 네이버 금융 크롤러 호출
                with st.spinner(f"{company_name} ({ticker}) 재무 분석 데이터를 가져오는 중..."):
                    stock_data = get_stock_financials_naver(ticker)
                    
                if "error" in stock_data:
                    st.error(stock_data["error"])
                else:
                    curr_price = stock_data["current_price"]
                    cols = stock_data["columns"]
                    financials = stock_data["financial_data"]
                    cols_annual = cols[:4]

                    # 키 매핑 유연화
                    eps_key = next((k for k in financials.keys() if 'EPS' in k), None)
                    per_key = next((k for k in financials.keys() if 'PER' in k), None)

                    e_years = [y for y in cols_annual if '(E)' in y]

                    st.subheader(f"📊 {company_name} ({ticker}) 기업 가치 진단")

                    selected_year = None
                    expected_eps = None
                    expected_per = None
                    avg_past_per = 10.0
                    has_consensus = len(e_years) > 0

                    # 과거 평균 PER 구하기
                    if per_key and per_key in financials:
                        past_pers = []
                        for i, y in enumerate(cols_annual):
                            if '(E)' not in y and i < len(financials[per_key]):
                                val = financials[per_key][i]
                                if val is not None and isinstance(val, (int, float)):
                                    past_pers.append(val)
                        if past_pers:
                            avg_past_per = sum(past_pers) / len(past_pers)

                    # EPS 백업값
                    backup_eps = 1000
                    non_e_years = [y for y in cols_annual if '(E)' not in y]
                    if non_e_years and eps_key:
                        backup_idx = cols.index(non_e_years[-1])
                        if backup_idx < len(financials[eps_key]) and financials[eps_key][backup_idx] is not None:
                            backup_eps = financials[eps_key][backup_idx]

                    if has_consensus:
                        col_opt1, col_opt2 = st.columns(2)
                        with col_opt1:
                            selected_year = st.selectbox(
                                "📅 가치 평가 기준 미래 시점 선택", 
                                e_years, 
                                index=0, 
                                help="네이버 금융에 집계된 향후 시장 컨센서스(E) 연도입니다."
                            )
                            year_idx = cols.index(selected_year)
                            
                            if eps_key and year_idx < len(financials[eps_key]):
                                expected_eps = financials[eps_key][year_idx]
                            if per_key and year_idx < len(financials[per_key]):
                                expected_per = financials[per_key][year_idx]
                                
                        if expected_eps is None:
                            st.warning(f"⚠️ {selected_year}의 예상 EPS(이익) 공시 데이터가 존재하지 않습니다. 최근 확정 실적의 EPS({backup_eps:,.0f}원)를 대체 적용하여 평가합니다.")
                            expected_eps_val = backup_eps
                            eps_str = f"정보 없음 (대체 적용: {backup_eps:,.0f}원)"
                        else:
                            expected_eps_val = expected_eps
                            eps_str = f"{expected_eps:,.0f}원"
                            
                        with col_opt2:
                            st.info(f"💡 **시장의 컨센서스 ({selected_year})**\n* 예상 EPS: **{eps_str}**" + 
                                    (f"\n* 예상 PER: **{expected_per:.1f}배**" if expected_per is not None else "\n* 예상 PER: 정보 없음"))
                    else:
                        st.warning("⚠️ 해당 종목은 애널리스트의 미래 예상 실적(컨센서스) 데이터가 존재하지 않는 소형/비인기 주식입니다. 대신 최근 확정 실적을 기초 자료로 활용하며, 수동 조정을 권장합니다.")
                        if non_e_years:
                            selected_year = non_e_years[-1]
                            year_idx = cols.index(selected_year)
                            if eps_key and year_idx < len(financials[eps_key]):
                                expected_eps = financials[eps_key][year_idx]
                            if per_key and year_idx < len(financials[per_key]):
                                expected_per = financials[per_key][year_idx]
                        
                        expected_eps_val = expected_eps if expected_eps is not None else backup_eps
                        expected_per = expected_per if expected_per is not None else 10.0

                    # 4. 슬라이더 및 수동 조정 패널
                    st.markdown("---")
                    st.markdown("##### **🛠️ 가치 시뮬레이션 변수 커스터마이징**")
                    
                    col_sim1, col_sim2 = st.columns(2)
                    with col_sim1:
                        custom_eps_check = st.checkbox("예상 EPS(이익) 직접 입력/수정하기", value=False)
                        if custom_eps_check:
                            init_eps = int(expected_eps_val) if expected_eps_val is not None else 1000
                            user_eps = st.number_input("✍️ 적용할 예상 EPS (원)", min_value=1, value=init_eps, step=100)
                        else:
                            user_eps = int(expected_eps_val) if expected_eps_val is not None else 1000
                            st.metric("적용 예상 EPS", f"{user_eps:,.0f} 원", help="크롤링된 재무제표의 미래 예상 EPS가 자동 대입되었습니다.")
                            
                    with col_sim2:
                        default_per_val = float(expected_per) if expected_per is not None else (float(avg_past_per) if avg_past_per else 10.0)
                        default_per_val = max(1.0, min(50.0, default_per_val))
                        user_per = st.slider(
                            "⚙️ 적용할 Target PER (밸류에이션 멀티플 배수)", 
                            min_value=1.0, 
                            max_value=50.0, 
                            value=round(default_per_val, 1),
                            step=0.1,
                            help="빈센트 애널리스트의 코스피 기준은 10배입니다. 종목 고유의 특성에 맞춰 조절해 보세요."
                        )

                    # 5. 가치 평가 계산
                    vincent_fair_price = 10.0 * user_eps
                    past_avg_fair_price = avg_past_per * user_eps
                    user_fair_price = user_per * user_eps
                    divergence = ((user_fair_price - curr_price) / curr_price) * 100

                    # 판정
                    if divergence > 20.0:
                        val_status = "🟢 강력 저평가 (매수 기회)"
                        val_color = "#27ae60"
                        val_desc = f"현재 주가가 사용자가 계산한 적정 가치({user_fair_price:,.0f}원) 대비 20% 이상 크게 할인되어 거래되고 있습니다. 주가 회복 시 큰 기대 수익률을 보입니다."
                    elif 5.0 < divergence <= 20.0:
                        val_status = "🟡 적정 저평가 (매수 관심)"
                        val_color = "#f1c40f"
                        val_desc = "현재 주가가 적정 주가보다 다소 낮아 안정 마진(안전 마진)이 어느 정도 확보된 상태입니다."
                    elif -5.0 <= divergence <= 5.0:
                        val_status = "⚪ 적정 가치 (보유)"
                        val_color = "#7f8c8d"
                        val_desc = "현재 주가가 시장이 부여하는 미래 가치를 정확히 반영하고 있는 합리적인 상태입니다."
                    elif -20.0 <= divergence < -5.0:
                        val_status = "🟠 약간 고평가 (비중 조절)"
                        val_color = "#e67e22"
                        val_desc = "현재 주가가 미래 예상 가치를 선반영하여 다소 높게 형성되어 있습니다. 신규 진입에는 주의가 필요합니다."
                    else:
                        val_status = "🔴 과열 고평가 (주의/매도)"
                        val_color = "#c0392b"
                        val_desc = "현재 주가가 미래 이익 및 멀티플 대비 심각하게 과대평가 영역에 속해 있습니다. 단기 조정 또는 밸류에이션 부담이 큽니다."

                    st.markdown("---")
                    st.markdown("##### **📊 적정 가치 평가 요약**")
                    m_col1, m_col2, m_col3 = st.columns(3)
                    with m_col1:
                        st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-title">현재 시장 가격</div>
                                <div class="metric-value">{curr_price:,.0f}원</div>
                                <div class="metric-sub">실시간 기준</div>
                            </div>
                        """, unsafe_allow_html=True)
                    with m_col2:
                        st.markdown(f"""
                            <div class="metric-card" style="border-left: 5px solid {val_color};">
                                <div class="metric-title">사용자 설정 적정 주가</div>
                                <div class="metric-value" style="color: {val_color};">{user_fair_price:,.0f}원</div>
                                <div class="metric-sub">PER {user_per:.1f}배 × EPS {user_eps:,.0f}원</div>
                            </div>
                        """, unsafe_allow_html=True)
                    with m_col3:
                        div_sign = "+" if divergence >= 0 else ""
                        st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-title">목표가 괴리율 (상승 여력)</div>
                                <div class="metric-value" style="color: {val_color};">{div_sign}{divergence:.1f}%</div>
                                <div class="metric-sub">적정 가치 대비 변동 여력</div>
                            </div>
                        """, unsafe_allow_html=True)

                    st.markdown(f"""
                        <div style="background-color: #f8f9fa; border-left: 6px solid {val_color}; padding: 15px; border-radius: 6px; margin-top: 15px; margin-bottom: 25px;">
                            <strong style="font-size: 1.05rem; color: {val_color};">📢 종합 진단 의견: {val_status}</strong><br>
                            <p style="font-size: 0.9rem; color: #2c3e50; margin-top: 8px; margin-bottom: 0; line-height: 1.5;">{val_desc}</p>
                        </div>
                    """, unsafe_allow_html=True)

                    # 차트
                    st.markdown("##### **📊 적정 주가 시나리오별 비교 차트**")
                    labels = [
                        '현재 주가', 
                        '빈센트 기준 주가<br>(PER 10배 적용)', 
                        f'종목 과거 평균 기준 주가<br>(PER {avg_past_per:.1f}배 적용)', 
                        f'사용자 설정 기준 주가<br>(PER {user_per:.1f}배 적용)'
                    ]
                    prices = [curr_price, vincent_fair_price, past_avg_fair_price, user_fair_price]
                    colors = ['#7f8c8d', '#2980b9', '#1abc9c', val_color]

                    fig = go.Figure(data=[go.Bar(
                        x=labels, 
                        y=prices,
                        text=[f"{p:,.0f}원" for p in prices],
                        textposition='auto',
                        marker_color=colors,
                        width=0.5
                    )])
                    fig.update_layout(
                        height=380,
                        margin=dict(l=20, r=20, t=20, b=40),
                        yaxis=dict(title='주가 (원)', tickformat=',d'),
                        xaxis=dict(tickangle=0),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # 상세 테이블
                    with st.expander(f"📋 {company_name} 연간 기업실적분석 데이터 상세 보기", expanded=False):
                        df_fin = pd.DataFrame(financials, index=cols).T
                        df_fin_annual = df_fin[cols_annual]
                        
                        # PyArrow 변환 에러(ValueError) 방지 및 가독성 향상을 위해 데이터 포맷팅
                        def format_val(val, idx_name):
                            if val is None or pd.isna(val) or val == '-':
                                return "-"
                            try:
                                val_float = float(val)
                                if '률' in idx_name or '%' in idx_name or '성향' in idx_name or 'ROE' in idx_name:
                                    return f"{val_float:.2f}%"
                                elif '배' in idx_name or 'PER' in idx_name or 'PBR' in idx_name:
                                    return f"{val_float:.2f}배"
                                else:
                                    return f"{int(val_float):,}"
                            except:
                                return str(val)
                                
                        df_formatted = df_fin_annual.astype(object)
                        for idx in df_formatted.index:
                            df_formatted.loc[idx] = df_formatted.loc[idx].apply(lambda x: format_val(x, idx))
                            
                        st.dataframe(df_formatted, use_container_width=True)
                        st.caption("※ 네이버 금융에서 제공하는 연간 실적 지표를 기반으로 집계된 데이터입니다. (E)는 예상 전망치(컨센서스)를 의미합니다.")

    # --- 탭 2: KOSPI 상승 여력 랭킹 ---
    with tab_kospi:
        st.subheader("🏆 KOSPI 시가총액 상위 상승 여력 랭킹")
        st.markdown("""
        코스피 시가총액 상위 종목의 **1년 뒤 예상 실적(컨센서스)**을 크롤링하여, 
        빈센트 기준 적정 가치(PER 10배 × 미래 EPS) 대비 **상승 여력(괴리율)이 높은 종목부터 내림차순**으로 정렬하여 제공합니다.
        """)
        
        top_n_kospi = st.slider("조회할 시가총액 상위 종목 수", min_value=10, max_value=50, value=30, step=5, key="slider_top_n_kospi")
        
        # 캐싱된 랭킹 데이터 호출
        with st.spinner("KOSPI 종목 가치 분석 및 랭킹 정렬 중 (멀티스레드 가동)..."):
            df_kospi_rank = get_market_valuation_ranking("KOSPI", top_n=top_n_kospi)
            
        if not df_kospi_rank.empty:
            # 스타일링된 데이터프레임 노출
            st.dataframe(df_kospi_rank, use_container_width=True, hide_index=True)
            st.info("💡 **상승 여력(괴리율) 해석**: +%가 높을수록 현재 주가가 빈센트 공식으로 도출된 적정 가치(PER 10배) 대비 극심하게 저평가되어 있음을 뜻합니다.")
        else:
            st.warning("데이터를 불러오지 못했습니다. 새로고침을 시도해 보세요.")

    # --- 탭 3: KOSDAQ 상승 여력 랭킹 ---
    with tab_kosdaq:
        st.subheader("🏆 KOSDAQ 시가총액 상위 상승 여력 랭킹")
        st.markdown("""
        코스닥 시가총액 상위 종목의 **1년 뒤 예상 실적(컨센서스)**을 크롤링하여, 
        빈센트 기준 적정 가치(PER 10배 × 미래 EPS) 대비 **상승 여력(괴리율)이 높은 종목부터 내림차순**으로 정렬하여 제공합니다.
        """)
        
        top_n_kosdaq = st.slider("조회할 시가총액 상위 종목 수", min_value=10, max_value=50, value=30, step=5, key="slider_top_n_kosdaq")
        
        with st.spinner("KOSDAQ 종목 가치 분석 및 랭킹 정렬 중 (멀티스레드 가동)..."):
            df_kosdaq_rank = get_market_valuation_ranking("KOSDAQ", top_n=top_n_kosdaq)
            
        if not df_kosdaq_rank.empty:
            st.dataframe(df_kosdaq_rank, use_container_width=True, hide_index=True)
            st.info("💡 **상승 여력(괴리율) 해석**: +%가 높을수록 현재 주가가 빈센트 공식으로 도출된 적정 가치(PER 10배) 대비 극심하게 저평가되어 있음을 뜻합니다.")
        else:
            st.warning("데이터를 불러오지 못했습니다. 새로고침을 시도해 보세요.")

# 독립 실행 테스트를 위한 엔트리 포인트
if __name__ == "__main__":
    st.set_page_config(layout="wide")
    render_vincent_valuation_page()

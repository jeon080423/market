import pandas as pd
import numpy as np

def calculate_weighted_score(
    mention_counts: dict,
    subscriber_map: dict,
    video_channel_map: dict,
    stock_list
) -> pd.DataFrame:
    """
    가중치 점수 = Σ (해당 종목 언급 횟수 × log10(채널 구독자 수 + 1))
    - 구독자 수에 log 스케일 적용 (대형 채널 독점 방지)
    - stock_list: List of dicts or DataFrame containing ['ticker', 'name']
    - 반환: DataFrame [ticker, stock_name, weighted_score, mention_count, channel_count]
    """
    # Build ticker-to-name mapping
    ticker_to_name = {}
    if isinstance(stock_list, pd.DataFrame):
        for _, row in stock_list.iterrows():
            t = str(row.get("ticker", "")).strip()
            n = str(row.get("name", "")).strip()
            if t:
                ticker_to_name[t] = n
    elif isinstance(stock_list, list):
        for s in stock_list:
            t = str(s.get("ticker", "")).strip()
            n = str(s.get("name", "")).strip()
            if t:
                ticker_to_name[t] = n

    # Aggregation maps
    scores = {}
    total_mentions = {}
    ticker_channels = {}  # {ticker: set(channel_id)}

    # Process each video's mentions
    for video_id, stock_mentions in mention_counts.items():
        channel_id = video_channel_map.get(video_id)
        if not channel_id:
            continue
            
        subscribers = subscriber_map.get(channel_id, 1000)
        
        # log10(subscribers + 1)
        weight = float(np.log10(subscribers + 1))
        
        for ticker, count in stock_mentions.items():
            if count <= 0:
                continue
                
            # Accumulate scores and counts
            scores[ticker] = scores.get(ticker, 0.0) + (count * weight)
            total_mentions[ticker] = total_mentions.get(ticker, 0) + count
            
            # Record unique channels
            if ticker not in ticker_channels:
                ticker_channels[ticker] = set()
            ticker_channels[ticker].add(channel_id)

    # Compile data rows
    rows = []
    for ticker, score in scores.items():
        # Ensure we only include stocks that are in our stock mapping list (filters out raw tickers/unmatched stocks)
        if ticker not in ticker_to_name:
            continue
        name = ticker_to_name[ticker]
        m_count = total_mentions.get(ticker, 0)
        c_count = len(ticker_channels.get(ticker, []))
        
        rows.append({
            "ticker": ticker,
            "stock_name": name,
            "weighted_score": round(score, 2),
            "mention_count": m_count,
            "channel_count": c_count
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["ticker", "stock_name", "weighted_score", "mention_count", "channel_count"])

    # Sort descending by weighted_score
    df = df.sort_values(by="weighted_score", ascending=False).reset_index(drop=True)
    return df

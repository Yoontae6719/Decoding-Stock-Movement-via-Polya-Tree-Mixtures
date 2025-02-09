import pandas as pd
import numpy as np
import yfinance as yf

def moving_average(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()

def exponential_moving_average(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False).mean()

def rate_of_change(series: pd.Series, window: int) -> pd.Series:
    shifted = series.shift(window)
    return (series - shifted) / (shifted + 1e-9)

def rsi(series: pd.Series, window: int) -> pd.Series:
    diff = series.diff(1)
    gain = diff.clip(lower=0)
    loss = diff.clip(upper=0).abs()
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))

def macd_signal(series: pd.Series, short_window: int = 12, long_window: int = 26, signal_window: int = 9):
    ema_short = exponential_moving_average(series, short_window)
    ema_long = exponential_moving_average(series, long_window)
    macd_line = ema_short - ema_long
    signal_line = exponential_moving_average(macd_line, signal_window)
    return macd_line, signal_line

def stochastic_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14):
    highest_high = high.rolling(window).max()
    lowest_low = low.rolling(window).min()
    k = (close - lowest_low) / (highest_high - lowest_low + 1e-9) * 100
    d = k.rolling(3).mean()
    return k, d

def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14):
    highest_high = high.rolling(window).max()
    lowest_low = low.rolling(window).min()
    wr = (highest_high - close) / (highest_high - lowest_low + 1e-9) * -100
    return wr

def bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0):
    m_avg = series.rolling(window).mean()
    m_std = series.rolling(window).std(ddof=0)
    upper_band = m_avg + num_std * m_std
    lower_band = m_avg - num_std * m_std
    return m_avg, upper_band, lower_band

def cci(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20):
    typical_price = (high + low + close) / 3
    ma = typical_price.rolling(window).mean()
    md = (typical_price - ma).rolling(window).apply(lambda x: np.mean(np.abs(x)))
    cci_val = (typical_price - ma) / (md * 0.015 + 1e-9)
    return cci_val

def create_features_and_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values('Date').reset_index(drop=True)

    features_df = df.copy()
    features_df['NextClose'] = features_df['Close'].shift(-1)
    features_df['Y'] = np.where(features_df['NextClose'] > features_df['Close'], 'BUY', 'SELL')
    features_df.dropna(subset=['NextClose'], inplace=True)

    period_list = list(range(2, 31))
    technical_features = []

    for w in period_list:
        col_sma = f'SMA_{w}'
        features_df[col_sma] = moving_average(features_df['Close'], w)
        technical_features.append(col_sma)

        col_ema = f'EMA_{w}'
        features_df[col_ema] = exponential_moving_average(features_df['Close'], w)
        technical_features.append(col_ema)

        col_roc = f'ROC_{w}'
        features_df[col_roc] = rate_of_change(features_df['Close'], w)
        technical_features.append(col_roc)

        col_rsi = f'RSI_{w}'
        features_df[col_rsi] = rsi(features_df['Close'], w)
        technical_features.append(col_rsi)

        col_wr = f'WR_{w}'
        features_df[col_wr] = williams_r(features_df['High'], features_df['Low'], features_df['Close'], w)
        technical_features.append(col_wr)

        col_cci = f'CCI_{w}'
        features_df[col_cci] = cci(features_df['High'], features_df['Low'], features_df['Close'], w)
        technical_features.append(col_cci)

        col_bb_mid = f'BBmid_{w}'
        col_bb_up = f'BBup_{w}'
        col_bb_dn = f'BBdn_{w}'
        bb_mid, bb_up, bb_dn = bollinger_bands(features_df['Close'], w)
        features_df[col_bb_mid] = bb_mid
        features_df[col_bb_up] = bb_up
        features_df[col_bb_dn] = bb_dn
        technical_features += [col_bb_mid, col_bb_up, col_bb_dn]

        col_sto_k = f'StoK_{w}'
        col_sto_d = f'StoD_{w}'
        sto_k, sto_d = stochastic_oscillator(features_df['High'], features_df['Low'], features_df['Close'], w)
        features_df[col_sto_k] = sto_k
        features_df[col_sto_d] = sto_d
        technical_features += [col_sto_k, col_sto_d]

    macd_line, signal_line = macd_signal(features_df['Close'], 12, 26, 9)
    features_df['MACD'] = macd_line
    features_df['MACD_Signal'] = signal_line
    technical_features += ['MACD', 'MACD_Signal']

    for w in period_list:
        col_vol_sma = f'Volume_SMA_{w}'
        features_df[col_vol_sma] = moving_average(features_df['Volume'], w)
        technical_features.append(col_vol_sma)

        col_vol_ema = f'Volume_EMA_{w}'
        features_df[col_vol_ema] = exponential_moving_average(features_df['Volume'], w)
        technical_features.append(col_vol_ema)

    for w in period_list:
        col_range = f'Range_{w}'
        features_df[col_range] = (features_df['High'] - features_df['Low']).rolling(w).mean()
        technical_features.append(col_range)

        col_close_std = f'CloseStd_{w}'
        features_df[col_close_std] = features_df['Close'].rolling(w).std()
        technical_features.append(col_close_std)

    features_df.dropna(subset=technical_features, inplace=True)

    result_df = features_df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Y']].join(
        features_df[technical_features]
    )
    return result_df

if __name__ == "__main__":
    start_date = "2020-01-01"
    end_date = "2025-01-31"

    tickers = [
        'BIIB', 'BA', 'AXP', 'SLB', 'COP', 'AVGO', 'TMO', 'NEE', 'NKE', 'MO',
        'WBA', 'QCOM', 'COST', 'ACN', 'CVS', 'T', 'CVX', 'HD', 'DUK', 'CL',
        'MMM', 'CSCO', 'BAC', 'LOW', 'BLK', 'MDLZ', 'PM', 'UNH', 'VZ', 'CAT',
        'NVDA', 'FDX', 'RTX', 'AIG', 'TMUS', 'INTC', 'PEP', 'TGT', 'GD', 'GS',
        'MDT', 'IBM', 'DIS', 'ORCL', 'COF', 'MSFT', 'KO', 'BKNG', 'V', 'LLY',
        'ADBE', 'AMZN', 'SBUX', 'BMY', 'MRK', 'XOM', 'F', 'JNJ', 'USB', 'AMT',
        'EXC', 'AAPL', 'SPG', 'TXN', 'PFE', 'PG', 'LMT', 'MCD', 'NFLX', 'UNP',
        'HON', 'C', 'GOOG', 'AMGN', 'JPM', 'MA', 'CMCSA', 'ABT', 'SO',
        'GILD', 'MET', 'MS', 'EMR', 'UPS', 'CRM', 'DHR', 'GOOGL', 'GE', 'WFC',
        'WMT', 'ICSA', 'UMCSENT', 'HSN1F', 'UNRATE', 'HYBS'
    ]

    all_results = []

    for ticker in tickers:
        try:
            df = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                group_by='column'   
            )

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = ['_'.join(col).strip() for col in df.columns.values]


            possible_cols = list(df.columns)
            rename_dict = {}
            for col in ['Open','High','Low','Close','Volume']:
                alt_name = col + '_' + ticker
                if alt_name in possible_cols:
                    rename_dict[alt_name] = col

            df.rename(columns=rename_dict, inplace=True)

            required_cols = {'Open','High','Low','Close','Volume'}
            if not required_cols.issubset(df.columns):
                continue

            df.reset_index(inplace=True)
            result_df = create_features_and_labels(df)
            result_df['Stock'] = ticker
            all_results.append(result_df)

        except Exception as e:
            continue

    if all_results:
        final_df = pd.concat(all_results, axis=0).reset_index(drop=True)
    else:
        print("There are No data")

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 기존 타겟 데이터 생성 함수들 (변경 없음) ---
def get_reversal_points(price_series, T=5, P=0.05):
    """
    Bao and Yang (2008) 기준에 따라 추세 전환점을 식별합니다.
    """
    if isinstance(price_series, pd.DataFrame):
        if 'Price' in price_series.columns:
            price_series = price_series['Price']
        else:
            price_series = price_series.iloc[:, 0]

    reversal_points_df = price_series.to_frame(name='Price')
    reversal_points_df['Is_Reversal'] = True

    # 인덱스가 datetime이 아닐 경우 datetime으로 변환 시도
    if not isinstance(reversal_points_df.index, pd.DatetimeIndex):
        try:
            reversal_points_df.index = pd.to_datetime(reversal_points_df.index)
        except Exception as e:
            print(f"경고: 인덱스를 datetime으로 변환할 수 없습니다. 시간 차이 계산에 문제가 있을 수 있습니다: {e}")
            time_diff = T  # T를 만족하는 것으로 간주하거나, 다른 처리 필요

    i = 0
    while i < len(reversal_points_df) - 1:
        if isinstance(reversal_points_df.index, pd.DatetimeIndex):
            time_diff = (reversal_points_df.index[i + 1] - reversal_points_df.index[i]).days
        else:
            time_diff = T

        if time_diff < T:
            price_i = reversal_points_df['Price'].iloc[i]
            price_iplus1 = reversal_points_df['Price'].iloc[i + 1]

            denominator = (price_i + price_iplus1) / 2
            if denominator == 0:
                vibration_percentage = 0
            else:
                vibration_percentage = abs(price_iplus1 - price_i) / denominator

            if vibration_percentage < P:
                reversal_points_df.loc[reversal_points_df.index[i], 'Is_Reversal'] = False
                reversal_points_df.loc[reversal_points_df.index[i + 1], 'Is_Reversal'] = False
                i += 2
                continue
        i += 1

    reversal_points_df['Smoothed_Price'] = np.nan
    reversal_points_df.loc[reversal_points_df['Is_Reversal'], 'Smoothed_Price'] = reversal_points_df['Price']

    if reversal_points_df['Smoothed_Price'].isnull().all():
        reversal_points_df['Smoothed_Price'] = reversal_points_df['Price']
    else:
        # 변경: bfill()은 이제 interpolate()가 처리할 NaN을 남겨두지 않으므로, 이 부분을 나중에 처리
        # reversal_points_df['Smoothed_Price'] = reversal_points_df['Smoothed_Price'].bfill() # 이 부분 제거

        # 일단 선형 보간을 먼저 시도
        reversal_points_df['Smoothed_Price'] = reversal_points_df['Smoothed_Price'].interpolate(method='linear')
        # 보간 후에도 남아있는 시작점의 NaN은 첫 유효값으로 채움
        reversal_points_df['Smoothed_Price'] = reversal_points_df['Smoothed_Price'].bfill()
        # 보간 후에도 남아있는 끝점의 NaN은 마지막 유효값으로 채움
        reversal_points_df['Smoothed_Price'] = reversal_points_df['Smoothed_Price'].ffill()

    return reversal_points_df


def assign_alternating_buy_sell_signals(reversal_points_df):
    """
    `get_reversal_points`에서 식별된 추세 전환점을 기반으로,
    'buy'와 'sell' 신호가 번갈아 가며 나타나도록 생성합니다.
    """
    signals = pd.Series(None, index=reversal_points_df.index, dtype=object)

    reversal_points_indices = reversal_points_df[reversal_points_df['Is_Reversal']].index.tolist()
    reversal_points_prices = reversal_points_df.loc[reversal_points_indices, 'Price'].tolist()

    if len(reversal_points_indices) < 2:
        reversal_points_df['Signal'] = signals
        return reversal_points_df

    last_signal_was = None
    # 첫 전환점부터 신호를 할당합니다.
    if reversal_points_prices[1] > reversal_points_prices[0]:
        signals.loc[reversal_points_indices[0]] = 'buy'
        last_signal_was = 'buy'
    elif reversal_points_prices[1] < reversal_points_prices[0]:
        signals.loc[reversal_points_indices[0]] = 'sell'
        last_signal_was = 'sell'

    for i in range(len(reversal_points_indices) - 1):
        current_idx = reversal_points_indices[i]
        next_idx = reversal_points_indices[i + 1]

        current_price = reversal_points_prices[i]
        next_price = reversal_points_prices[i + 1]

        if next_price > current_price:  # 상승 추세 (저점 -> 고점)
            if last_signal_was != 'buy':
                signals.loc[current_idx] = 'buy'
                last_signal_was = 'buy'
        elif next_price < current_price:  # 하락 추세 (고점 -> 저점)
            if last_signal_was != 'sell':
                signals.loc[current_idx] = 'sell'
                last_signal_was = 'sell'

    reversal_points_df['Signal'] = signals
    return reversal_points_df


def transform_and_interpolate_signals(signals_df):
    """
    'Signal' 컬럼을 숫자로 변환하고, 유효 신호들 사이의 NaN만 선형 보간합니다.
    'buy'는 1, 'sell'은 -1, 그 외 'NaN'으로 변환합니다.
    첫 유효 신호 이전과 마지막 유효 신호 이후는 '0.0'으로 채웁니다.
    """
    numerical_signals = pd.Series(np.nan, index=signals_df.index, dtype=float)

    # 'buy'는 1.0, 'sell'은 -1.0으로 변환하고 나머지는 NaN으로 유지
    numerical_signals.loc[signals_df['Signal'] == 'buy'] = 1.0
    numerical_signals.loc[signals_df['Signal'] == 'sell'] = -1.0

    if numerical_signals.empty:
        signals_df['Numerical_Signal'] = pd.Series(0.0, index=signals_df.index)
        return signals_df

    # 첫 유효 신호 인덱스 찾기
    first_valid_idx = numerical_signals.first_valid_index()
    # 마지막 유효 신호 인덱스 찾기
    last_valid_idx = numerical_signals.last_valid_index()

    # 1. 첫 유효 신호 이전의 NaN 값을 0으로 채움
    if first_valid_idx is not None:
        numerical_signals.loc[:first_valid_idx] = numerical_signals.loc[:first_valid_idx].fillna(0.0)
    else:  # 유효 신호가 전혀 없는 경우
        signals_df['Numerical_Signal'] = pd.Series(0.0, index=signals_df.index)
        return signals_df

    # 2. 유효 신호들 사이의 NaN을 선형 보간
    # 이 부분이 실수값을 생성하는 핵심입니다.
    numerical_signals = numerical_signals.interpolate(method='linear')

    # 3. 마지막 유효 신호 이후의 NaN 값을 0으로 채움
    if last_valid_idx is not None:
        numerical_signals.loc[last_valid_idx:] = numerical_signals.loc[last_valid_idx:].fillna(0.0)

    signals_df['Numerical_Signal'] = numerical_signals
    return signals_df


# --- 메인 실행 로직 (변경 없음) ---
def process_and_save_target_data(ticker_list, base_directory='.'):
    """
    주어진 종목 리스트에 대해 타겟 데이터를 생성하고 각 종목 디렉토리에 저장합니다.
    """
    T = 5
    P = 0.05

    start_date = '2015-01-01'
    end_date = '2023-01-01'

    for ticker in ticker_list:
        print(f"\n--- {ticker} 종목 타겟 데이터 생성 시작 ---")
        ticker_directory = os.path.join(base_directory, ticker)
        if not os.path.exists(ticker_directory):
            os.makedirs(ticker_directory)
            print(f"디렉토리 생성: {ticker_directory}")
        else:
            print(f"디렉토리 확인: {ticker_directory}")

        try:
            df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
            if df.empty or 'Close' not in df.columns:
                print(f"경고: {ticker} 데이터를 다운로드할 수 없거나 'Close' 컬럼이 없습니다. 건너뜀.")
                continue

            price_series = df['Close'].dropna().squeeze()
            if price_series.empty:
                print(f"경고: {ticker} 'Close' 컬럼에 유효한 데이터가 없습니다. 건너뛴.")
                continue

            reversal_df = get_reversal_points(price_series, T, P)
            signals_df = assign_alternating_buy_sell_signals(reversal_df)
            final_data_df = transform_and_interpolate_signals(signals_df)

            output_filename = os.path.join(ticker_directory, f'{ticker}_target_data.csv')
            final_data_df.to_csv(output_filename, index=True)

            print(f"'{ticker}' 종목의 타겟 데이터가 '{output_filename}' 파일로 성공적으로 저장되었습니다.")

            # 시각화 추가 (오류 확인 및 디버깅용으로 잠시 주석 해제하여 사용)
            # plt.figure(figsize=(18, 10))
            # plt.subplot(2, 1, 1)
            # plt.plot(price_series.index, price_series.values, label='원본 종가', color='gray', alpha=0.7)
            # buys_plot = final_data_df[final_data_df['Signal'] == 'buy']
            # sells_plot = final_data_df[final_data_df['Signal'] == 'sell']
            # plt.scatter(buys_plot.index, buys_plot['Price'], marker='^', color='green', s=150, label='매수 신호', zorder=5)
            # plt.scatter(sells_plot.index, sells_plot['Price'], marker='v', color='red', s=150, label='매도 신호', zorder=5)
            # plt.title(f"{ticker} 주가 및 이산적인 매수/매도 신호")
            # plt.legend()
            # plt.grid(True)
            #
            # plt.subplot(2, 1, 2)
            # plt.plot(final_data_df.index, final_data_df['Numerical_Signal'], label='선형 보간된 숫자 신호 (0=홀드)', color='purple')
            # plt.axhline(1, color='green', linestyle=':', linewidth=1, alpha=0.7, label='Buy (1)')
            # plt.axhline(-1, color='red', linestyle=':', linewidth=1, alpha=0.7, label='Sell (-1)')
            # plt.axhline(0, color='blue', linestyle='--', linewidth=1, alpha=0.7, label='Hold (0)')
            # plt.title(f"{ticker} 선형 보간된 매수/매도 타겟 데이터 (시작 및 끝 0=홀드)")
            # plt.legend()
            # plt.grid(True)
            # plt.ylim(-1.1, 1.1)
            # plt.tight_layout()
            # plt.show()


        except Exception as e:
            print(f"오류 발생: {ticker} 종목 처리 중 예외 발생 - {e}")
            continue

    print("\n--- 모든 종목의 타겟 데이터 생성이 완료되었습니다. ---")


if __name__ == "__main__":
    target_tickers = ['AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'BRK-B', 'TSLA', 'AVGO', 'GOOG']
    base_data_directory = '.'

    if not os.path.exists(base_data_directory):
        print(f"경고: 지정된 기본 데이터 디렉토리 '{base_data_directory}'가 존재하지 않습니다. 스크립트 실행 경로를 다시 확인해주세요.")

    process_and_save_target_data(target_tickers, base_directory=base_data_directory)
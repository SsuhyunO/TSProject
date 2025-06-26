import pandas as pd
import numpy as np
import os
from pyESN import ESN
import matplotlib.pyplot as plt
import warnings
import yfinance as yf
import vectorbt as vbt
import inspect # 디버깅을 위해 inspect 모듈 임포트

warnings.filterwarnings('ignore')

# --- 디버깅: 현재 Vectorbt 버전 확인 ---
try:
    print(f"현재 Vectorbt 버전: {vbt.__version__}")
except AttributeError:
    print("Vectorbt 버전 정보를 찾을 수 없습니다.")
# --- 디버깅 끝 ---

# --- 0. 고정된 최적 파라미터 설정 ---
FIXED_N_RESERVOIR = 782
FIXED_SPECTRAL_RADIUS = 0.75
FIXED_THRESHOLD_BUY = 0.5
FIXED_THRESHOLD_SELL = -0.5

# --- 1. 파일 경로 설정 및 데이터 로드 ---
base_data_directory = r'C:\Users\suhyu\PycharmProjects\CDProject\TSProject'
ticker = 'AVGO'

gdc_file = os.path.join(base_data_directory, ticker, f'{ticker}_optimized_gdc_signals.csv')
roc_file = os.path.join(base_data_directory, ticker, f'{ticker}_optimized_roc_signals.csv')
rsi_file = os.path.join(base_data_directory, ticker, f'{ticker}_optimized_rsi_signals.csv')
target_file = os.path.join(base_data_directory, ticker, f'{ticker}_target_data.csv')

try:
    df_gdc = pd.read_csv(gdc_file, index_col='Date', parse_dates=True)
    df_roc = pd.read_csv(roc_file, index_col='Date', parse_dates=True)
    df_rsi = pd.read_csv(rsi_file, index_col='Date', parse_dates=True)
    df_target = pd.read_csv(target_file, index_col='Date', parse_dates=True)

except FileNotFoundError as e:
    print(f"Error: Required file not found - {e.filename}")
    print("Please ensure 'NVDA' directory exists and contains all necessary CSV files.")
    exit()

df_gdc = df_gdc.rename(columns={'Numerical_Signal': 'GDC_Signal'})
df_roc = df_roc.rename(columns={'Numerical_Signal': 'ROC_Signal'})
df_rsi = df_rsi.rename(columns={'Numerical_Signal': 'RSI_Signal'})
df_target = df_target.rename(columns={'Numerical_Signal': 'Target_Signal'})

merged_df_features = pd.concat([df_gdc['GDC_Signal'], df_roc['ROC_Signal'], df_rsi['RSI_Signal']], axis=1)
merged_df_features.fillna(0, inplace=True)

df_target.fillna(0, inplace=True)

print(
    f"Initial merged_df_features length: {len(merged_df_features)}, start: {merged_df_features.index.min()}, end: {merged_df_features.index.max()}")
print(f"Initial df_target length: {len(df_target)}, start: {df_target.index.min()}, end: {df_target.index.max()}")

overall_start_date = min(df_gdc.index.min(), df_roc.index.min(), df_rsi.index.min(), df_target.index.min())
overall_end_date = max(df_gdc.index.max(), df_roc.index.max(), df_rsi.index.max(), df_target.index.max())

try:
    current_today = pd.to_datetime('today')
    if overall_end_date > current_today:
        print(f"Warning: Data end date {overall_end_date.strftime('%Y-%m-%d')} is in the future. Adjusting to today's date.")
        overall_end_date = current_today

    stock_data = yf.download(ticker, start=overall_start_date, end=overall_end_date + pd.Timedelta(days=1), progress=False)

    print(
        f"Downloaded stock_data length: {len(stock_data)}, start: {stock_data.index.min()}, end: {stock_data.index.max()}")

    if stock_data.empty:
        raise ValueError("No stock price data downloaded from yfinance. Check dates or ticker.")

    actual_prices_series = stock_data['Close']

except Exception as e:
    print(f"Error downloading stock data or aligning: {e}")
    print("Please check ticker symbol, internet connection, and data availability.")
    exit()

common_trading_days = merged_df_features.index.intersection(df_target.index).intersection(actual_prices_series.index)
print(f"Common trading days count (before final reindex): {len(common_trading_days)}")

if len(common_trading_days) < 100:
    print(
        f"Warning: Too few common trading days ({len(common_trading_days)}). Please check your data sources and periods.")
    exit()

merged_df = merged_df_features.reindex(common_trading_days)
df_target = df_target.reindex(common_trading_days)
actual_prices_series = actual_prices_series.reindex(common_trading_days)

X_full = merged_df.values
y_full = df_target['Target_Signal'].values.astype(float)

# 날짜 인덱스를 NumPy 배열과 동기화하기 위해 저장
full_index = merged_df.index

print(f"\nFinal aligned X_full length: {len(X_full)}")
print(f"Final aligned y_full length: {len(y_full)}")
print(f"Final merged_df (index) start: {merged_df.index.min()}, end: {merged_df.index.max()}")

print("\n--- ESN 모델 전체 데이터를 12개 부분으로 나누어 학습 및 테스트 시작 ---")

num_segments = 12
data_length = len(X_full)
segment_size = data_length // num_segments

if segment_size < 2:
    print(
        f"Error: Total data length ({data_length}) is too short to divide into {num_segments} segments, or segments are too small. Decrease num_segments or provide more data.")
    exit()

esn_segment_results = []

for test_segment_idx_0based in range(2, num_segments): # 0, 1 세그먼트는 훈련 데이터가 부족하여 건너뜁니다.
    test_segment_num_1based = test_segment_idx_0based + 1

    train_end_idx_exclusive = test_segment_idx_0based * segment_size

    current_train_X = X_full[0: train_end_idx_exclusive]
    current_train_y = y_full[0: train_end_idx_exclusive]
    current_train_index = full_index[0: train_end_idx_exclusive]

    test_start_idx_inclusive = train_end_idx_exclusive
    test_end_idx_exclusive = min(data_length, (test_segment_idx_0based + 1) * segment_size)

    if test_segment_idx_0based == num_segments - 1:
        test_end_idx_exclusive = data_length

    current_test_X = X_full[test_start_idx_inclusive: test_end_idx_exclusive]
    current_test_prices = actual_prices_series.iloc[test_start_idx_inclusive: test_end_idx_exclusive]
    current_test_index = full_index[test_start_idx_inclusive: test_end_idx_exclusive]

    print(f"\n--- Processing Test Segment {test_segment_num_1based}/{num_segments} ---")
    print(f"Train data length: {len(current_train_X)} (Segments 1 to {test_segment_num_1based - 1})")
    print(f"Test data length: {len(current_test_X)} (Segment {test_segment_num_1based})")
    print(
        f"Test Date Range: {current_test_index.min().strftime('%Y-%m-%d')} to {current_test_index.max().strftime('%Y-%m-%d')}")

    if len(current_train_X) == 0:
        print(f"Skipping segment {test_segment_num_1based} due to insufficient training data.")
        continue
    if len(current_test_X) < 2:
        print(
            f"Skipping segment {test_segment_num_1based} due to insufficient test data length ({len(current_test_X)} days).")
        continue

    segment_esn = ESN(
        n_inputs=current_train_X.shape[1],
        n_outputs=current_train_y.shape[1] if current_train_y.ndim > 1 else 1,
        n_reservoir=FIXED_N_RESERVOIR,
        spectral_radius=FIXED_SPECTRAL_RADIUS,
        sparsity=0.5,
        random_state=42,
        silent=True
    )
    segment_esn.fit(current_train_X, current_train_y)

    esn_pred_raw_segment = segment_esn.predict(current_test_X)
    esn_classified_signals_segment = np.zeros_like(esn_pred_raw_segment, dtype=int)
    esn_classified_signals_segment[esn_pred_raw_segment > FIXED_THRESHOLD_BUY] = 1
    esn_classified_signals_segment[esn_pred_raw_segment < FIXED_THRESHOLD_SELL] = -1
    esn_signals_series_segment = pd.Series(esn_classified_signals_segment.flatten(), index=current_test_index)

    # --- 백테스팅 (Vectorbt 0.20.0+ 버전용, from_signals 사용) ---
    try:
        # ESN 신호 (1: 롱 진입, -1: 숏 진입, 0: 포지션 없음/청산)을 from_signals에 맞게 매핑
        esn_portfolio_segment = vbt.Portfolio.from_signals(
            current_test_prices,
            entries=esn_signals_series_segment == 1,       # 신호가 1일 때 롱 포지션 진입
            exits=esn_signals_series_segment == 0,         # 신호가 0일 때 모든 포지션 청산
            short_entries=esn_signals_series_segment == -1, # 신호가 -1일 때 숏 포지션 진입
            short_exits=esn_signals_series_segment == 0,   # 신호가 0일 때 모든 숏 포지션 청산
            init_cash=10000,
            fees=0.001,
            slippage=0.0001,
            freq='1D',
        )
    except Exception as e:
        print(f"디버깅 오류: Portfolio.from_signals 호출 중 오류 발생: {e}")
        print("Vectorbt 환경에 심각한 문제가 있을 수 있습니다. PyCharm 재시작 및 Vectorbt 완전 재설치 시도 권장.")
        continue


    esn_stats_raw_segment = esn_portfolio_segment.stats()
    esn_stats_segment = esn_stats_raw_segment.to_series() if not isinstance(esn_stats_raw_segment,
                                                                            pd.Series) else esn_stats_raw_segment

    esn_segment_results.append({
        'Test_Segment': test_segment_num_1based,
        'Train_Period_End': current_train_index.max().strftime('%Y-%m-%d') if len(current_train_index) > 0 else 'N/A',
        'Test_Period_Start': current_test_index.min().strftime('%Y-%m-%d'),
        'Test_Period_End': current_test_index.max().strftime('%Y-%m-%d'),
        'ESN_Return_[%]': esn_stats_segment['Total Return [%]'],
        'ESN_MDD_[%]': esn_stats_segment['Max Drawdown [%]'],
        'ESN_Sharpe': esn_stats_segment['Sharpe Ratio']
    })

# --- 모든 세그먼트별 결과 출력 ---
print("\n--- ESN Strategy Performance Per Test Segment ---")
esn_results_df = pd.DataFrame(esn_segment_results).set_index('Test_Segment')

# 명시적으로 '누적 수익률'이라는 컬럼명을 추가
esn_results_df.rename(columns={'ESN_Return_[%]': '누적 수익률 (%)'}, inplace=True)
print(esn_results_df.round(2))

print("\n스크립트 실행 완료.")
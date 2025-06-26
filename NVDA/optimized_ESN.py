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
FIXED_N_RESERVOIR = 512
FIXED_SPECTRAL_RADIUS = 0.87
FIXED_THRESHOLD_BUY = 0.5
FIXED_THRESHOLD_SELL = -0.5

# --- 1. 파일 경로 설정 및 데이터 로드 ---
base_data_directory = r'C:\Users\suhyu\PycharmProjects\CDProject\TSProject'
ticker = 'NVDA'

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

# --- ESN 모델 학습 및 백테스팅 (전체 데이터를 12개 부분으로 나누어 수행) ---

print("\n--- ESN 모델 전체 데이터를 12개 부분으로 나누어 학습 및 테스트 시작 ---")

num_segments = 12
data_length = len(X_full)
segment_size = data_length // num_segments

if segment_size < 2:
    print(
        f"Error: Total data length ({data_length}) is too short to divide into {num_segments} segments, or segments are too small. Decrease num_segments or provide more data.")
    exit()

esn_segment_results = []

for test_segment_idx_0based in range(2, num_segments):
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

    # --- 디버깅: vbt.Portfolio.from_holding의 시그니처 확인 ---
    try:
        print(f"\n--- 디버깅: from_holding 시그니처 ---")
        # 여기서 'cls'는 메서드의 첫 인자로 자동 전달되므로, 실제 사용 시에는 명시하지 않음
        print(f"vbt.Portfolio.from_holding 시그니처: {inspect.signature(vbt.Portfolio.from_holding)}")
        print(f"------------------------------------")
    except Exception as e:
        print(f"디버깅 경고: vbt.Portfolio.from_holding 시그니처를 가져올 수 없습니다. 오류: {e}")
    # --- 디버깅 끝 ---

    # --- 백테스팅 (Vectorbt 0.20.0+ 버전용) ---
    # `TypeError: Portfolio.from_holding() takes 2 positional arguments but 3 were given` 오류가
    # 0.27.3 버전에서 발생한다면, 이는 매우 특이한 환경 문제입니다.
    # 아래 코드는 0.20.0+ 버전의 표준 API를 따릅니다.
    # 만약 이 코드로도 에러가 발생한다면, PyCharm 환경 또는 Python 환경 자체를 재설정해보세요.
    try:
        esn_portfolio_segment = vbt.Portfolio.from_holding(
            current_test_prices,
            esn_signals_series_segment, # this is the 'holding' argument (2nd positional)
            initial_cash=10000,
            fees=0.001,
            slippage=0.0001,
            freq='1D',
            call_gc=False
        )
        print("디버깅: Portfolio.from_holding이 모든 인자를 사용하여 성공적으로 호출되었습니다.")
    except TypeError as e:
        print(f"디버깅 오류: Portfolio.from_holding 호출 중 TypeError 발생: {e}")
        print("현재 Vectorbt 버전은 최신이지만, from_holding이 여전히 추가 인자를 받지 않는 것 같습니다.")
        print("이는 환경 문제일 가능성이 높습니다. PyCharm 재시작, 가상환경 재활성화 또는 Vectorbt 재설치 시도 권장.")
        # Fallback: 2개 인자만 사용하여 호출 시도 (이전 버전 호환성)
        try:
            esn_portfolio_segment = vbt.Portfolio.from_holding(
                current_test_prices,
                esn_signals_series_segment
            )
            print("디버깅: 2개의 위치 인자만 사용하여 Portfolio.from_holding이 성공적으로 호출되었습니다.")
            # 이 경우, 초기 현금, 수수료, 슬리피지는 기본값(보통 0)으로 설정됩니다.
            # 이 버전에서는 이 속성들을 포트폴리오 생성 후 직접 설정하는 것이 불가능할 수 있습니다.
            try:
                esn_portfolio_segment.initial_cash = 10000
                esn_portfolio_segment.fees = 0.001
                esn_portfolio_segment.slippage = 0.0001
                print("디버깅: Portfolio 객체에 initial_cash, fees, slippage 속성 설정 시도됨.")
            except AttributeError:
                print("디버깅 경고: Portfolio 객체에 initial_cash, fees, slippage 속성을 직접 설정할 수 없습니다. 기본값이 사용됩니다.")
        except Exception as inner_e:
            print(f"디버깅 치명적 오류: 2개의 위치 인자를 사용한 Portfolio.from_holding 호출도 실패했습니다. 오류: {inner_e}")
            print("Vectorbt 설치가 손상되었을 수 있습니다. Vectorbt를 완전히 제거하고 다시 설치하세요.")
            continue # 다음 세그먼트로 이동하거나 스크립트 중단 고려


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

print(esn_results_df.round(2))

print("\n스크립트 실행 완료.")
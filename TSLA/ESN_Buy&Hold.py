import pandas as pd
import numpy as np
import os
from pyESN import ESN
import matplotlib.pyplot as plt
import warnings
import yfinance as yf
import vectorbt as vbt # Vectorbt 라이브러리 임포트

warnings.filterwarnings('ignore')

# --- 0. 고정된 최적 파라미터 설정 ---
FIXED_N_RESERVOIR = 510
FIXED_SPECTRAL_RADIUS = 0.89
FIXED_THRESHOLD_BUY = 0.5
FIXED_THRESHOLD_SELL = -0.5

# --- 1. 파일 경로 설정 및 데이터 로드 ---
base_data_directory = r'C:\Users\suhyu\PycharmProjects\CDProject\TSProject'
ticker = 'TSLA'

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
    print(f"오류: 필수 파일이 없습니다 - {e.filename}")
    print(" 'NVDA' 디렉토리가 존재하고 필요한 모든 CSV 파일이 포함되어 있는지 확인하세요.")
    exit()

df_gdc = df_gdc.rename(columns={'Numerical_Signal': 'GDC_Signal'})
df_roc = df_roc.rename(columns={'Numerical_Signal': 'ROC_Signal'})
df_rsi = df_rsi.rename(columns={'Numerical_Signal': 'RSI_Signal'})
df_target = df_target.rename(columns={'Numerical_Signal': 'Target_Signal'})

merged_df_features = pd.concat([df_gdc['GDC_Signal'], df_roc['ROC_Signal'], df_rsi['RSI_Signal']], axis=1)
merged_df_features.fillna(0, inplace=True)

df_target.fillna(0, inplace=True)


print(f"초기 merged_df_features 길이: {len(merged_df_features)}, 시작: {merged_df_features.index.min()}, 끝: {merged_df_features.index.max()}")
print(f"초기 df_target 길이: {len(df_target)}, 시작: {df_target.index.min()}, 끝: {df_target.index.max()}")


overall_start_date = min(df_gdc.index.min(), df_roc.index.min(), df_rsi.index.min(), df_target.index.min())
overall_end_date = max(df_gdc.index.max(), df_roc.index.max(), df_rsi.index.max(), df_target.index.max())

try:
    # yfinance는 end 날짜의 다음 날까지 가져와야 해당 end 날짜의 데이터를 포함합니다.
    stock_data = yf.download(ticker, start=overall_start_date, end=overall_end_date + pd.Timedelta(days=1), progress=False)
    print(f"다운로드된 stock_data 길이: {len(stock_data)}, 시작: {stock_data.index.min()}, 끝: {stock_data.index.max()}")

    if stock_data.empty:
        raise ValueError("yfinance에서 주가 데이터가 다운로드되지 않았습니다. 날짜 또는 티커를 확인하세요.")

    actual_prices_series = stock_data['Close']

except Exception as e:
    print(f"주가 데이터 다운로드 또는 정렬 오류: {e}")
    print("티커 심볼, 인터넷 연결 및 데이터 가용성을 확인하세요.")
    exit()

common_trading_days = merged_df_features.index.intersection(df_target.index).intersection(actual_prices_series.index)
print(f"공통 거래일 수 (최종 재인덱싱 전): {len(common_trading_days)}")

if len(common_trading_days) < 100:
    print(f"경고: 공통 거래일이 너무 적습니다 ({len(common_trading_days)}). 데이터 소스 및 기간을 확인하세요.")
    exit() # 데이터가 너무 적으면 중단

merged_df = merged_df_features.reindex(common_trading_days)
df_target = df_target.reindex(common_trading_days)
actual_prices_series = actual_prices_series.reindex(common_trading_days)

X_full = merged_df.values
y_full = df_target['Target_Signal'].values.astype(float)
actual_prices_1d_array = actual_prices_series.values.astype(float).flatten() # 이 부분은 사용하지 않더라도 혹시 몰라 남겨둡니다.

print(f"\n최종 정렬된 X_full 길이: {len(X_full)}")
print(f"최종 정렬된 y_full 길이: {len(y_full)}")
print(f"최종 정렬된 actual_prices 길이: {len(actual_prices_1d_array)}")
print(f"최종 merged_df (인덱스) 시작: {merged_df.index.min()}, 끝: {merged_df.index.max()}")


# --- ESN 모델 학습 및 신호 생성 ---

print("\n--- ESN 모델 최종 학습 및 백테스팅 신호 생성 ---")

# 백테스팅 기간 설정 (예: 전체 데이터의 마지막 20%)
backtest_fraction = 0.2
backtest_start_idx = int(len(merged_df) * (1 - backtest_fraction))
backtest_end_idx = len(merged_df)

# 최종 훈련 데이터 (백테스팅 시작 직전까지의 모든 데이터)
final_train_X = X_full[:backtest_start_idx]
final_train_y = y_full[:backtest_start_idx]

# 백테스팅 기간의 입력 데이터 및 실제 주가 데이터
backtest_X = X_full[backtest_start_idx:backtest_end_idx]
backtest_prices = actual_prices_series.iloc[backtest_start_idx:backtest_end_idx] # Vectorbt를 위해 Series 유지
backtest_index = merged_df.index[backtest_start_idx:backtest_end_idx]

if len(final_train_X) == 0 or len(backtest_X) == 0 or len(backtest_prices) < 2:
    print("오류: 최종 훈련 또는 백테스팅을 위한 데이터가 부족합니다. 'backtest_fraction' 또는 데이터 범위를 조정하세요.")
    exit()

print(f"최종 훈련 데이터 길이: {len(final_train_X)}")
print(f"백테스팅 데이터 길이: {len(backtest_X)}")
print(f"백테스팅 날짜 범위: {backtest_index.min().strftime('%Y-%m-%d')} ~ {backtest_index.max().strftime('%Y-%m-%d')}")

# ESN 모델 학습
final_esn = ESN(
    n_inputs=final_train_X.shape[1],
    n_outputs=final_train_y.shape[1] if final_train_y.ndim > 1 else 1,
    n_reservoir=FIXED_N_RESERVOIR,
    spectral_radius=FIXED_SPECTRAL_RADIUS,
    sparsity=0.5,
    random_state=42,
    silent=True
)
final_esn.fit(final_train_X, final_train_y)

# 백테스팅 기간 동안 신호 예측
esn_pred_raw = final_esn.predict(backtest_X)

esn_classified_signals = np.zeros_like(esn_pred_raw, dtype=int)
esn_classified_signals[esn_pred_raw > FIXED_THRESHOLD_BUY] = 1
esn_classified_signals[esn_pred_raw < FIXED_THRESHOLD_SELL] = -1

# 예측된 신호를 Pandas Series로 변환 (Vectorbt 입력 형식에 맞춤)
esn_signals_series = pd.Series(esn_classified_signals.flatten(), index=backtest_index)


# --- Vectorbt를 사용한 ESN 전략 백테스팅 (from_orders 활용) ---
print("\n--- Vectorbt를 사용한 ESN 전략 백테스팅 시작 ---")

# ESN 신호 (1: 롱 포지션, -1: 숏 포지션, 0: 포지션 없음/현금)을 목표 비중으로 매핑
# 1 -> 1.0 (100% 롱), -1 -> -1.0 (100% 숏), 0 -> 0.0 (0% 포지션)
esn_target_positions = esn_signals_series.astype(float)

# 가격 데이터와 신호 데이터의 인덱스를 완벽하게 일치시키고 결측값 처리
price_data_for_vbt = backtest_prices.sort_index().drop_duplicates()
esn_target_positions = esn_target_positions.reindex(price_data_for_vbt.index).fillna(method='ffill').fillna(0) # 결측값은 이전 값으로 채우고, 앞쪽에 남은 결측은 0으로 채움


esn_portfolio = vbt.Portfolio.from_orders(
    price_data_for_vbt,
    size=esn_target_positions,
    size_type='target_percent', # 'size' 인자를 포트폴리오의 목표 비중(%)으로 해석
    init_cash=10000,
    fees=0.001, # 거래 수수료 0.1% 가정
    slippage=0.0001, # 슬리피지 0.01% 가정
    freq='1D', # 일별 데이터
)

esn_stats_raw = esn_portfolio.stats()
# esn_stats_raw를 Pandas Series로 변환
esn_stats = esn_stats_raw.to_series() if not isinstance(esn_stats_raw, pd.Series) else esn_stats_raw


print("\nESN 전략 성과:")
# .at 접근자 사용 (단일 스칼라 값 접근에 유용)
print(esn_stats.loc[['Total Return [%]', 'Max Drawdown [%]', 'Sharpe Ratio']]) # list of keys uses .loc

# --- Buy & Hold 전략 백테스팅 (from_holding 활용) ---
print("\n--- Buy & Hold 전략 백테스팅 시작 ---")

buy_hold_portfolio = vbt.Portfolio.from_holding(
    price_data_for_vbt, # ESN과 동일한 가격 데이터 사용
    init_cash=10000,
    fees=0.001, # 초기 매수 시 수수료 적용
    freq='1D',
)

buy_hold_stats_raw = buy_hold_portfolio.stats()
# buy_hold_stats_raw를 Pandas Series로 변환
buy_hold_stats = buy_hold_stats_raw.to_series() if not isinstance(buy_hold_stats_raw, pd.Series) else buy_hold_stats_raw

print("\nBuy & Hold 전략 성과:")
# .at 접근자 사용 (단일 스칼라 값 접근에 유용)
print(buy_hold_stats.loc[['Total Return [%]', 'Max Drawdown [%]', 'Sharpe Ratio']])


# --- 결과 비교 및 시각화 ---
print("\n--- 전략 성능 최종 비교 ---")

comparison_df = pd.DataFrame({
    'ESN 전략': [
        f"{esn_stats.at['Total Return [%]']:.2f}%", # .at 접근자 사용
        f"{esn_stats.at['Max Drawdown [%]']:.2f}%", # .at 접근자 사용
        f"{esn_stats.at['Sharpe Ratio']:.2f}"
    ],
    'Buy & Hold 전략': [
        f"{buy_hold_stats.at['Total Return [%]']:.2f}%", # .at 접근자 사용
        f"{buy_hold_stats.at['Max Drawdown [%]']:.2f}%", # .at 접근자 사용
        f"{buy_hold_stats.at['Sharpe Ratio']:.2f}"
    ]
}, index=['총 수익률', '최대 낙폭', '샤프 비율'])

print(comparison_df)

# 포트폴리오 가치 변화 시각화
plt.figure(figsize=(18, 9))

esn_value = esn_portfolio.asset_value()
buy_hold_value = buy_hold_portfolio.asset_value()

plt.plot(esn_value.index, esn_value, label='ESN 전략', color='blue', linewidth=2)
plt.plot(buy_hold_value.index, buy_hold_value, label='Buy & Hold 전략', color='red', linewidth=2)

plt.title(f'[{ticker}] ESN 전략 vs. Buy & Hold 성과 ({backtest_index.min().strftime("%Y-%m-%d")} ~ {backtest_index.max().strftime("%Y-%m-%d")})')
plt.xlabel('날짜')
plt.ylabel('포트폴리오 가치 ($)')
plt.legend()
plt.grid(True)
plt.tight_layout()

output_comparison_plot_file = os.path.join(base_data_directory, ticker, f'{ticker}_esn_vs_buyhold_comparison_plot.png')
plt.savefig(output_comparison_plot_file)
print(f"\n전략 비교 그래프가 '{output_comparison_plot_file}'에 저장되었습니다.")
plt.show()

print("\n스크립트 실행 완료.")
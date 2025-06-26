import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # 데이터 시각화 라이브러리 추가
from scipy import stats # 통계 분석 (Q-Q Plot)

# ------------------------
# 1️⃣ 종목 및 기간 설정
# ------------------------
tickers = ['AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'BRK-B', 'TSLA', 'AVGO', 'GOOG']
start_date = '2015-01-01'
end_date = '2023-01-01'

print("--- EDA 시작 ---")

# ------------------------
# 2️⃣ 데이터 다운로드
# ------------------------
print(f"\n2️⃣ 데이터 다운로드 및 전처리: {start_date} ~ {end_date}")
raw_data = yf.download(tickers, start=start_date, end=end_date)

# 종가(Close)와 거래량(Volume) 데이터 추출
close_data = raw_data['Close']
volume_data = raw_data['Volume']

print("\n✅ 데이터 로드 완료.")

# ------------------------
# 3️⃣ 데이터 개요 및 통계
# ------------------------
print("\n--- 3️⃣ 데이터 개요 및 통계 ---")
print("\n[Close Price Data Info]")
close_data.info()
print("\n[Volume Data Info]")
volume_data.info()

print("\n[Close Price Data Descriptive Statistics]")
print(close_data.describe())
print("\n[Volume Data Descriptive Statistics]")
print(volume_data.describe())
print("✅ 데이터 개요 및 통계 출력 완료.")

# ------------------------
# 4️⃣ 로그 수익률 계산 및 분석
# ------------------------
print("\n--- 4️⃣ 로그 수익률 계산 및 분석 ---")
log_returns = np.log(close_data / close_data.shift(1)).dropna()
print("✅ 로그 수익률 계산 완료.")

# 로그 수익률 통계
print("\n[Log Returns Descriptive Statistics (Overall)]")
print(log_returns.describe())

# 종목별 로그 수익률 분포 시각화 (히스토그램 + KDE)
num_rows_hist = (len(tickers) + 2) // 3 # 3열로 배치
fig_hist, axes_hist = plt.subplots(num_rows_hist, 3, figsize=(18, 5 * num_rows_hist))
axes_hist = axes_hist.flatten()
plt.suptitle('Distribution of Log Returns (Histogram + KDE)', fontsize=20, y=1.02)

for i, ticker in enumerate(tickers):
    sns.histplot(log_returns[ticker], kde=True, ax=axes_hist[i], bins=50, color='blue')
    axes_hist[i].set_title(f'{ticker} Log Returns')
    axes_hist[i].set_xlabel('Log Return')
    axes_hist[i].set_ylabel('Density / Count')
    axes_hist[i].grid(True, linestyle='--', alpha=0.6)

# 남은 빈 서브플롯 숨기기
for j in range(i + 1, len(axes_hist)):
    fig_hist.delaxes(axes_hist[j])

plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.show()

# 종목별 로그 수익률 Q-Q Plot (정규성 시각적 확인)
num_rows_qq = (len(tickers) + 2) // 3
fig_qq, axes_qq = plt.subplots(num_rows_qq, 3, figsize=(18, 5 * num_rows_qq))
axes_qq = axes_qq.flatten()
plt.suptitle('Q-Q Plot of Log Returns vs. Normal Distribution', fontsize=20, y=1.02)

for i, ticker in enumerate(tickers):
    stats.probplot(log_returns[ticker], dist="norm", plot=axes_qq[i])
    axes_qq[i].set_title(f'{ticker} Q-Q Plot')
    axes_qq[i].set_xlabel('Theoretical Quantiles')
    axes_qq[i].set_ylabel('Ordered Values')

# 남은 빈 서브플롯 숨기기
for j in range(i + 1, len(axes_qq)):
    fig_qq.delaxes(axes_qq[j])

plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.show()
print("✅ 로그 수익률 분포 및 Q-Q Plot 시각화 완료.")

# ------------------------
# 5️⃣ 변동성 분석
# ------------------------
print("\n--- 5️⃣ 변동성 분석 ---")
# 일일 변동성 (로그 수익률의 이동 표준편차)
daily_volatility = log_returns.rolling(window=20).std().dropna() # 20일 이동 표준편차

# 연간 변동성 (일일 변동성 * sqrt(거래일 수))
annualized_volatility = daily_volatility * np.sqrt(252)

print("\n[Annualized Volatility (Last Value)]")
print(annualized_volatility.iloc[-1].sort_values(ascending=False))

# 일일 변동성 시계열 그래프
num_rows_vol = (len(tickers) + 2) // 3
fig_vol, axes_vol = plt.subplots(num_rows_vol, 3, figsize=(18, 5 * num_rows_vol))
axes_vol = axes_vol.flatten()
plt.suptitle('Daily Volatility (20-Day Rolling Std Dev of Log Returns)', fontsize=20, y=1.02)

for i, ticker in enumerate(tickers):
    axes_vol[i].plot(daily_volatility.index, daily_volatility[ticker], color='purple')
    axes_vol[i].set_title(f'{ticker} Daily Volatility')
    axes_vol[i].set_xlabel('Date')
    axes_vol[i].set_ylabel('Volatility')
    axes_vol[i].grid(True, linestyle='--', alpha=0.6)

# 남은 빈 서브플롯 숨기기
for j in range(i + 1, len(axes_vol)):
    fig_vol.delaxes(axes_vol[j])

plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.show()
print("✅ 변동성 분석 완료.")

# ------------------------
# 6️⃣ 상관 관계 분석 (로그 수익률)
# ------------------------
print("\n--- 6️⃣ 상관 관계 분석 (로그 수익률) ---")
correlation_matrix = log_returns.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix of Log Returns', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
print("✅ 종목 간 상관 관계 히트맵 시각화 완료.")

# ------------------------
# 7️⃣ 거래량 분포 (이상치 탐지는 포함하지 않음)
# ------------------------
print("\n--- 7️⃣ 거래량 분포 ---")

# 거래량 분포 (히스토그램 + KDE) - 거래량이 매우 왜곡되어 있을 수 있으므로 로그 스케일 거래량으로 시각화
log_volume_data = np.log(volume_data + 1) # 거래량은 로그 스케일로 보는 것이 분포 파악에 용이
num_rows_vol_dist = (len(tickers) + 2) // 3
fig_vol_dist, axes_vol_dist = plt.subplots(num_rows_vol_dist, 3, figsize=(18, 5 * num_rows_vol_dist))
axes_vol_dist = axes_vol_dist.flatten()
plt.suptitle('Distribution of Log(Volume + 1) (Histogram + KDE)', fontsize=20, y=1.02)

for i, ticker in enumerate(tickers):
    sns.histplot(log_volume_data[ticker], kde=True, ax=axes_vol_dist[i], bins=50, color='orange')
    axes_vol_dist[i].set_title(f'{ticker} Log(Volume + 1)')
    axes_vol_dist[i].set_xlabel('Log(Volume + 1)')
    axes_vol_dist[i].set_ylabel('Density / Count')
    axes_vol_dist[i].grid(True, linestyle='--', alpha=0.6)

# 남은 빈 서브플롯 숨기기
for j in range(i + 1, len(axes_vol_dist)):
    fig_vol_dist.delaxes(axes_vol_dist[j])

plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.show()
print("✅ 로그 거래량 분포 시각화 완료.")

# ------------------------
# 8️⃣ 종가 + 이동평균선 시계열 그래프
# ------------------------
print("\n--- 8️⃣ 종가 + 이동평균선 시계열 그래프 ---")
ma_windows = [20, 60, 120]
ma_data = {ticker: pd.DataFrame(index=close_data.index) for ticker in tickers}

for ticker in tickers:
    ma_data[ticker]['Close'] = close_data[ticker]
    for window in ma_windows:
        ma_data[ticker][f'MA{window}'] = close_data[ticker].rolling(window=window).mean()

for ticker in tickers:
    fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # (1) 로그 수익률 그래프
    axs[0].plot(log_returns.index, log_returns[ticker], label='Log Return', color='gray', alpha=0.8)
    axs[0].set_title(f'{ticker} - Log Returns', fontsize=14)
    axs[0].set_ylabel('Log Return')
    axs[0].grid(True, linestyle='--', alpha=0.6)
    axs[0].legend()


    # (2) 종가 + 이동평균선 그래프
    axs[1].plot(close_data.index, close_data[ticker], label='Close Price', color='black')
    axs[1].plot(ma_data[ticker].index, ma_data[ticker]['MA20'], label='MA20', color='blue')
    axs[1].plot(ma_data[ticker].index, ma_data[ticker]['MA60'], label='MA60', color='orange')
    axs[1].plot(ma_data[ticker].index, ma_data[ticker]['MA120'], label='MA120', color='green')
    axs[1].set_title(f'{ticker} - Close Price & Moving Averages', fontsize=14)
    axs[1].set_ylabel('Price')
    axs[1].legend()
    axs[1].grid(True, linestyle='--', alpha=0.6)
    axs[1].set_xlabel('Date')


    plt.tight_layout()
    plt.show()
print("✅ 종가 및 이동평균선 그래프 시각화 완료.")
print("\n--- EDA 종료 ---")
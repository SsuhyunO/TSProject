import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# 종목 및 기간 설정
tickers = ['AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'BRK-B', 'TSLA', 'AVGO', 'GOOG']
start_date = '2015-01-01'
end_date = '2023-01-01'

# 데이터 불러오기: Close 가격과 Volume 데이터 모두 불러옴
raw_data = yf.download(tickers, start=start_date, end=end_date)
close_data = raw_data['Close']
volume_data = raw_data['Volume']

# 결측치 처리 (Interpolate, ffill, bfill)
close_data = close_data.interpolate(method='time').ffill().bfill()
volume_data = volume_data.interpolate(method='time').ffill().bfill()

# --- (선택 사항) 결측치 확인 시각화 및 출력 - 필요 시 활성화 ---
# plt.figure(figsize=(12, 6))
# close_data.count().plot(kind='bar', color='skyblue')
# plt.title('Number of Data Points per Stock (Confirming No Missing Values)')
# plt.xlabel('Ticker')
# plt.ylabel('Number of Data Points')
# plt.xticks(rotation=45)
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.show()
# print("\n--- Missing Value Ratios After Preprocessing ---")
# print("Close Data Missing Ratios:\n", close_data.isnull().mean().sort_values(ascending=False))
# print("\nVolume Data Missing Ratios:\n", volume_data.isnull().mean().sort_values(ascending=False))
# print("-" * 50)
# --- (선택 사항) 끝 ---


# 로그 스케일 변환 (Close 가격 및 Volume)
# 로그 변환은 분포의 왜곡을 줄이고 이상치의 영향을 완화하는 데 도움을 줍니다.
# log(x+1)은 0 값을 처리하기 위해 사용합니다.
log_close_data = np.log(close_data + 1)
# log_volume_data = np.log(volume_data + 1) # 이번 시각화에서는 원본 거래량을 사용할 것이므로 주석 처리 또는 제거

# --- 1. 로그 스케일 종가 박스플롯 시각화 및 이상치 종목 감지 ---
num_tickers = len(tickers)
num_cols = 5 # 한 줄에 5개 종목
num_rows = (num_tickers + num_cols - 1) // num_cols

fig_log_boxplot, axes_log_boxplot = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows))
axes_log_boxplot = axes_log_boxplot.flatten()

plt.suptitle('Boxplot of Log-Scaled Close Price (with IQR Outliers)', fontsize=18, y=1.02)
box_color = 'skyblue'

# 로그 스케일 데이터에서 IQR 이상치 감지 및 저장
iqr_outliers_log_by_ticker = {}
tickers_with_log_outliers = []

print("\n--- Detecting IQR Outliers for Each Ticker's Log-Scaled Close Price ---")
for i, ticker in enumerate(tickers):
    ax = axes_log_boxplot[i]
    data_for_plot_log = log_close_data[ticker].dropna()

    # 박스플롯 그리기 (IQR 이상치는 기본으로 표시됨)
    bp = ax.boxplot(data_for_plot_log, patch_artist=True,
                   boxprops=dict(facecolor=box_color, color='gray'),
                   medianprops=dict(color='black'),
                   whiskerprops=dict(color='gray'),
                   capprops=dict(color='gray'),
                   flierprops=dict(marker='o', markerfacecolor='red', markersize=5, linestyle='none')) # 이상치 점 색상

    ax.set_title(ticker)
    ax.set_ylabel('Log(Close Price + 1)')
    ax.set_xticks([]) # x축 틱 제거 (하나의 박스만 있으므로 필요 없음)
    ax.grid(True, linestyle=':', alpha=0.7)

    # IQR 이상치 데이터 추출 (로그 스케일 데이터 기준)
    q1 = data_for_plot_log.quantile(0.25)
    q3 = data_for_plot_log.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers_df_log = data_for_plot_log[(data_for_plot_log < lower_bound) | (data_for_plot_log > upper_bound)]

    if not outliers_df_log.empty:
        iqr_outliers_log_by_ticker[ticker] = outliers_df_log
        tickers_with_log_outliers.append(ticker)
        print(f"'{ticker}' has {len(outliers_df_log)} IQR outliers in log-scaled data.")
    else:
        print(f"'{ticker}' has no IQR outliers in log-scaled data.")


# 남은 빈 서브플롯 숨기기
for j in range(i + 1, len(axes_log_boxplot)):
    fig_log_boxplot.delaxes(axes_log_boxplot[j])

plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.show()

print("-" * 60)

# --- 2. 로그 스케일 박스플롯에서 이상치가 감지된 종목에 대해 원본 Close Price & Volume 시계열 시각화 ---
print("\n--- Visualizing RAW Price & Volume for Log-Scaled Outlier-Detected Tickers ---")

if not tickers_with_log_outliers:
    print("No tickers with IQR outliers detected in log-scaled Close Price to visualize raw data.")
else:
    for ticker in tickers_with_log_outliers:
        # 로그 스케일 박스플롯에서 감지된 이상치 날짜 가져오기 (해당 날짜에 대한 원본 데이터를 확인할 것임)
        # iqr_outliers_log_by_ticker[ticker]는 Series이고, index가 날짜입니다.
        outlier_dates = iqr_outliers_log_by_ticker[ticker].index

        plt.figure(figsize=(16, 8))

        # 첫 번째 서브플롯: 원본 종가 시계열 및 이상치 발생 시점 표시
        ax1 = plt.subplot(2, 1, 1) # 2행 1열 중 첫 번째
        ax1.plot(close_data.index, close_data[ticker], alpha=0.8, label='RAW Close Price', color='blue')
        # 이상치 발생 시점의 원본 가격을 마커로 표시
        ax1.scatter(outlier_dates, close_data.loc[outlier_dates, ticker], color='red', s=70, zorder=5, label='Price Outliers (from Log-Scale IQR)')
        ax1.set_title(f"{ticker} - RAW Price & Volume (Log-Scale IQR Outliers Marked)")
        ax1.set_ylabel("Close Price")
        ax1.legend(loc='upper left')
        ax1.grid(True, linestyle='--', alpha=0.5)
        # 이상치 시점 수직선 표시
        for outlier_date in outlier_dates:
            ax1.axvline(x=outlier_date, color='gray', linestyle='--', lw=1, alpha=0.7)

        # 두 번째 서브플롯: 원본 거래량 시계열
        ax2 = plt.subplot(2, 1, 2, sharex=ax1) # 2행 1열 중 두 번째, X축 공유
        ax2.plot(volume_data.index, volume_data[ticker], alpha=0.8, label='RAW Volume', color='green')
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Volume")
        ax2.legend(loc='upper left')
        ax2.grid(True, linestyle='--', alpha=0.5)
        # 이상치 시점 수직선 표시 (거래량 차트에도 동일하게)
        for outlier_date in outlier_dates:
            ax2.axvline(x=outlier_date, color='gray', linestyle='--', lw=1, alpha=0.7)

        plt.tight_layout()
        plt.show()

# Min-Max 정규화 (이전 코드에서 유지)
scaler = MinMaxScaler()
close_data_scaled = pd.DataFrame(scaler.fit_transform(close_data),
                                 columns=close_data.columns,
                                 index=close_data.index)

# 정규화 결과 확인 (일부 출력)
print("\nMin-Max Normalized Close Data (Head):\n", close_data_scaled.head())
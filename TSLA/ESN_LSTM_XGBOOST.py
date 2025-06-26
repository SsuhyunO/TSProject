import pandas as pd
import numpy as np
import os

from TSProject.NVDA.optimized_ESN import FIXED_SPECTRAL_RADIUS
from pyESN import ESN
import matplotlib.pyplot as plt
import warnings
import yfinance as yf
import vectorbt as vbt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import xgboost as xgb
from scipy.stats import mannwhitneyu
import inspect # 디버깅을 위해 inspect 모듈 임포트

warnings.filterwarnings('ignore')

# --- 0. 고정된 최적 파라미터 설정 ---
FIXED_N_RESERVOIR = 1000
FIXED_SPECTRAL_RADIUS = 0.85
FIXED_THRESHOLD_BUY = 0.5
FIXED_THRESHOLD_SELL = -0.5

# LSTM 파라미터 (예시, 최적화 필요)
LSTM_UNITS = 128
LSTM_DROPOUT = 0.2
LSTM_EPOCHS = 50 # EarlyStopping 사용
LSTM_BATCH_SIZE = 32
LSTM_LOOK_BACK = 5 # 시계열 데이터의 과거 몇 단계를 볼 것인가

# XGBoost 파라미터 (예시, 최적화 필요)
XGB_N_ESTIMATORS = 300
XGB_MAX_DEPTH = 5
XGB_LEARNING_RATE = 0.1
XGB_SUBSAMPLE = 0.8

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
    current_today = pd.to_datetime('today')
    if overall_end_date > current_today:
        print(f"경고: 데이터 종료일 {overall_end_date.strftime('%Y-%m-%d')}이 미래입니다. 오늘 날짜로 조정합니다.")
        overall_end_date = current_today

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
full_index = merged_df.index # 날짜 인덱스를 NumPy 배열과 동기화하기 위해 저장

print(f"\n최종 정렬된 X_full 길이: {len(X_full)}")
print(f"최종 정렬된 y_full 길이: {len(y_full)}")
print(f"최종 merged_df (인덱스) 시작: {merged_df.index.min()}, 끝: {merged_df.index.max()}")

# --- 데이터 전처리 및 시계열 변환 함수 (LSTM용) ---
def create_dataset(X, y, look_back=1):
    dataX, dataY = [], []
    # look_back + 1 이유는 y[i + look_back] 때문 (i부터 look_back까지의 X로 look_back 다음의 y를 예측)
    for i in range(len(X) - look_back): # 이전에는 -1이 있었는데, 마지막 데이터까지 사용하기 위해 제거
        dataX.append(X[i:(i + look_back), :])
        dataY.append(y[i + look_back])
    return np.array(dataX), np.array(dataY)


# --- 모델 학습 및 백테스팅 통합 함수 ---
def run_model_backtest(model_name, train_X, train_y, test_X, test_prices, test_index,
                       esn_params=None, lstm_params=None, xgb_params=None):
    """
    각 모델을 학습하고 백테스팅을 수행하여 성능 지표를 반환합니다.
    """
    model_pred_raw = None
    original_test_prices = test_prices
    original_test_index = test_index

    print(f"\n[DEBUG - {model_name}] --- 모델 학습 및 예측 시작 ---")
    print(f"  Train X shape: {train_X.shape}, Train y shape: {train_y.shape}")
    print(f"  Test X shape: {test_X.shape}, Test prices shape: {test_prices.shape}")

    if model_name == 'ESN':
        esn = ESN(
            n_inputs=train_X.shape[1],
            n_outputs=train_y.shape[1] if train_y.ndim > 1 else 1,
            n_reservoir=esn_params['n_reservoir'],
            spectral_radius=esn_params['spectral_radius'],
            sparsity=0.5,
            random_state=42,
            silent=True
        )
        esn.fit(train_X, train_y)
        model_pred_raw = esn.predict(test_X)

    elif model_name == 'LSTM':
        scaler_X = MinMaxScaler(feature_range=(0, 1))
        train_X_scaled = scaler_X.fit_transform(train_X)
        test_X_scaled = scaler_X.transform(test_X)

        scaler_y = MinMaxScaler(feature_range=(0, 1))
        train_y_scaled = scaler_y.fit_transform(train_y.reshape(-1, 1))

        look_back = lstm_params['look_back']
        train_X_lstm, train_y_lstm = create_dataset(train_X_scaled, train_y_scaled.flatten(), look_back)
        test_X_lstm, _ = create_dataset(test_X_scaled, np.zeros(len(test_X_scaled)).flatten(), look_back)

        print(f"  LSTM look_back: {look_back}")
        print(f"  LSTM train_X_lstm shape: {train_X_lstm.shape}, train_y_lstm shape: {train_y_lstm.shape}")
        print(f"  LSTM test_X_lstm shape: {test_X_lstm.shape}")


        lstm_model = Sequential([
            LSTM(lstm_params['units'], return_sequences=True, input_shape=(look_back, train_X.shape[1])),
            Dropout(lstm_params['dropout']),
            LSTM(lstm_params['units']),
            Dropout(lstm_params['dropout']),
            Dense(1)
        ])
        lstm_model.compile(loss='mean_squared_error', optimizer='adam')
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        val_split_idx = int(len(train_X_lstm) * 0.8)
        if len(train_X_lstm) > 0 and val_split_idx < len(train_X_lstm):
            history = lstm_model.fit(
                train_X_lstm[:val_split_idx], train_y_lstm[:val_split_idx],
                epochs=lstm_params['epochs'],
                batch_size=lstm_params['batch_size'],
                validation_data=(train_X_lstm[val_split_idx:], train_y_lstm[val_split_idx:]),
                callbacks=[early_stopping],
                verbose=0
            )
        elif len(train_X_lstm) > 0:
             history = lstm_model.fit(
                train_X_lstm, train_y_lstm,
                epochs=lstm_params['epochs'],
                batch_size=lstm_params['batch_size'],
                callbacks=[early_stopping],
                verbose=0
            )
        else:
            print(f"경고: LSTM 훈련 데이터가 부족하여 학습을 건너뜁니다.")
            return {'Total Return [%]': np.nan, 'Max Drawdown [%]': np.nan, 'Sharpe Ratio': np.nan, 'Final Value': np.nan}


        if len(test_X_lstm) == 0:
            print(f"경고: LSTM 예측을 위한 테스트 데이터가 부족합니다.")
            return {'Total Return [%]': np.nan, 'Max Drawdown [%]': np.nan, 'Sharpe Ratio': np.nan, 'Final Value': np.nan}

        lstm_pred_scaled = lstm_model.predict(test_X_lstm, verbose=0)
        model_pred_raw = scaler_y.inverse_transform(lstm_pred_scaled).flatten()

        test_prices = original_test_prices.iloc[look_back:]
        test_index = original_test_index[look_back:]

        print(f"  LSTM 조정된 test_prices 길이: {len(test_prices)}, test_index 길이: {len(test_index)}")
        print(f"  LSTM 예측값 (model_pred_raw) 길이: {len(model_pred_raw)}")


    elif model_name == 'XGBoost':
        xgb_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=xgb_params['n_estimators'],
            max_depth=xgb_params['max_depth'],
            learning_rate=xgb_params['learning_rate'],
            subsample=xgb_params['subsample'],
            random_state=42,
            n_jobs=-1
        )
        xgb_model.fit(train_X, train_y)
        model_pred_raw = xgb_model.predict(test_X)

    if model_pred_raw is None or len(model_pred_raw) == 0:
        print(f"경고: {model_name} - 예측 결과가 없거나 비어 있습니다.")
        return {'Total Return [%]': np.nan, 'Max Drawdown [%]': np.nan, 'Sharpe Ratio': np.nan, 'Final Value': np.nan}

    print(f"  {model_name} 원시 예측값 (model_pred_raw)의 min: {np.min(model_pred_raw):.4f}, max: {np.max(model_pred_raw):.4f}, mean: {np.mean(model_pred_raw):.4f}")


    # 예측된 신호를 1, -1, 0으로 분류
    classified_signals = np.zeros_like(model_pred_raw, dtype=int)
    classified_signals[model_pred_raw > FIXED_THRESHOLD_BUY] = 1
    classified_signals[model_pred_raw < FIXED_THRESHOLD_SELL] = -1

    signals_series = pd.Series(classified_signals.flatten(), index=test_index)

    print(f"  {model_name} 생성된 신호 분포 (1: {np.sum(classified_signals == 1)}, -1: {np.sum(classified_signals == -1)}, 0: {np.sum(classified_signals == 0)})")
    if np.sum(classified_signals != 0) == 0:
        print(f"경고: {model_name} - 모든 신호가 0입니다. 거래가 발생하지 않습니다.")
        return {'Total Return [%]': 0.0, 'Max Drawdown [%]': np.nan, 'Sharpe Ratio': np.inf, 'Final Value': 10000.0} # 0% 수익률, NaN MDD, Inf Sharpe

    # Vectorbt 백테스팅
    price_data_for_vbt = test_prices.sort_index().drop_duplicates()
    signals_series_for_vbt = signals_series.reindex(price_data_for_vbt.index).fillna(method='ffill').fillna(0)

    print(f"  Vectorbt 입력 - Prices 길이: {len(price_data_for_vbt)}, Signals 길이: {len(signals_series_for_vbt)}")
    print(f"  Vectorbt 입력 - Prices Index Min: {price_data_for_vbt.index.min()}, Max: {price_data_for_vbt.index.max()}")
    print(f"  Vectorbt 입력 - Signals Index Min: {signals_series_for_vbt.index.min()}, Max: {signals_series_for_vbt.index.max()}")

    if price_data_for_vbt.empty or len(signals_series_for_vbt) < 2 or len(price_data_for_vbt) != len(signals_series_for_vbt):
        print(f"경고: {model_name} - 백테스팅을 위한 데이터가 너무 적거나 인덱스 불일치. (Prices: {len(price_data_for_vbt)}, Signals: {len(signals_series_for_vbt)})")
        return {'Total Return [%]': np.nan, 'Max Drawdown [%]': np.nan, 'Sharpe Ratio': np.nan, 'Final Value': np.nan}


    try:
        portfolio = vbt.Portfolio.from_signals(
            price_data_for_vbt,
            entries=signals_series_for_vbt == 1,
            exits=signals_series_for_vbt == 0,
            short_entries=signals_series_for_vbt == -1,
            short_exits=signals_series_for_vbt == 0,
            init_cash=10000,
            fees=0.001,
            slippage=0.0001,
            freq='1D',
        )
        stats = portfolio.stats()
        return {
            'Total Return [%]': stats['Total Return [%]'],
            'Max Drawdown [%]': stats['Max Drawdown [%]'],
            'Sharpe Ratio': stats['Sharpe Ratio'],
            'Final Value': portfolio.final_value()
        }
    except Exception as e:
        print(f"오류: {model_name} 백테스팅 중 오류 발생 - {e}")
        return {'Total Return [%]': np.nan, 'Max Drawdown [%]': np.nan, 'Sharpe Ratio': np.nan, 'Final Value': np.nan}

# --- 확장 윈도우 (Expanding Window) 백테스팅 설정 ---
# 초기 훈련 윈도우 크기 (예: 3년 = 252 * 3 거래일)
initial_train_window_size = 252 * 3
# 테스트 윈도우 크기 (예: 3개월 = 63 거래일)
test_window_size = 63
# 윈도우 이동 간격 (테스트 윈도우 크기와 동일하게 설정하여 겹치지 않게 함)
step_size = test_window_size

# 전체 데이터 길이
data_length = len(X_full)

# 결과를 저장할 딕셔너리
model_results = {
    'ESN': {'returns': [], 'mdds': [], 'sharpes': []},
    'LSTM': {'returns': [], 'mdds': [], 'sharpes': []},
    'XGBoost': {'returns': [], 'mdds': [], 'sharpes': []}
}

# 백테스팅 시작 인덱스 (초기 훈련 윈도우 이후부터 시작)
start_test_index = initial_train_window_size

# 루프는 테스트 윈도우의 시작점을 기준으로 반복
for i in range(start_test_index, data_length, step_size):
    # 훈련 데이터는 시작은 고정하고 끝은 현재 i까지 확장
    train_start_i = 0 # 확장 윈도우의 핵심: 훈련 데이터 시작은 항상 0
    train_end_i = i

    # 테스트 데이터는 현재 i부터 고정된 test_window_size만큼
    test_start_i = i
    test_end_i = min(i + test_window_size, data_length)

    # 테스트 윈도우가 최소 길이를 충족하는지 확인
    if test_end_i - test_start_i < 2:
        print(f"경고: 테스트 기간이 너무 짧아 (시작: {test_start_i}, 끝: {test_end_i}) 남은 윈도우를 스킵합니다.")
        continue

    # 훈련 데이터도 최소 길이를 충족하는지 확인
    if train_end_i - train_start_i < initial_train_window_size:
        print(f"경고: 훈련 기간이 초기 설정치보다 짧아 (현재: {train_end_i - train_start_i}) 스킵합니다.")
        continue


    current_train_X = X_full[train_start_i:train_end_i]
    current_train_y = y_full[train_start_i:train_end_i]
    current_test_X = X_full[test_start_i:test_end_i]
    current_test_prices = actual_prices_series.iloc[test_start_i:test_end_i]
    current_test_index = full_index[test_start_i:test_end_i]

    print(f"\n--- 확장 윈도우 {test_start_i} ~ {test_end_i} ({current_test_index.min().strftime('%Y-%m-%d')} ~ {current_test_index.max().strftime('%Y-%m-%d')}) ---")
    print(f"훈련 데이터 기간: {full_index[train_start_i].strftime('%Y-%m-%d')} ~ {full_index[train_end_i-1].strftime('%Y-%m-%d')}")
    print(f"테스트 데이터 기간: {current_test_index.min().strftime('%Y-%m-%d')} ~ {current_test_index.max().strftime('%Y-%m-%d')}")

    # ESN 백테스팅
    esn_stats = run_model_backtest(
        'ESN',
        current_train_X, current_train_y, current_test_X, current_test_prices, current_test_index,
        esn_params={'n_reservoir': FIXED_N_RESERVOIR, 'spectral_radius': FIXED_SPECTRAL_RADIUS}
    )
    if not np.isnan(esn_stats['Total Return [%]']): # NaN 값이 아니면 저장
        model_results['ESN']['returns'].append(esn_stats['Total Return [%]'])
        model_results['ESN']['mdds'].append(esn_stats['Max Drawdown [%]'])
        model_results['ESN']['sharpes'].append(esn_stats['Sharpe Ratio'])
    print(f"ESN 결과 - 누적 수익률: {esn_stats['Total Return [%]']:.2f}%, MDD: {esn_stats['Max Drawdown [%]']:.2f}%, Sharpe: {esn_stats['Sharpe Ratio']:.2f}")

    # LSTM 백테스팅
    lstm_stats = run_model_backtest(
        'LSTM',
        current_train_X, current_train_y, current_test_X, current_test_prices, current_test_index,
        lstm_params={
            'units': LSTM_UNITS, 'dropout': LSTM_DROPOUT, 'epochs': LSTM_EPOCHS,
            'batch_size': LSTM_BATCH_SIZE, 'look_back': LSTM_LOOK_BACK
        }
    )
    if not np.isnan(lstm_stats['Total Return [%]']):
        model_results['LSTM']['returns'].append(lstm_stats['Total Return [%]'])
        model_results['LSTM']['mdds'].append(lstm_stats['Max Drawdown [%]'])
        model_results['LSTM']['sharpes'].append(lstm_stats['Sharpe Ratio'])
    print(f"LSTM 결과 - 누적 수익률: {lstm_stats['Total Return [%]']:.2f}%, MDD: {lstm_stats['Max Drawdown [%]']:.2f}%, Sharpe: {lstm_stats['Sharpe Ratio']:.2f}")

    # XGBoost 백테스팅
    xgb_stats = run_model_backtest(
        'XGBoost',
        current_train_X, current_train_y, current_test_X, current_test_prices, current_test_index,
        xgb_params={
            'n_estimators': XGB_N_ESTIMATORS, 'max_depth': XGB_MAX_DEPTH,
            'learning_rate': XGB_LEARNING_RATE, 'subsample': XGB_SUBSAMPLE
        }
    )
    if not np.isnan(xgb_stats['Total Return [%]']):
        model_results['XGBoost']['returns'].append(xgb_stats['Total Return [%]'])
        model_results['XGBoost']['mdds'].append(xgb_stats['Max Drawdown [%]'])
        model_results['XGBoost']['sharpes'].append(xgb_stats['Sharpe Ratio'])
    print(f"XGBoost 결과 - 누적 수익률: {xgb_stats['Total Return [%]']:.2f}%, MDD: {xgb_stats['Max Drawdown [%]']:.2f}%, Sharpe: {xgb_stats['Sharpe Ratio']:.2f}")


# --- Mann-Whitney U 검정 및 결과 출력 ---
print("\n--- Mann-Whitney U 검정 결과 ---")

metrics = ['returns', 'mdds', 'sharpes']
model_names = ['ESN', 'LSTM', 'XGBoost']

for metric in metrics:
    print(f"\nMetric: {metric.replace('returns', '누적 수익률').replace('mdds', 'MDD').replace('sharpes', 'Sharpe 비율')}")
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            model1_name = model_names[i]
            model2_name = model_names[j]

            data1 = np.array(model_results[model1_name][metric])
            data2 = np.array(model_results[model2_name][metric])

            # 데이터에 NaN이 있으면 제거 (백테스팅 실패 등으로 발생할 수 있음)
            data1 = data1[~np.isnan(data1)]
            data2 = data2[~np.isnan(data2)]

            if len(data1) < 2 or len(data2) < 2:
                print(f"  경고: {model1_name} vs {model2_name} - {metric} 비교를 위한 데이터 포인트가 부족합니다. (데이터1: {len(data1)}, 데이터2: {len(data2)})")
                continue

            stat, p_value = mannwhitneyu(data1, data2, alternative='two-sided') # 양측 검정

            print(f"  {model1_name} vs {model2_name}: U-statistic = {stat:.2f}, p-value = {p_value:.3f}")
            if p_value < 0.05:
                print(f"    -> 통계적으로 유의미한 차이가 있습니다 (p < 0.05).")
                # 어떤 모델이 더 좋은지 평균으로 간단히 판단
                avg1 = np.mean(data1)
                avg2 = np.mean(data2)
                if metric == 'mdds': # MDD는 낮을수록 좋음
                    if avg1 < avg2:
                        print(f"    ({model1_name}의 평균 {metric}: {avg1:.2f}이(가) {model2_name}의 평균 {metric}: {avg2:.2f}보다 좋습니다.)")
                    else:
                        print(f"    ({model2_name}의 평균 {metric}: {avg2:.2f}이(가) {model1_name}의 평균 {metric}: {avg1:.2f}보다 좋습니다.)")
                else: # 수익률, 샤프 비율은 높을수록 좋음
                    if avg1 > avg2:
                        print(f"    ({model1_name}의 평균 {metric}: {avg1:.2f}이(가) {model2_name}의 평균 {metric}: {avg2:.2f}보다 좋습니다.)")
                    else:
                        print(f"    ({model2_name}의 평균 {metric}: {avg2:.2f}이(가) {model1_name}의 평균 {metric}: {avg1:.2f}보다 좋습니다.)")
            else:
                print(f"    -> 통계적으로 유의미한 차이가 없습니다 (p >= 0.05).")


# --- 결과 시각화 (박스 플롯) ---
print("\n--- 성능 지표 박스 플롯 ---")

fig, axes = plt.subplots(1, 3, figsize=(20, 7))
fig.suptitle(f'[{ticker}] 모델별 성능 지표 비교 (확장 윈도우 백테스팅)')

# 누적 수익률
returns_data = [model_results[model]['returns'] for model in model_names]
axes[0].boxplot(returns_data, labels=model_names, vert=True, patch_artist=True)
axes[0].set_title('누적 수익률 (%)')
axes[0].set_ylabel('수익률 (%)')
axes[0].grid(axis='y', linestyle='--', alpha=0.7)

# MDD
mdds_data = [model_results[model]['mdds'] for model in model_names]
axes[1].boxplot(mdds_data, labels=model_names, vert=True, patch_artist=True)
axes[1].set_title('최대 낙폭 (MDD) (%)')
axes[1].set_ylabel('MDD (%)')
axes[1].grid(axis='y', linestyle='--', alpha=0.7)

# Sharpe Ratio
sharpes_data = [model_results[model]['sharpes'] for model in model_names]
axes[2].boxplot(sharpes_data, labels=model_names, vert=True, patch_artist=True)
axes[2].set_title('샤프 비율')
axes[2].set_ylabel('샤프 비율')
axes[2].grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # adjust layout to make space for suptitle
output_comparison_plot_file = os.path.join(base_data_directory, ticker, f'{ticker}_model_comparison_expanding_window_boxplot.png')
plt.savefig(output_comparison_plot_file)
print(f"\n모델 비교 박스 플롯이 '{output_comparison_plot_file}'에 저장되었습니다.")
plt.show()

print("\n스크립트 실행 완료.")
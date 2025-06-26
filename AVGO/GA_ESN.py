import pandas as pd
import numpy as np
import os
from pyESN import ESN
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import warnings
import random
import multiprocessing

# DEAP 라이브러리 임포트
from deap import base, creator, tools, algorithms

warnings.filterwarnings('ignore')

# --- 0. DEAP 설정 ---
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# --- 1. 파일 경로 설정 및 데이터 로드 (기존과 동일) ---
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

merged_df = pd.concat([df_gdc['GDC_Signal'], df_roc['ROC_Signal'], df_rsi['RSI_Signal'], df_target['Target_Signal']],
                      axis=1)
merged_df.dropna(inplace=True)

X_full = merged_df[['GDC_Signal', 'ROC_Signal', 'RSI_Signal']].values
y_full = merged_df['Target_Signal'].values.astype(float)

# 훈련/테스트 데이터 분할 (단일 분할 - GA 평가시 항상 동일한 분할 사용)
train_split_ratio = 0.8
split_index = int(len(X_full) * train_split_ratio)

X_train_global = X_full[:split_index]
y_train_global = y_full[:split_index]
X_test_global = X_full[split_index:]
y_test_global = y_full[split_index:]

# --- 고정된 임계값 설정 ---
FIXED_THRESHOLD_BUY = 0.7
FIXED_THRESHOLD_SELL = -0.7


# --- 2. GA의 평가 함수 정의 (evaluate_individual) ---
def evaluate_individual(individual):
    """
    DEAP의 평가 함수. 각 개체(individual)의 성능을 계산하여 반환.
    individual: GA가 제안하는 파라미터 조합 [n_reservoir, spectral_radius]
    """
    n_reservoir, spectral_radius = individual

    # 파라미터 제약 조건 및 유효성 검사 (조정된 범위에 따라 업데이트)
    if not (500 <= n_reservoir <= 1000 and
            0.7 <= spectral_radius <= 1.0):
        return -float('inf'),

    n_reservoir = int(n_reservoir)

    # ESN 모델 구축 및 학습 (단일 학습)
    try:
        esn = ESN(
            n_inputs=X_train_global.shape[1],
            n_outputs=y_train_global.shape[1] if y_train_global.ndim > 1 else 1,
            n_reservoir=n_reservoir,
            spectral_radius=spectral_radius,
            sparsity=0.5,
            random_state=42,
            silent=True
        )

        esn.fit(X_train_global, y_train_global)
        y_pred_raw = esn.predict(X_test_global)

        # 분류 로직 (고정된 임계값 사용)
        y_pred_classified = np.zeros_like(y_pred_raw, dtype=int)
        y_pred_classified[y_pred_raw > FIXED_THRESHOLD_BUY] = 1
        y_pred_classified[y_pred_raw < FIXED_THRESHOLD_SELL] = -1

    except Exception as e:
        # print(f"Error during ESN training/prediction: {e}") # 디버깅용
        return -float('inf'),

    # 누적된 예측 신호로 평가 지표 계산
    if not y_pred_classified.size > 0:
        return -float('inf'),

    initial_capital = 10000
    capital = initial_capital
    daily_returns = []

    for i in range(len(y_pred_classified)):
        signal = y_pred_classified[i]
        true_change_rate = y_test_global[i]

        daily_return_rate = 0
        if signal == 1:
            daily_return_rate = true_change_rate * 0.01
        elif signal == -1:
            daily_return_rate = -true_change_rate * 0.01
        else:
            daily_return_rate = 0

        capital *= (1 + daily_return_rate)
        daily_returns.append(daily_return_rate)

    current_portfolio_value = initial_capital
    portfolio_values_history = [current_portfolio_value]
    for ret_rate in daily_returns:
        current_portfolio_value *= (1 + ret_rate)
        portfolio_values_history.append(current_portfolio_value)

    portfolio_values_history = np.array(portfolio_values_history)

    peak_value = portfolio_values_history[0]
    max_drawdown = 0.0
    for val in portfolio_values_history:
        if val > peak_value:
            peak_value = val
        drawdown = (peak_value - val) / peak_value
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    cumulative_return = (capital / initial_capital - 1) * 100

    std_dev_returns = np.std(daily_returns)
    avg_daily_return = np.mean(daily_returns)

    sharpe_ratio = 0.0
    if std_dev_returns != 0:
        sharpe_ratio = (avg_daily_return * np.sqrt(252)) / std_dev_returns
    elif avg_daily_return > 0:
        sharpe_ratio = 1000.0
    elif avg_daily_return < 0:
        sharpe_ratio = -1000.0

    fitness_score = sharpe_ratio * 100 + cumulative_return

    if max_drawdown > 0.5 or sharpe_ratio < -0.1:
        fitness_score = -float('inf')

    return fitness_score,


# --- 3. GA 도구 상자(Toolbox) 설정 ---
toolbox = base.Toolbox()

# 유전자(Gene) 생성 함수 등록: n_reservoir, spectral_radius (수정된 범위)
toolbox.register("attr_n_reservoir", random.randint, 500, 1000)  # 리저버 수 범위 변경
toolbox.register("attr_spectral_radius", lambda: random.uniform(0.7, 1.0))  # 스펙트럴 반경 범위 변경

# 개체(Individual) 생성 함수 등록
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_n_reservoir,
                  toolbox.attr_spectral_radius),
                 n=1)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 연산자 등록
toolbox.register("evaluate", evaluate_individual)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=[0, 0], sigma=[50, 0.05],
                 indpb=0.1)  # sigma for spectral_radius adjusted to 0.05


# --- 4. GA 실행 ---
def main():
    random.seed(42)

    pop = toolbox.population(n=30)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    print("\n--- DEAP GA 최적화 시작 (단일 학습, 고정 임계값, 조정된 범위) ---")
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=50,
                                   stats=stats, halloffame=hof, verbose=True)

    print("--- DEAP GA 최적화 완료 ---")

    return pop, log, hof


if __name__ == "__main__":
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        toolbox.register("map", pool.map)

        pop, log, hof = main()

        best_individual = hof[0]
        final_n_reservoir, final_spectral_radius = best_individual
        final_n_reservoir = int(final_n_reservoir)

        print(f"\n\n최적의 파라미터 조합 (Best Solution): {best_individual}")
        print(f"최고 Fitness (Sharpe+Return): {best_individual.fitness.values[0]:.2f}")

        print(f"\n최적 파라미터: n_reservoir={final_n_reservoir}, spectral_radius={final_spectral_radius:.2f}, "
              f"고정 임계값: Buy={FIXED_THRESHOLD_BUY:.1f}, Sell={FIXED_THRESHOLD_SELL:.1f}")

        # 최적 파라미터로 최종 시뮬레이션 및 시각화
        esn = ESN(
            n_inputs=X_train_global.shape[1],
            n_outputs=y_train_global.shape[1] if y_train_global.ndim > 1 else 1,
            n_reservoir=final_n_reservoir,
            spectral_radius=final_spectral_radius,
            sparsity=0.5,
            random_state=42,
            silent=True
        )
        esn.fit(X_train_global, y_train_global)
        y_pred_raw_final = esn.predict(X_test_global)

        y_pred_classified_final = np.zeros_like(y_pred_raw_final, dtype=int)
        y_pred_classified_final[y_pred_raw_final > FIXED_THRESHOLD_BUY] = 1
        y_pred_classified_final[y_pred_raw_final < FIXED_THRESHOLD_SELL] = -1

        # 최종 평가 지표 계산
        initial_capital = 10000
        capital = initial_capital
        daily_returns = []

        for i in range(len(y_pred_classified_final)):
            signal = y_pred_classified_final[i]
            true_change_rate = y_test_global[i]

            daily_return_rate = 0
            if signal == 1:
                daily_return_rate = true_change_rate * 0.01
            elif signal == -1:
                daily_return_rate = -true_change_rate * 0.01
            else:
                daily_return_rate = 0

            capital *= (1 + daily_return_rate)
            daily_returns.append(daily_return_rate)

        current_portfolio_value = initial_capital
        portfolio_values_history = [current_portfolio_value]
        for ret_rate in daily_returns:
            current_portfolio_value *= (1 + ret_rate)
            portfolio_values_history.append(current_portfolio_value)

        portfolio_values_history = np.array(portfolio_values_history)

        peak_value = portfolio_values_history[0]
        max_drawdown = 0.0
        for val in portfolio_values_history:
            if val > peak_value:
                peak_value = val
            drawdown = (peak_value - val) / peak_value
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        cumulative_return = (capital / initial_capital - 1) * 100
        std_dev_returns = np.std(daily_returns)
        avg_daily_return = np.mean(daily_returns)

        sharpe_ratio = 0.0
        if std_dev_returns != 0:
            sharpe_ratio = (avg_daily_return * np.sqrt(252)) / std_dev_returns
        elif avg_daily_return > 0:
            sharpe_ratio = 1000.0
        elif avg_daily_return < 0:
            sharpe_ratio = -1000.0

        print(f"최적 파라미터로 최종 시뮬레이션 결과:")
        print(f"누적 수익률: {cumulative_return:.2f}%")
        print(f"최대 낙폭 (MDD): {max_drawdown:.2f}")
        print(f"샤프 비율 (Sharpe Ratio): {sharpe_ratio:.2f}")

        # 최종 결과 DataFrame 생성 및 저장 (테스트 데이터 기간만)
        if len(y_test_global) > 0:
            results_df = pd.DataFrame({
                'True_Signal': y_test_global.flatten(),  # <-- 이 부분 수정
                'Predicted_Signal': y_pred_classified_final.flatten()  # <-- 이 부분 수정
            }, index=merged_df.index[split_index:])

            output_results_file = os.path.join(base_data_directory, ticker,
                                               f'{ticker}_esn_deap_optimized_narrow_range_predictions.csv')
            results_df.to_csv(output_results_file)
            print(f"\nDEAP GA 최적화된 예측 결과 (조정된 범위)가 '{output_results_file}'에 저장되었습니다.")

            # --- 7. 시각화 ---
            plt.figure(figsize=(18, 9))
            plt.plot(merged_df.index[split_index:], y_test_global, label='True Signal (Target - Test Set)', alpha=0.7,
                     color='blue', linewidth=1)
            plt.plot(merged_df.index[split_index:], y_pred_classified_final,
                     label='Predicted Signal (ESN - Classified - Test Set)', linestyle='--', alpha=0.7, color='red',
                     linewidth=1)

            plt.title(f'[{ticker}] ESN Model: True vs. Predicted Trading Signals (DEAP GA Optimized - Adjusted Range)')
            plt.xlabel('Date')
            plt.ylabel('Signal Value (-1:Sell, 0:Hold, 1:Buy)')
            plt.yticks([-1, 0, 1], ['Sell (-1)', 'Hold (0)', 'Buy (1)'])
            plt.ylim(-1.2, 1.2)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            output_plot_file = os.path.join(base_data_directory, ticker,
                                            f'{ticker}_esn_deap_optimized_narrow_range_signals_comparison.png')
            plt.savefig(output_plot_file)
            print(f"DEAP GA 최적화된 시그널 비교 그래프 (조정된 범위)가 '{output_plot_file}'에 저장되었습니다.")
            plt.show()

        else:
            print("\n\nDEAP GA 최적화 후 테스트를 위한 유효한 데이터가 충분하지 않아 예측을 수행할 수 없었습니다.")
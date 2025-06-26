import pandas as pd
import numpy as np
from deap import base, creator, tools, algorithms
import yfinance as yf
from numba import jit
import warnings
import optuna
from scipy.stats import mannwhitneyu
import sys  # sys 모듈 임포트

warnings.filterwarnings('ignore')

# Optuna의 불필요한 로그를 억제합니다.
optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- 공통 설정 (전역 변수) ---
TRANSACTION_COST = 0.001
NUM_OPTIMIZATION_RUNS = 1  # 각 GA 및 Optuna 최적화를 반복할 횟수 (통계 검정용)

# --- GDC 파라미터 범위 설정 ---
GDC_N1_LOW, GDC_N1_UP = 15, 50
GDC_N2_LOW, GDC_N2_UP = 75, 150
BASELINE_GDC_N1 = 20
BASELINE_GDC_N2 = 60


# --- 지표 계산기 클래스 ---
class IndicatorCalculator:
    @staticmethod
    def calculate_gdc(df, n1, n2):
        """GDC (Golden/Dead Cross) 계산"""
        temp_df = df.copy()
        close_series = temp_df['Close'].values
        temp_df['MA1'] = pd.Series(close_series).rolling(window=int(n1), min_periods=1).mean().values
        temp_df['MA2'] = pd.Series(close_series).rolling(window=int(n2), min_periods=1).mean().values
        return temp_df


# --- 백테스터 클래스 ---
class Backtester:
    @staticmethod
    def run(df_with_indicators, signals):
        """
        지표가 계산된 df와 신호에 따라 백테스트를 실행합니다.
        """
        if df_with_indicators.empty or signals.empty or len(df_with_indicators) < 2:
            return pd.Series([0.0], dtype=float), pd.Series([0.0], dtype=float)

        # 신호 DataFrame이 df_with_indicators와 동일한 인덱스를 가지도록 정렬
        signals = signals.reindex(df_with_indicators.index).fillna(False).astype(bool)

        close_prices = df_with_indicators['Close']

        position = pd.Series(0, index=df_with_indicators.index, dtype=float)
        transactions = pd.Series(0, index=df_with_indicators.index, dtype=int)

        for i in range(1, len(df_with_indicators)):
            prev_position = position.iloc[i - 1]

            if signals['Buy'].iloc[i - 1] and prev_position == 0:
                position.iloc[i] = 1
                transactions.iloc[i] = 1
            elif signals['Sell'].iloc[i - 1] and prev_position == 1:
                position.iloc[i] = 0
                transactions.iloc[i] = 1
            else:
                position.iloc[i] = prev_position

        daily_returns = close_prices.pct_change()

        strategy_returns = daily_returns * position.shift(1).fillna(0)
        strategy_returns = strategy_returns - TRANSACTION_COST * transactions

        strategy_returns = strategy_returns.dropna()

        cumulative_returns = (1 + strategy_returns).cumprod() - 1

        if cumulative_returns.empty:
            return pd.Series([0.0], dtype=float), pd.Series([0.0], dtype=float)

        return strategy_returns, cumulative_returns


# --- 성능 평가 함수 ---
def calculate_sharpe(returns, risk_free_rate=0.0):
    if returns.empty or returns.std() == 0 or not np.isfinite(returns.values).all():
        return -100.0
    annualized_return = returns.mean() * 252
    annualized_volatility = returns.std() * np.sqrt(252)
    if annualized_volatility == 0:
        return -100.0
    return (annualized_return - risk_free_rate) / annualized_volatility


def calculate_mdd(cumulative_returns):
    if cumulative_returns.empty or not np.isfinite(cumulative_returns.values).all():
        return 1.0
    return (1 - (1 + cumulative_returns) / (1 + cumulative_returns).cummax()).max()


# --- GA 최적화 클래스 (GDC 전용) ---
class GAOptimizerGDC:
    def __init__(self, df):
        self.df = df
        self._setup_deap()

    def _setup_deap(self):
        # creator가 이미 생성되었는지 확인
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "IndividualGDC"):
            creator.create("IndividualGDC", list, fitness=creator.FitnessMax,
                           sharpe=None, mdd=None, cumret=None)

        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_n1", np.random.randint, GDC_N1_LOW, GDC_N1_UP + 1)
        self.toolbox.register("attr_n2", np.random.randint, GDC_N2_LOW, GDC_N2_UP + 1)
        self.toolbox.register("individual", tools.initCycle, creator.IndividualGDC,
                              (self.toolbox.attr_n1, self.toolbox.attr_n2), n=1)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self._evaluate_gdc)
        self.toolbox.register("mate", tools.cxUniform, indpb=0.5)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def _evaluate_gdc(self, individual):
        individual.sharpe = -100.0;
        individual.mdd = 1.0;
        individual.cumret = -1.0
        try:
            gdc_n1, gdc_n2 = individual
            if not (GDC_N1_LOW <= gdc_n1 <= GDC_N1_UP and GDC_N2_LOW <= gdc_n2 <= GDC_N2_UP and gdc_n1 < gdc_n2):
                return -99999.0,

            df_calc = IndicatorCalculator.calculate_gdc(self.df.copy(), gdc_n1, gdc_n2)
            signals_df = pd.DataFrame({'Buy': df_calc['MA1'] > df_calc['MA2'], 'Sell': df_calc['MA1'] < df_calc['MA2']},
                                      index=df_calc.index)
            df_for_backtest = df_calc.dropna(subset=['MA1', 'MA2', 'Close'])
            signals_for_backtest = signals_df.reindex(df_for_backtest.index).fillna(False)

            min_len_needed = int(gdc_n2) + 2
            if len(df_for_backtest) < min_len_needed or signals_for_backtest.empty:
                return -99999.0,

            rets, cumrets = Backtester.run(df_for_backtest, signals_for_backtest)
            if rets.empty or not np.isfinite(rets.values).all() or cumrets.empty or not np.isfinite(
                    cumrets.values).all():
                return -99999.0,
            sharpe = calculate_sharpe(rets)
            mdd = calculate_mdd(cumrets)
            individual.sharpe = sharpe;
            individual.mdd = mdd;
            individual.cumret = cumrets.iloc[-1]
            fitness = sharpe / (1 + mdd)
            if not np.isfinite(fitness) or fitness < -1e5: return -99999.0,
            return fitness,
        except Exception as e:
            # print(f"GA evaluation error for {individual}: {e}", file=sys.stderr) # 디버깅용
            return -99999.0,

    def optimize(self, generations=10, pop_size=15):
        self.toolbox.register("mutate", tools.mutUniformInt,
                              low=[GDC_N1_LOW, GDC_N2_LOW],
                              up=[GDC_N1_UP, GDC_N2_UP],
                              indpb=0.2)
        pop = self.toolbox.population(n=pop_size)
        hof = tools.HallOfFame(1)
        algorithms.eaSimple(pop, self.toolbox, cxpb=0.7, mutpb=0.2,
                            ngen=generations, halloffame=hof, verbose=False)
        best = hof[0]
        _ = self._evaluate_gdc(best)
        return best


# --- Optuna 최적화 함수 (GDC 전용) ---
def objective_gdc(trial, df):
    gdc_n1 = trial.suggest_int('gdc_n1', GDC_N1_LOW, GDC_N1_UP)
    gdc_n2 = trial.suggest_int('gdc_n2', GDC_N2_LOW, GDC_N2_UP)

    if not (gdc_n1 < gdc_n2):  # n1 < n2 제약
        raise optuna.exceptions.TrialPruned()

    df_calc = IndicatorCalculator.calculate_gdc(df.copy(), gdc_n1, gdc_n2)
    signals_df = pd.DataFrame({'Buy': df_calc['MA1'] > df_calc['MA2'], 'Sell': df_calc['MA1'] < df_calc['MA2']},
                              index=df_calc.index)
    df_for_backtest = df_calc.dropna(subset=['MA1', 'MA2', 'Close'])
    signals_for_backtest = signals_df.reindex(df_for_backtest.index).fillna(False)

    min_len_needed = int(gdc_n2) + 2
    if len(df_for_backtest) < min_len_needed or signals_for_backtest.empty:
        raise optuna.exceptions.TrialPruned()

    rets, cumrets = Backtester.run(df_for_backtest, signals_for_backtest)
    if rets.empty or not np.isfinite(rets.values).all() or cumrets.empty or not np.isfinite(cumrets.values).all():
        raise optuna.exceptions.TrialPruned()

    sharpe = calculate_sharpe(rets)
    mdd = calculate_mdd(cumrets)

    # Optuna에 성능 지표 저장
    trial.set_user_attr('sharpe', sharpe)
    trial.set_user_attr('mdd', mdd)
    trial.set_user_attr('cumret', cumrets.iloc[-1])

    fitness = sharpe / (1 + mdd)
    if not np.isfinite(fitness) or fitness < -1e5:
        raise optuna.exceptions.TrialPruned()
    return fitness


# --- 메인 실행 로직 ---
def main():
    tickers = ['AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'BRK-B', 'TSLA', 'AVGO', 'GOOG']
    start_date = '2015-01-01'
    end_date = '2023-01-01'

    # 모든 결과를 저장할 딕셔너리 (이제 개별 종목별 평균은 저장하지 않음)
    # all_results = {'GDC': {'GA': {}, 'Optuna': {}}} # 사용하지 않으므로 주석 처리

    # 통계 검정을 위한 모든 종목/모든 실행의 성능 값 리스트
    all_ga_sharpes_gdc = []
    all_optuna_sharpes_gdc = []
    all_ga_mdds_gdc = []
    all_optuna_mdds_gdc = []
    all_ga_cumrets_gdc = []
    all_optuna_cumrets_gdc = []

    for ticker in tickers:
        print(f"\n======== {ticker} 데이터 처리 시작 ========")
        df = yf.download(ticker, start=start_date, end=end_date).ffill()

        # MultiIndex 컬럼 처리 및 중복 컬럼 제거
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
            df = df.loc[:, ~df.columns.duplicated()]

        # 데이터 유효성 검사 및 'Close' 컬럼 전처리
        if df.empty:
            print(f"Error: {ticker} 데이터를 다운로드할 수 없거나 비어 있습니다. 다음 종목으로 넘어갑니다.")
            continue
        if 'Close' not in df.columns:
            print(f"Error: {ticker} DataFrame에 'Close' 컬럼이 없습니다. 다음 종목으로 넘어갑니다.")
            continue
        if isinstance(df['Close'], pd.DataFrame):
            if df['Close'].shape[1] == 1:
                df['Close'] = df['Close'].iloc[:, 0]
            else:
                print(f"Warning: {ticker} 'Close' 컬럼이 여러 컬럼을 가진 DataFrame입니다. 건너뜁니다.")
                continue
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        if df['Close'].isnull().all() or len(df) < (GDC_N2_UP + 5):
            print(f"Error: {ticker} 'Close' 컬럼의 모든 값이 NaN이거나 데이터 기간이 너무 짧습니다. 다음 종목으로 넘어갑니다.")
            continue

        # --- GDC 전략 최적화 ---
        print(f"\n--- {ticker} GDC 전략 최적화 시작 ---")
        for i in range(NUM_OPTIMIZATION_RUNS):
            # GA 최적화
            optimizer_ga = GAOptimizerGDC(df)
            best_ga = optimizer_ga.optimize(generations=10, pop_size=15)  # 세대 및 인구 수 조정
            all_ga_sharpes_gdc.append(best_ga.sharpe)
            all_ga_mdds_gdc.append(best_ga.mdd)
            all_ga_cumrets_gdc.append(best_ga.cumret)

            # Optuna 최적화
            study_optuna = optuna.create_study(direction='maximize',
                                               sampler=optuna.samplers.TPESampler(seed=i))  # 시드 설정
            study_optuna.optimize(lambda trial: objective_gdc(trial, df), n_trials=150,
                                  show_progress_bar=False)  # 시도 횟수 조정
            best_optuna_trial = study_optuna.best_trial
            all_optuna_sharpes_gdc.append(best_optuna_trial.user_attrs['sharpe'])
            all_optuna_mdds_gdc.append(best_optuna_trial.user_attrs['mdd'])
            all_optuna_cumrets_gdc.append(best_optuna_trial.user_attrs['cumret'])

        print(f"======== {ticker} 데이터 처리 완료 ========")

    print("\n\n======== 모든 종목 GDC 최적화 완료! 통합 성능 비교 시작 ========")

    # 모든 종목의 모든 실행 결과를 모은 리스트로 Mann-Whitney U 검정 수행
    strategy_name = 'GDC'

    print(f"\n--- 전체 {strategy_name} 전략 성능 비교 (GA vs Optuna) ---")

    # GA 및 Optuna의 통합 평균 계산
    total_ga_avg_sharpe = np.mean(all_ga_sharpes_gdc)
    total_optuna_avg_sharpe = np.mean(all_optuna_sharpes_gdc)
    total_ga_avg_mdd = np.mean(all_ga_mdds_gdc)
    total_optuna_avg_mdd = np.mean(all_optuna_mdds_gdc)
    total_ga_avg_cumret = np.mean(all_ga_cumrets_gdc)
    total_optuna_avg_cumret = np.mean(all_optuna_cumrets_gdc)

    print(
        f"\nGA (전체 평균): Sharpe={total_ga_avg_sharpe:.2f}, MDD={total_ga_avg_mdd:.2%}, CumRet={total_ga_avg_cumret:.2%}")
    print(
        f"Optuna (전체 평균): Sharpe={total_optuna_avg_sharpe:.2f}, MDD={total_optuna_avg_mdd:.2%}, CumRet={total_optuna_avg_cumret:.2%}")

    # Mann-Whitney U 검정
    # Sharpe (높을수록 좋음)
    stat_sharpe, p_sharpe = mannwhitneyu(all_ga_sharpes_gdc, all_optuna_sharpes_gdc, alternative='two-sided')

    # MDD (낮을수록 좋음)
    stat_mdd, p_mdd = mannwhitneyu(all_ga_mdds_gdc, all_optuna_mdds_gdc, alternative='two-sided')

    # 누적 수익률 (높을수록 좋음)
    stat_cumret, p_cumret = mannwhitneyu(all_ga_cumrets_gdc, all_optuna_cumrets_gdc, alternative='two-sided')

    print("\n--- Mann-Whitney U Test p-values (alpha=0.05) ---")
    print(f"Sharpe p-value: {p_sharpe:.4f} (GA vs Optuna)")
    if p_sharpe < 0.05:
        print(f"  -> 통계적으로 유의미한 차이 (평균 Sharpe: GA={total_ga_avg_sharpe:.2f}, Optuna={total_optuna_avg_sharpe:.2f})")
        if total_optuna_avg_sharpe > total_ga_avg_sharpe:
            print("     Optuna의 Sharpe가 GA보다 통계적으로 유의미하게 높습니다.")
        else:
            print("     GA의 Sharpe가 Optuna보다 통계적으로 유의미하게 높습니다.")
    else:
        print("  -> 통계적으로 유의미한 차이 없음")

    print(f"MDD p-value: {p_mdd:.4f} (GA vs Optuna)")
    if p_mdd < 0.05:
        print(f"  -> 통계적으로 유의미한 차이 (평균 MDD: GA={total_ga_avg_mdd:.2%}, Optuna={total_optuna_avg_mdd:.2%})")
        if total_optuna_avg_mdd < total_ga_avg_mdd:  # MDD는 낮을수록 좋으므로 부등호 반대
            print("     Optuna의 MDD가 GA보다 통계적으로 유의미하게 낮습니다.")
        else:
            print("     GA의 MDD가 Optuna보다 통계적으로 유의미하게 낮습니다.")
    else:
        print("  -> 통계적으로 유의미한 차이 없음")

    print(f"Cumulative Return p-value: {p_cumret:.4f} (GA vs Optuna)")
    if p_cumret < 0.05:
        print(
            f"  -> 통계적으로 유의미한 차이 (평균 Cumulative Return: GA={total_ga_avg_cumret:.2%}, Optuna={total_optuna_avg_cumret:.2%})")
        if total_optuna_avg_cumret > total_ga_avg_cumret:
            print("     Optuna의 누적 수익률이 GA보다 통계적으로 유의미하게 높습니다.")
        else:
            print("     GA의 누적 수익률이 Optuna보다 통계적으로 유의미하게 높습니다.")
    else:
        print("  -> 통계적으로 유의미한 차이 없음")

    print("\n\n======== 모든 최적화 및 통합 통계 검정 완료! ========")


if __name__ == "__main__":
    main()
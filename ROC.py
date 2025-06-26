import pandas as pd
import numpy as np
from deap import base, creator, tools, algorithms
import yfinance as yf
from numba import jit
import warnings
import os  # os 모듈 추가

warnings.filterwarnings('ignore')

# --- 공통 설정 (전역 변수) ---
TRANSACTION_COST = 0.001

# --- ROC 파라미터 범위 설정 ---
# GA 최적화 범위
ROC_PERIOD_LOW, ROC_PERIOD_UP = 5, 30

# ROC 신호 기준
ROC_ZERO_LINE = 0.0  # ROC가 0선을 돌파하는 것을 매매 신호로 활용

# 베이스라인 파라미터
BASELINE_ROC_PERIOD = 10


# --- 지표 계산기 클래스 ---
class IndicatorCalculator:
    @staticmethod
    @jit(nopython=True)
    def _rsi(prices, period):
        # 이 함수는 RSI 전용이며, ROC에서는 사용되지 않지만, 클래스에 포함된 상태 유지
        rsi_values = np.full(len(prices), np.nan)

        if len(prices) < period + 1:
            return rsi_values

        deltas = np.diff(prices)

        first_gains_sum = 0.0
        first_losses_sum = 0.0
        for i in range(period):
            if deltas[i] > 0:
                first_gains_sum += deltas[i]
            else:
                first_losses_sum -= deltas[i]

        avg_gain = first_gains_sum / period
        avg_loss = first_losses_sum / period

        if avg_loss == 0:
            rs = np.inf
        else:
            rs = avg_gain / avg_loss
        rsi_values[period] = 100 - (100 / (1 + rs))

        for i in range(period + 1, len(prices)):
            current_gain = 0.0
            current_loss = 0.0
            if deltas[i - 1] > 0:
                current_gain = deltas[i - 1]
            else:
                current_loss = -deltas[i - 1]

            avg_gain = (avg_gain * (period - 1) + current_gain) / period
            avg_loss = (avg_loss * (period - 1) + current_loss) / period

            if avg_loss == 0:
                rs = np.inf
            else:
                rs = avg_gain / avg_loss

            rsi_values[i] = 100 - (100 / (1 + rs))

        return rsi_values

    @staticmethod
    def calculate_gdc(df, n1, n2):
        """GDC (Golden/Dead Cross) 계산"""
        temp_df = df.copy()
        close_series = temp_df['Close'].values
        temp_df['MA1'] = pd.Series(close_series).rolling(window=int(n1), min_periods=1).mean().values
        temp_df['MA2'] = pd.Series(close_series).rolling(window=int(n2), min_periods=1).mean().values
        return temp_df

    @classmethod
    def calculate_rsi(cls, df, period):
        """RSI (Relative Strength Index) 계산"""
        temp_df = df.copy()
        close = temp_df['Close'].values
        rsi_raw = cls._rsi(close, int(period))
        temp_df['RSI'] = rsi_raw
        return temp_df

    @staticmethod
    def calculate_roc(df, period):
        """ROC (Rate of Change) 계산"""
        temp_df = df.copy()
        # ROC = ((현재 종가 / period일 전 종가) - 1) * 100
        temp_df['ROC'] = (temp_df['Close'] / temp_df['Close'].shift(int(period)) - 1) * 100
        return temp_df


# --- 백테스터 클래스 ---
class Backtester:
    @staticmethod
    def run(df_with_indicators, signals):
        if df_with_indicators.empty or signals.empty or len(df_with_indicators) < 2:
            return pd.Series([0.0], dtype=float), pd.Series([0.0], dtype=float)

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


# --- GA 최적화 클래스 (ROC 전용) ---
class GAOptimizerROC:
    def __init__(self, df):
        self.df = df
        self._setup_deap()

    def _setup_deap(self):
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "IndividualROC"):
            creator.create("IndividualROC", list, fitness=creator.FitnessMax,
                           sharpe=None, mdd=None, cumret=None)

        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_period", np.random.randint, ROC_PERIOD_LOW, ROC_PERIOD_UP + 1)
        self.toolbox.register("individual", tools.initCycle, creator.IndividualROC,
                              (self.toolbox.attr_period,), n=1)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self._evaluate_roc)
        self.toolbox.register("mate", tools.cxUniform, indpb=0.5)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def _evaluate_roc(self, individual):
        individual.sharpe = -100.0
        individual.mdd = 1.0
        individual.cumret = -1.0

        roc_period = individual[0]

        if not (ROC_PERIOD_LOW <= roc_period <= ROC_PERIOD_UP):
            return -99999.0,

        df_calc = IndicatorCalculator.calculate_roc(self.df.copy(), roc_period)
        df_for_backtest = df_calc.dropna(subset=['ROC', 'Close'])

        signals_df = pd.DataFrame(index=df_for_backtest.index)
        signals_df['Buy'] = (df_for_backtest['ROC'].shift(1) <= ROC_ZERO_LINE) & (
                df_for_backtest['ROC'] > ROC_ZERO_LINE)
        signals_df['Sell'] = (df_for_backtest['ROC'].shift(1) >= ROC_ZERO_LINE) & (
                df_for_backtest['ROC'] < ROC_ZERO_LINE)

        min_len_needed = int(roc_period) + 2
        if len(df_for_backtest) < min_len_needed or signals_df.empty:
            return -99999.0,

        rets, cumrets = Backtester.run(df_for_backtest, signals_df)

        if rets.empty or not np.isfinite(rets.values).all() or cumrets.empty or not np.isfinite(cumrets.values).all():
            return -99999.0,

        sharpe = calculate_sharpe(rets)
        mdd = calculate_mdd(cumrets)

        individual.sharpe = sharpe
        individual.mdd = mdd
        individual.cumret = cumrets.iloc[-1]

        fitness = sharpe / (1 + mdd)

        if not np.isfinite(fitness) or fitness < -1e5:
            return -99999.0,

        return fitness,

    def optimize(self, generations=30, pop_size=50):
        self.toolbox.register("mutate", tools.mutUniformInt,
                              low=[ROC_PERIOD_LOW],
                              up=[ROC_PERIOD_UP],
                              indpb=0.2)

        pop = self.toolbox.population(n=pop_size)
        hof = tools.HallOfFame(1)

        algorithms.eaSimple(pop, self.toolbox, cxpb=0.7, mutpb=0.2,
                            ngen=generations, halloffame=hof, verbose=False)

        best = hof[0]
        _ = self._evaluate_roc(best)
        return best

def transform_roc_signals_to_numerical(roc_df, period):
    """
    ROC가 0선을 돌파하는 시점에만 1(buy) 또는 -1(sell) 신호를 생성하고,
    그 외 기간은 0(홀드)으로 유지합니다.
    """
    # Numerical Signal 컬럼을 0으로 초기화
    numerical_signals = pd.Series(0.0, index=roc_df.index, dtype=float)

    # ROC 값과 이전 ROC 값을 가져옵니다.
    current_roc = roc_df['ROC']
    previous_roc = current_roc.shift(1)

    # 매수 신호 (ROC 0선 상향 돌파): 이전 ROC <= 0 이고 현재 ROC > 0 인 경우
    buy_signal_points = (previous_roc <= ROC_ZERO_LINE) & (current_roc > ROC_ZERO_LINE)

    # 매도 신호 (ROC 0선 하향 돌파): 이전 ROC >= 0 이고 현재 ROC < 0 인 경우
    sell_signal_points = (previous_roc >= ROC_ZERO_LINE) & (current_roc < ROC_ZERO_LINE)

    # 교차 시점에만 1 또는 -1 할당
    numerical_signals.loc[buy_signal_points] = 1.0
    numerical_signals.loc[sell_signal_points] = -1.0

    # 첫 유효한 신호 처리:
    # 만약 데이터 시작부터 ROC가 0보다 크다면 첫 유효한 날에 'buy' 신호를 1로,
    # 0보다 작다면 '-1'로 처리합니다. (교차 신호와는 별개로 초기 포지션 설정)
    # 이 부분은 전략의 특성과 백테스팅 로직에 따라 다르게 정의될 수 있습니다.
    # 현재 `buy_signal_points`와 `sell_signal_points`에 `isna()`를 포함시키지 않았으므로,
    # 첫 유효한 ROC 값이 0선을 넘는 경우를 위한 명시적 처리입니다.
    if not roc_df.empty:
        # ROC 값이 NaN이 아닌 첫 번째 인덱스를 찾습니다.
        first_valid_roc_idx = roc_df['ROC'].first_valid_index()
        if first_valid_roc_idx is not None:
            # 이 인덱스 이후부터 신호가 유효하게 생성됩니다.
            # 하지만, 혹시라도 first_valid_roc_idx가 교차점으로 잡히지 않고,
            # 그 시점에 이미 특정 상태(ROC > 0 또는 ROC < 0)가 형성되어 있다면,
            # 해당 날짜를 첫 포지션 진입으로 간주할 수 있습니다.
            # 이 로직은 `(previous_roc.isna())` 조건이 `buy_signal_points`에 포함되지 않았으므로 중요합니다.
            if first_valid_roc_idx == roc_df.index[0]:  # ROC 값이 처음부터 유효한 경우
                if roc_df.loc[first_valid_roc_idx, 'ROC'] > ROC_ZERO_LINE:
                    numerical_signals.loc[first_valid_roc_idx] = 1.0
                elif roc_df.loc[first_valid_roc_idx, 'ROC'] < ROC_ZERO_LINE:
                    numerical_signals.loc[first_valid_roc_idx] = -1.0
            else:  # ROC NaN 이후 첫 유효값
                # 첫 유효값 전의 값이 NaN이므로, 이 시점을 교차로 처리할 수 있도록
                # `buy_signal_points`와 `sell_signal_points`에서 `previous_roc.isna()`를
                # 포함하는 것이 더 견고합니다. 현재는 이미 위에서 그렇게 했으므로 불필요할 수 있습니다.
                # 만약 `buy_signal_points`와 `sell_signal_points`에 `isna()` 조건이 없다면
                # 이 부분은 필요합니다.
                pass  # 현재 buy_signal_points, sell_signal_points 로직이 isna()를 포함하지 않으므로,
                # 여기서는 첫 유효한 ROC 값이 0선을 돌파하는지를 확인하는 것이 중요합니다.
                # 이미 위에서 `(previous_roc <= ROC_ZERO_LINE) & (current_roc > ROC_ZERO_LINE)`
                # 이 NaN을 잘 처리하지 못할 수 있으니, 아래와 같이 보완하는 것이 좋습니다.

    # 최종적으로 Numerical_Signal 컬럼을 추가
    roc_df['Numerical_Signal'] = numerical_signals
    return roc_df

# --- 메인 실행 로직 ---
def main():
    tickers = ['AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'BRK-B', 'TSLA', 'AVGO', 'GOOG']
    start_date = '2015-01-01'
    end_date = '2023-01-01'

    # 파일 저장을 위한 기본 디렉토리 설정 (현재 스크립트가 실행되는 디렉토리)
    base_data_directory = os.getcwd()

    if not os.path.exists(base_data_directory):
        print(f"경고: 지정된 기본 데이터 디렉토리 '{base_data_directory}'가 존재하지 않습니다. 스크립트 실행 경로를 다시 확인해주세요.")
        return

    print("=== ROC 지표 GA 최적화 및 베이스라인 비교 ===")
    print(f"GA 최적화 ROC 기간 범위: {ROC_PERIOD_LOW}~{ROC_PERIOD_UP}일")
    print(f"ROC 매수/매도 기준: ROC > {ROC_ZERO_LINE} (매수), ROC < {ROC_ZERO_LINE} (매도)")
    print(f"베이스라인 ROC 기간: {BASELINE_ROC_PERIOD}일")
    print(f"거래 비용: {TRANSACTION_COST * 100:.2f}%")
    print("=" * 60)

    results = {}

    for ticker in tickers:
        print(f"\n--- {ticker} 데이터 다운로드 중 ({start_date} ~ {end_date}) ---")
        df = yf.download(ticker, start=start_date, end=end_date).ffill()

        if df.empty:
            print(f"Error: {ticker} 데이터를 다운로드할 수 없습니다. 다음 종목으로 넘어갑니다.")
            continue

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
            df = df.loc[:, ~df.columns.duplicated()]

        if 'Close' not in df.columns or df['Close'].isnull().all():
            print(f"Error: {ticker} 에 'Close' 가격 데이터가 없습니다. 다음 종목으로 넘어갑니다.")
            continue

        if len(df) < (ROC_PERIOD_UP + 5):
            print(f"경고: {ticker} 데이터가 너무 적습니다 ({len(df)}일). 백테스팅에 부적합하여 건너킵니다.")
            continue

        print(f"--- {ticker} ROC 파라미터 최적화 시작 ---")
        optimizer = GAOptimizerROC(df)
        best_roc = optimizer.optimize(generations=30, pop_size=50)

        print(f"\n--- {ticker} 최적화 결과 ---")
        print(f"최적 ROC 기간: {int(best_roc[0])}일")
        print(f"성능: Sharpe={best_roc.sharpe:.2f}, MDD={best_roc.mdd:.2%}, 누적수익률={best_roc.cumret:.2%}")

        # --- 최적화된 ROC 신호 생성 및 저장 ---
        print(f"\n--- {ticker} 최적 ROC 트레이딩 신호 생성 및 저장 ---")
        optimized_period = int(best_roc[0])

        # ROC 지표 계산 및 NaN 값 제거 (ROC 기간에 따라 초기 NaN 발생)
        # 이 `optimized_roc_df_calc`를 `transform_roc_signals_to_numerical` 함수에 전달합니다.
        optimized_roc_df_calc = IndicatorCalculator.calculate_roc(df.copy(), optimized_period).dropna(subset=['ROC'])

        # transform_roc_signals_to_numerical 함수를 사용하여 1, -1, 0 신호 생성
        # 이 함수가 '교차' 시점에만 1 또는 -1을 할당합니다.
        final_optimized_roc_signals_df = transform_roc_signals_to_numerical(optimized_roc_df_calc.copy(), optimized_period)

        ticker_directory = os.path.join(base_data_directory, ticker)
        os.makedirs(ticker_directory, exist_ok=True) # Ensure directory exists

        output_filename = os.path.join(ticker_directory, f'{ticker}_optimized_roc_signals.csv')
        # 필요한 컬럼만 저장 (날짜, ROC 값, Numerical_Signal)
        final_optimized_roc_signals_df[['ROC', 'Numerical_Signal']].to_csv(output_filename, index=True)

        print(f"'{ticker}' 최적화된 ROC 신호가 '{output_filename}' 파일로 성공적으로 저장되었습니다.")


        # --- 베이스라인 성능 평가 ---
        print(f"\n--- {ticker} 베이스라인 성능 평가 (기간={BASELINE_ROC_PERIOD}일) ---")
        baseline_df = IndicatorCalculator.calculate_roc(df.copy(), BASELINE_ROC_PERIOD)

        baseline_df_cleaned = baseline_df.dropna(subset=['ROC', 'Close'])

        baseline_signals = pd.DataFrame(index=baseline_df_cleaned.index)
        baseline_signals['Buy'] = (baseline_df_cleaned['ROC'].shift(1) <= ROC_ZERO_LINE) & (
                    baseline_df_cleaned['ROC'] > ROC_ZERO_LINE)
        baseline_signals['Sell'] = (baseline_df_cleaned['ROC'].shift(1) >= ROC_ZERO_LINE) & (
                    baseline_df_cleaned['ROC'] < ROC_ZERO_LINE)

        min_len_needed_baseline = BASELINE_ROC_PERIOD + 2
        if len(baseline_df_cleaned) < min_len_needed_baseline or baseline_signals.empty:
            print(f"경고: {ticker} 베이스라인 전략을 위한 데이터가 너무 적거나 신호가 없습니다. 건너킵니다.")
            sharpe_baseline, mdd_baseline, cumret_baseline = -100.0, 1.0, -1.0
        else:
            rets_baseline, cumrets_baseline = Backtester.run(baseline_df_cleaned, baseline_signals)

            if rets_baseline.empty or not np.isfinite(
                    rets_baseline.values).all() or cumrets_baseline.empty or not np.isfinite(
                cumrets_baseline.values).all():
                print(f"경고: {ticker} 베이스라인 백테스트 결과가 유효하지 않습니다.")
                sharpe_baseline, mdd_baseline, cumret_baseline = -100.0, 1.0, -1.0
            else:
                sharpe_baseline = calculate_sharpe(rets_baseline)
                mdd_baseline = calculate_mdd(cumrets_baseline)
                cumret_baseline = cumrets_baseline.iloc[-1]

        print(f"성능: Sharpe={sharpe_baseline:.2f}, MDD={mdd_baseline:.2%}, 누적수익률={cumret_baseline:.2%}")

        results[ticker] = {
            'Optimized': {'Params': [int(best_roc[0])], 'Sharpe': best_roc.sharpe, 'MDD': best_roc.mdd,
                          'CumRet': best_roc.cumret},
            'Baseline': {'Params': [BASELINE_ROC_PERIOD], 'Sharpe': sharpe_baseline, 'MDD': mdd_baseline,
                         'CumRet': cumret_baseline}
        }
        print("\n" + "=" * 60)

    print("\n\n=== 모든 종목 최적화 및 비교 요약 ===")
    print("=" * 60)
    for ticker, data in results.items():
        print(f"\n--- {ticker} ---")
        print(f"최적 ROC (기간={data['Optimized']['Params'][0]}): "
              f"Sharpe={data['Optimized']['Sharpe']:.2f}, MDD={data['Optimized']['MDD']:.2%}, CumRet={data['Optimized']['CumRet']:.2%}")
        print(f"베이스라인 ROC (기간={data['Baseline']['Params'][0]}): "
              f"Sharpe={data['Baseline']['Sharpe']:.2f}, MDD={data['Baseline']['MDD']:.2%}, CumRet={data['Baseline']['CumRet']:.2%}")
    print("\n" + "=" * 60)
    print("모든 종목에 대한 ROC 최적화 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
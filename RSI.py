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

# --- RSI 파라미터 범위 설정 (기울어진 선 제외) ---
# GA 최적화 대상 파라미터
RSI_PERIOD_LOW, RSI_PERIOD_UP = 10, 20  # RSI 기간 범위
RSI_UPPER_BOUND_LOW, RSI_UPPER_BOUND_UP = 65, 85  # 과매수 기준선 범위
RSI_LOWER_BOUND_LOW, RSI_LOWER_BOUND_UP = 15, 35  # 과매도 기준선 범위

# 베이스라인 파라미터 (최적화되지 않은 고정값)
BASELINE_RSI_PERIOD = 14
BASELINE_RSI_UPPER = 70
BASELINE_RSI_LOWER = 30


# --- 지표 계산기 클래스 ---
class IndicatorCalculator:
    @staticmethod
    @jit(nopython=True)
    def _rsi(prices, period):
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

    @classmethod
    def calculate_rsi(cls, df, period):
        temp_df = df.copy()
        close = temp_df['Close'].values
        rsi_raw = cls._rsi(close, int(period))
        temp_df['RSI'] = rsi_raw
        return temp_df


# --- 백테스터 클래스 (순수 RSI 과매수/과매도 돌파 로직) ---
class Backtester:
    @staticmethod
    def run_simple_rsi(df_with_rsi, rsi_period, rsi_upper, rsi_lower):
        if df_with_rsi.empty:
            return pd.Series([0.0], dtype=float), pd.Series([0.0], dtype=float)

        close_prices = df_with_rsi['Close']
        rsi_values = df_with_rsi['RSI']

        position = pd.Series(0, index=df_with_rsi.index, dtype=float)
        transactions = pd.Series(0, index=df_with_rsi.index, dtype=int)

        start_idx = rsi_values.first_valid_index()
        if start_idx is None:
            return pd.Series([0.0], dtype=float), pd.Series([0.0], dtype=float)
        start_loc = df_with_rsi.index.get_loc(start_idx)

        for i in range(start_loc, len(df_with_rsi)):
            current_rsi = rsi_values.iloc[i]
            prev_rsi = rsi_values.iloc[i - 1] if i > 0 else np.nan
            prev_position = position.iloc[i - 1] if i > 0 else 0

            if pd.isna(current_rsi) or pd.isna(prev_rsi):
                position.iloc[i] = prev_position
                continue

            # 매수 신호: RSI가 과매도 영역 (rsi_lower)을 상향 돌파
            if prev_rsi <= rsi_lower and current_rsi > rsi_lower:
                if prev_position == 0:  # 현재 포지션이 없으면 매수
                    position.iloc[i] = 1
                    transactions.iloc[i] = 1
            # 매도 신호: RSI가 과매수 영역 (rsi_upper)을 하향 돌파
            elif prev_rsi >= rsi_upper and current_rsi < rsi_upper:
                if prev_position == 1:  # 현재 매수 포지션이 있으면 매도 (청산)
                    position.iloc[i] = 0
                    transactions.iloc[i] = 1
            else:
                position.iloc[i] = prev_position  # 이전 포지션 유지

        daily_returns = close_prices.pct_change()
        strategy_returns = daily_returns * position.shift(1).fillna(0)
        strategy_returns = strategy_returns - TRANSACTION_COST * transactions

        strategy_returns = strategy_returns.dropna()

        cumulative_returns = (1 + strategy_returns).cumprod() - 1

        if cumulative_returns.empty:
            return pd.Series([0.0], dtype=float), pd.Series([0.0], dtype=float)

        return strategy_returns, cumulative_returns

    @staticmethod
    def run_baseline_rsi(df_with_rsi, rsi_period, rsi_upper, rsi_lower):
        return Backtester.run_simple_rsi(df_with_rsi, rsi_period, rsi_upper, rsi_lower)


# --- 성능 평가 함수 (기존과 동일) ---
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


# --- GA 최적화 클래스 (순수 RSI 전용) ---
class GAOptimizerSimpleRSI:
    def __init__(self, df):
        self.df = df
        self._setup_deap()

    def _setup_deap(self):
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "IndividualSimpleRSI"):
            creator.create("IndividualSimpleRSI", list, fitness=creator.FitnessMax,
                           sharpe=None, mdd=None, cumret=None)

        self.toolbox = base.Toolbox()

        self.toolbox.register("attr_rsi_period", np.random.randint, RSI_PERIOD_LOW, RSI_PERIOD_UP + 1)
        self.toolbox.register("attr_rsi_upper", np.random.randint, RSI_UPPER_BOUND_LOW, RSI_UPPER_BOUND_UP + 1)
        self.toolbox.register("attr_rsi_lower", np.random.randint, RSI_LOWER_BOUND_LOW, RSI_LOWER_BOUND_UP + 1)

        self.toolbox.register("individual", tools.initCycle, creator.IndividualSimpleRSI,
                              (self.toolbox.attr_rsi_period, self.toolbox.attr_rsi_upper,
                               self.toolbox.attr_rsi_lower), n=1)

        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self._evaluate_simple_rsi)
        self.toolbox.register("mate", tools.cxUniform, indpb=0.5)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def _evaluate_simple_rsi(self, individual):
        individual.sharpe = -100.0
        individual.mdd = 1.0
        individual.cumret = -1.0

        try:
            rsi_period, rsi_upper, rsi_lower = individual

            if not (RSI_PERIOD_LOW <= rsi_period <= RSI_PERIOD_UP and
                    RSI_UPPER_BOUND_LOW <= rsi_upper <= RSI_UPPER_BOUND_UP and
                    RSI_LOWER_BOUND_LOW <= rsi_lower <= RSI_LOWER_BOUND_UP and
                    rsi_upper > rsi_lower):
                return -99999.0,

            df_calc = IndicatorCalculator.calculate_rsi(self.df.copy(), rsi_period)

            df_for_backtest = df_calc.dropna(subset=['RSI', 'Close'])

            min_len_needed = int(rsi_period) + 2
            if len(df_for_backtest) < min_len_needed:
                return -99999.0,

            rets, cumrets = Backtester.run_simple_rsi(df_for_backtest, rsi_period, rsi_upper, rsi_lower)

            if rets.empty or not np.isfinite(rets.values).all() or cumrets.empty or not np.isfinite(
                    cumrets.values).all():
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

        except Exception as e:
            return -99999.0,

    def optimize(self, generations=30, pop_size=50):
        self.toolbox.register("mutate", tools.mutUniformInt,
                              low=[RSI_PERIOD_LOW, RSI_UPPER_BOUND_LOW, RSI_LOWER_BOUND_LOW],
                              up=[RSI_PERIOD_UP, RSI_UPPER_BOUND_UP, RSI_LOWER_BOUND_UP],
                              indpb=0.2)

        pop = self.toolbox.population(n=pop_size)
        hof = tools.HallOfFame(1)

        algorithms.eaSimple(pop, self.toolbox, cxpb=0.7, mutpb=0.2,
                            ngen=generations, halloffame=hof, verbose=False)

        best = hof[0]
        _ = self._evaluate_simple_rsi(best)
        return best

## RSI 신호 변환 함수 추가

def transform_rsi_signals_to_numerical(rsi_df, rsi_upper, rsi_lower):
    """
    RSI 과매수/과매도 기준선 돌파 시점에만 1(buy) 또는 -1(sell) 신호를 생성하고,
    그 외 기간은 0(홀드)으로 유지합니다.
    """
    numerical_signals = pd.Series(0.0, index=rsi_df.index, dtype=float)

    current_rsi = rsi_df['RSI']
    previous_rsi = current_rsi.shift(1)

    # 매수 신호: 이전 RSI가 과매도 기준선 이하이고 현재 RSI가 과매도 기준선을 상향 돌파
    # 또는 데이터 시작점에서 RSI가 과매도 기준선 아래에서 시작하여 바로 위로 올라오는 경우
    buy_signal_points = ((previous_rsi <= rsi_lower) | previous_rsi.isna()) & (current_rsi > rsi_lower)

    # 매도 신호: 이전 RSI가 과매수 기준선 이상이고 현재 RSI가 과매수 기준선을 하향 돌파
    # 또는 데이터 시작점에서 RSI가 과매수 기준선 위에서 시작하여 바로 아래로 내려오는 경우
    sell_signal_points = ((previous_rsi >= rsi_upper) | previous_rsi.isna()) & (current_rsi < rsi_upper)

    # 교차 시점에만 1 또는 -1 할당
    numerical_signals.loc[buy_signal_points] = 1.0
    numerical_signals.loc[sell_signal_points] = -1.0

    # 첫 유효한 RSI 값에 대한 초기 포지션 설정 로직 (선택 사항)
    # 만약 첫 번째 유효한 RSI 값이 이미 과매수/과매도 영역 밖에 있다면,
    # 해당 날짜를 첫 진입 신호로 간주할 수 있습니다.
    # 그러나 보통은 '교차'만을 신호로 삼으므로, 이 부분은 주석 처리합니다.
    # 만약 꼭 필요하다면, 백테스팅 로직의 position 초기화와 일관되게 구현해야 합니다.
    # if not rsi_df.empty:
    #     first_valid_rsi_idx = rsi_df['RSI'].first_valid_index()
    #     if first_valid_rsi_idx is not None:
    #         if numerical_signals.loc[first_valid_rsi_idx] == 0: # 아직 신호가 없다면
    #             if rsi_df.loc[first_valid_rsi_idx, 'RSI'] > rsi_upper:
    #                 numerical_signals.loc[first_valid_rsi_idx] = -1.0 # 과매수 상태면 매도 포지션으로 시작
    #             elif rsi_df.loc[first_valid_rsi_idx, 'RSI'] < rsi_lower:
    #                 numerical_signals.loc[first_valid_rsi_idx] = 1.0 # 과매도 상태면 매수 포지션으로 시작

    rsi_df['Numerical_Signal'] = numerical_signals
    return rsi_df

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

    print("=== 순수 RSI 과매수/과매도 돌파 GA 최적화 및 베이스라인 비교 ===")
    print(f"GA 최적화 RSI 기간 범위: {RSI_PERIOD_LOW}~{RSI_PERIOD_UP}일")
    print(f"GA 최적화 과매수 기준 범위: {RSI_UPPER_BOUND_LOW}~{RSI_UPPER_BOUND_UP}")
    print(f"GA 최적화 과매도 기준 범위: {RSI_LOWER_BOUND_LOW}~{RSI_LOWER_BOUND_UP}")
    print(f"베이스라인 RSI 기간: {BASELINE_RSI_PERIOD}일, 과매수={BASELINE_RSI_UPPER}, 과매도={BASELINE_RSI_LOWER}")
    print("="*60)

    results = {}

    for ticker in tickers:
        print(f"\n--- {ticker} 데이터 다운로드 중 ({start_date} ~ {end_date}) ---")
        df = yf.download(ticker, start=start_date, end=end_date).ffill()

        if df.empty:
            print(f"Error: {ticker} 데이터를 다운로드할 수 없습니다. 다음 종목으로 넘어갑니다.")
            continue

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
            df = df.loc[:,~df.columns.duplicated()]

        if 'Close' not in df.columns or df['Close'].isnull().all():
            print(f"Error: {ticker} 에 'Close' 가격 데이터가 없습니다. 다음 종목으로 넘어갑니다.")
            continue

        if len(df) < (RSI_PERIOD_UP + 5):
            print(f"경고: {ticker} 데이터가 너무 적습니다 ({len(df)}일). 백테스팅에 부적합하여 건너킵니다.")
            continue


        print(f"--- {ticker} 순수 RSI 파라미터 최적화 시작 ---")
        optimizer = GAOptimizerSimpleRSI(df)
        best_params = optimizer.optimize(generations=40, pop_size=50)

        best_rsi_period, best_rsi_upper, best_rsi_lower = best_params

        print(f"\n--- {ticker} 최적화 결과 ---")
        print(f"최적 파라미터: RSI Period={int(best_rsi_period)}, Upper={int(best_rsi_upper)}, Lower={int(best_rsi_lower)}")
        print(f"성능: Sharpe={best_params.sharpe:.2f}, MDD={best_params.mdd:.2%}, 누적수익률={best_params.cumret:.2%}")

        # --- 최적화된 RSI 신호 생성 및 저장 ---
        print(f"\n--- {ticker} 최적 RSI 트레이딩 신호 생성 및 저장 ---")
        # 최적화된 파라미터로 RSI 지표 다시 계산
        # RSI 계산 후 NaN이 있는 행을 제거하여 유효한 데이터만 사용합니다.
        optimized_rsi_df_calc = IndicatorCalculator.calculate_rsi(df.copy(), best_rsi_period).dropna(subset=['RSI'])

        # transform_rsi_signals_to_numerical 함수를 사용하여 1, -1, 0 신호 생성
        final_optimized_rsi_signals_df = transform_rsi_signals_to_numerical(
            optimized_rsi_df_calc.copy(), # .copy()를 사용하여 원본 DataFrame에 영향X
            best_rsi_upper,
            best_rsi_lower
        )

        ticker_directory = os.path.join(base_data_directory, ticker)
        os.makedirs(ticker_directory, exist_ok=True) # 폴더가 없으면 생성

        output_filename = os.path.join(ticker_directory, f'{ticker}_optimized_rsi_signals.csv')
        # RSI 값과 Numerical_Signal 컬럼만 CSV로 저장
        final_optimized_rsi_signals_df[['RSI', 'Numerical_Signal']].to_csv(output_filename, index=True)

        print(f"'{ticker}' 최적화된 RSI 신호가 '{output_filename}' 파일로 성공적으로 저장되었습니다.")


        # --- 베이스라인 성능 평가 (고정 파라미터 사용) ---
        print(f"\n--- {ticker} 베이스라인 성능 평가 (RSI 기간={BASELINE_RSI_PERIOD}, 과매수={BASELINE_RSI_UPPER}, 과매도={BASELINE_RSI_LOWER}) ---")
        baseline_df = IndicatorCalculator.calculate_rsi(df.copy(), BASELINE_RSI_PERIOD)

        baseline_df_cleaned = baseline_df.dropna(subset=['RSI', 'Close'])

        min_len_needed_baseline = BASELINE_RSI_PERIOD + 2
        if len(baseline_df_cleaned) < min_len_needed_baseline:
            print(f"경고: {ticker} 베이스라인 전략을 위한 데이터가 너무 적습니다. 건너킵니다.")
            sharpe_baseline, mdd_baseline, cumret_baseline = -100.0, 1.0, -1.0
        else:
            rets_baseline, cumrets_baseline = Backtester.run_simple_rsi(baseline_df_cleaned,
                                                                        BASELINE_RSI_PERIOD,
                                                                        BASELINE_RSI_UPPER,
                                                                        BASELINE_RSI_LOWER)

            if rets_baseline.empty or not np.isfinite(rets_baseline.values).all() or cumrets_baseline.empty or not np.isfinite(cumrets_baseline.values).all():
                 print(f"경고: {ticker} 베이스라인 백테스트 결과가 유효하지 않습니다.")
                 sharpe_baseline, mdd_baseline, cumret_baseline = -100.0, 1.0, -1.0
            else:
                sharpe_baseline = calculate_sharpe(rets_baseline)
                mdd_baseline = calculate_mdd(cumrets_baseline)
                cumret_baseline = cumrets_baseline.iloc[-1]

        print(f"성능: Sharpe={sharpe_baseline:.2f}, MDD={mdd_baseline:.2%}, 누적수익률={cumret_baseline:.2%}")

        results[ticker] = {
            'Optimized': {'Params': [int(best_rsi_period), int(best_rsi_upper), int(best_rsi_lower)],
                          'Sharpe': best_params.sharpe, 'MDD': best_params.mdd, 'CumRet': best_params.cumret},
            'Baseline': {'Params': [BASELINE_RSI_PERIOD, BASELINE_RSI_UPPER, BASELINE_RSI_LOWER],
                         'Sharpe': sharpe_baseline, 'MDD': mdd_baseline, 'CumRet': cumret_baseline}
        }
        print("\n" + "="*60)

    print("\n\n=== 모든 종목 최적화 및 비교 요약 ===")
    print("="*60)
    for ticker, data in results.items():
        print(f"\n--- {ticker} ---")
        print(f"최적 RSI (Period={data['Optimized']['Params'][0]}, Upper={data['Optimized']['Params'][1]}, Lower={data['Optimized']['Params'][2]}): "
              f"Sharpe={data['Optimized']['Sharpe']:.2f}, MDD={data['Optimized']['MDD']:.2%}, CumRet={data['Optimized']['CumRet']:.2%}")
        print(f"베이스라인 RSI (Period={data['Baseline']['Params'][0]}, Upper={data['Baseline']['Params'][1]}, Lower={data['Baseline']['Params'][2]}): "
              f"Sharpe={data['Baseline']['Sharpe']:.2f}, MDD={data['Baseline']['MDD']:.2%}, CumRet={data['Baseline']['CumRet']:.2%}")
    print("\n" + "="*60)
    print("모든 종목에 대한 순수 RSI 최적화 완료!")
    print("="*60)

if __name__ == "__main__":
    main()
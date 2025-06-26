import pandas as pd
import numpy as np
from deap import base, creator, tools, algorithms
import yfinance as yf
import warnings
import os

warnings.filterwarnings('ignore')

# --- 공통 설정 (전역 변수) ---
TRANSACTION_COST = 0.001

# --- GDC 파라미터 범위 설정 ---
# GA 최적화 범위
GDC_N1_LOW, GDC_N1_UP = 15, 50
GDC_N2_LOW, GDC_N2_UP = 75, 150

# 베이스라인 파라미터
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
        if df_with_indicators.empty or signals.empty:
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
        individual.sharpe = -100.0
        individual.mdd = 1.0
        individual.cumret = -1.0

        try:
            gdc_n1, gdc_n2 = individual

            if not (GDC_N1_LOW <= gdc_n1 <= GDC_N1_UP and
                    GDC_N2_LOW <= gdc_n2 <= GDC_N2_UP and
                    gdc_n1 < gdc_n2):
                return -99999.0,

            df_calc = IndicatorCalculator.calculate_gdc(self.df.copy(), gdc_n1, gdc_n2)

            signals_df = pd.DataFrame({
                'Buy': df_calc['MA1'] > df_calc['MA2'],
                'Sell': df_calc['MA1'] < df_calc['MA2']
            }, index=df_calc.index)

            df_for_backtest = df_calc.dropna(subset=['MA1', 'MA2', 'Close'])
            signals_for_backtest = signals_df.reindex(df_for_backtest.index).fillna(False)

            min_len_needed = int(gdc_n2) + 2
            if len(df_for_backtest) < min_len_needed:
                return -99999.0,

            rets, cumrets = Backtester.run(df_for_backtest, signals_for_backtest)

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


def transform_and_interpolate_signals(signals_df):
    """
    MA 교차 시점에만 1(buy) 또는 -1(sell) 신호를 생성하고, 그 외 기간은 0(홀드)으로 유지합니다.
    데이터 초반의 첫 유효한 MA 값들을 기반으로 초기 신호를 설정합니다.
    """
    numerical_signals = pd.Series(0.0, index=signals_df.index, dtype=float)

    # signals_df는 'Signal' 컬럼에 'buy', 'sell' 또는 None을 포함한다고 가정
    # 이 signals_df는 이미 MA 계산이 완료되고 NaN이 제거된 DataFrame을 기반으로 합니다.

    current_ma_state = signals_df['Signal']
    previous_ma_state = current_ma_state.shift(1)

    # 골든 크로스: 이전 상태가 'sell' 이었거나 (데이터 시작점이라 이전 상태가 없는) `NaN`인데, 현재 'buy'인 경우
    # `previous_ma_state.isna()`는 데이터의 첫 번째 행에서만 True가 될 가능성이 높습니다.
    # 또한, `current_ma_state == 'buy'`가 True인 경우에만 `1.0`을 할당합니다.
    buy_cross = (current_ma_state == 'buy') & ((previous_ma_state == 'sell') | (previous_ma_state.isna()))

    # 데드 크로스: 이전 상태가 'buy' 이었거나 (데이터 시작점이라 이전 상태가 없는) `NaN`인데, 현재 'sell'인 경우
    sell_cross = (current_ma_state == 'sell') & ((previous_ma_state == 'buy') | (previous_ma_state.isna()))

    # 교차 시점에만 신호 할당
    # 신호가 중복될 경우(예: buy_cross와 sell_cross가 동시에 True가 되는 경우)를 방지하기 위해 순서가 중요합니다.
    # 일반적으로 GDC에서는 동시에 발생하지 않으므로 큰 문제는 없지만, 명확하게 할당합니다.
    numerical_signals.loc[buy_cross] = 1.0
    numerical_signals.loc[sell_cross] = -1.0

    # 이제 signals_df에 Numerical_Signal 컬럼 추가
    signals_df['Numerical_Signal'] = numerical_signals
    return signals_df
# --- 메인 실행 로직 (변경 없음) ---
def main():
    tickers = ['AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'BRK-B', 'TSLA', 'AVGO', 'GOOG']
    start_date = '2015-01-01'
    end_date = '2023-01-01'

    base_data_directory = '.'

    if not os.path.exists(base_data_directory):
        print(f"경고: 지정된 기본 데이터 디렉토리 '{base_data_directory}'가 존재하지 않습니다. 스크립트 실행 경로를 다시 확인해주세요.")
        return

    print("=== GDC 지표 GA 최적화 및 베이스라인 비교 ===")
    print(f"GA 최적화 GDC MA1 범위: {GDC_N1_LOW}~{GDC_N1_UP}, MA2 범위: {GDC_N2_LOW}~{GDC_N2_UP}")
    print(f"베이스라인 GDC 파라미터: MA1={BASELINE_GDC_N1}, MA2={BASELINE_GDC_N2}")
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

        print(f"--- {ticker} GDC 파라미터 최적화 시작 ---")
        optimizer = GAOptimizerGDC(df)
        best_gdc = optimizer.optimize(generations=40, pop_size=50)

        print(f"\n--- {ticker} 최적화 결과 ---")
        print(f"최적 GDC 파라미터: MA1={int(best_gdc[0])}, MA2={int(best_gdc[1])}")
        print(f"성능: Sharpe={best_gdc.sharpe:.2f}, MDD={best_gdc.mdd:.2%}, 누적수익률={best_gdc.cumret:.2%}")

        # --- 최적화된 GDC 신호 생성 및 저장 ---
        print(f"\n--- {ticker} 최적 GDC 트레이딩 신호 생성 및 저장 ---")
        optimized_n1, optimized_n2 = int(best_gdc[0]), int(best_gdc[1])

        optimized_df_calc = IndicatorCalculator.calculate_gdc(df.copy(), optimized_n1, optimized_n2)

        signals_for_interpolation_df = pd.DataFrame(index=optimized_df_calc.index)
        signals_for_interpolation_df['Signal'] = None

        signals_for_interpolation_df.loc[optimized_df_calc['MA1'] > optimized_df_calc['MA2'], 'Signal'] = 'buy'
        signals_for_interpolation_df.loc[optimized_df_calc['MA1'] < optimized_df_calc['MA2'], 'Signal'] = 'sell'

        final_optimized_signals_df = transform_and_interpolate_signals(signals_for_interpolation_df)

        ticker_directory = os.path.join(base_data_directory, ticker)
        os.makedirs(ticker_directory, exist_ok=True)

        output_filename = os.path.join(ticker_directory, f'{ticker}_optimized_gdc_signals.csv')
        final_optimized_signals_df.to_csv(output_filename, index=True)

        print(f"'{ticker}' 최적화된 GDC 신호가 '{output_filename}' 파일로 성공적으로 저장되었습니다.")

        # --- 베이스라인 성능 평가 ---
        print(f"\n--- {ticker} 베이스라인 성능 평가 (MA1={BASELINE_GDC_N1}, MA2={BASELINE_GDC_N2}) ---")
        baseline_df = IndicatorCalculator.calculate_gdc(df.copy(), BASELINE_GDC_N1, BASELINE_GDC_N2)

        baseline_signals = pd.DataFrame({
            'Buy': baseline_df['MA1'] > baseline_df['MA2'],
            'Sell': baseline_df['MA1'] < baseline_df['MA2']
        }, index=baseline_df.index)

        baseline_df_cleaned = baseline_df.dropna(subset=['MA1', 'MA2', 'Close'])
        baseline_signals_cleaned = baseline_signals.reindex(baseline_df_cleaned.index).fillna(False)

        min_len_needed_baseline = BASELINE_GDC_N2 + 2
        if len(baseline_df_cleaned) < min_len_needed_baseline:
            print(f"경고: {ticker} 베이스라인 전략을 위한 데이터가 너무 적습니다. 건너뜁니다.")
            sharpe_baseline, mdd_baseline, cumret_baseline = -100.0, 1.0, -1.0
        else:
            rets_baseline, cumrets_baseline = Backtester.run(baseline_df_cleaned, baseline_signals_cleaned)

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
            'Optimized': {'Params': [optimized_n1, optimized_n2], 'Sharpe': best_gdc.sharpe,
                          'MDD': best_gdc.mdd, 'CumRet': best_gdc.cumret},
            'Baseline': {'Params': [BASELINE_GDC_N1, BASELINE_GDC_N2], 'Sharpe': sharpe_baseline, 'MDD': mdd_baseline,
                         'CumRet': cumret_baseline}
        }
        print("\n" + "=" * 60)

    print("\n\n=== 모든 종목 최적화 및 비교 요약 ===")
    print("=" * 60)
    for ticker, data in results.items():
        print(f"\n--- {ticker} ---")
        print(f"최적 GDC ({data['Optimized']['Params'][0]}, {data['Optimized']['Params'][1]}): "
              f"Sharpe={data['Optimized']['Sharpe']:.2f}, MDD={data['Optimized']['MDD']:.2%}, CumRet={data['Optimized']['CumRet']:.2%}")
        print(f"베이스라인 GDC ({data['Baseline']['Params'][0]}, {data['Baseline']['Params'][1]}): "
              f"Sharpe={data['Baseline']['Sharpe']:.2f}, MDD={data['Baseline']['MDD']:.2%}, CumRet={data['Baseline']['CumRet']:.2%}")
    print("\n" + "=" * 60)
    print("모든 종목에 대한 GDC 최적화 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
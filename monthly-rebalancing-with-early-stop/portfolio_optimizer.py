import numpy as np
from AlgorithmImports import *

class PortfolioOptimizer:
    def __init__(self, algorithm: QCAlgorithm):
        self.algorithm = algorithm
        self.config = algorithm.config

    def optimize_portfolio(self, selected_symbols):
        short_lookback = self.config.short_lookback
        returns = self.algorithm.history(selected_symbols, short_lookback, Resolution.DAILY)['close'].unstack(level=0).pct_change().dropna()
        n_assets = len(selected_symbols)
        n_portfolios = self.config.n_portfolios

        results = np.zeros((3, n_portfolios))
        weights_record = []

        for i in range(n_portfolios):
            weights = np.random.random(n_assets)
            weights /= np.sum(weights)

            portfolio_return = np.sum(returns.mean() * weights) * short_lookback
            portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * short_lookback, weights)))

            downside_stddev = np.sqrt(np.mean(np.minimum(0, returns).apply(lambda x: x**2, axis=0).dot(weights)))
            sortino_ratio = portfolio_return / downside_stddev

            results[0,i] = portfolio_return
            results[1,i] = portfolio_stddev
            results[2,i] = sortino_ratio

            weights_record.append(weights)

        best_sortino_idx = np.argmax(results[2])
        return weights_record[best_sortino_idx]

    def adjust_portfolio(self, target_weights):
        current_date = self.algorithm.time.strftime('%Y-%m-%d %H:%M:%S')
        self.algorithm.debug(f"{current_date}: Adjusting portfolio")
        
        current_symbols = set(self.algorithm.portfolio.keys)
        target_symbols = set(target_weights.keys())

        removed_symbols = current_symbols - target_symbols
        for symbol in removed_symbols:
            self.algorithm.liquidate(symbol)

        for symbol, target_weight in target_weights.items():
            current_weight = self.algorithm.portfolio[symbol].holdings_value / self.algorithm.portfolio.total_portfolio_value if symbol in self.algorithm.portfolio else 0
            adjusted_weight = current_weight * (1 - self.config.adjustment_step) + target_weight * self.config.adjustment_step
            self.algorithm.set_holdings(symbol, adjusted_weight)

        self.log_holdings()

    def log_holdings(self):
        holdings = {}
        sum_of_all_holdings = 0
        for symbol in self.algorithm.portfolio.keys:
            holding_percentage = self.algorithm.portfolio[symbol].holdings_value / self.algorithm.portfolio.total_portfolio_value * 100
            if holding_percentage > 1e-4:
                sum_of_all_holdings += holding_percentage
                holdings[symbol.id.to_string().split(" ")[0]] = round(holding_percentage, 2)
        current_date = self.algorithm.time.strftime('%Y-%m-%d %H:%M:%S')
        self.algorithm.debug(f"{current_date}: Updated holdings [{sum_of_all_holdings:.2f}%]:")
        for symbol, percentage in holdings.items():
            self.algorithm.debug(f"  {symbol}: {percentage}%")
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
        n = self.config.simulation_count

        weights = np.random.random((n, n_assets))
        weights /= weights.sum(axis=1)[:, np.newaxis]

        mean_returns = returns.mean().values
        # cov_matrix = returns.cov().values * short_lookback

        portfolio_returns = np.dot(weights, mean_returns) * short_lookback
        # portfolio_stddevs = np.sqrt(np.einsum('ij,jk,ik->i', weights, cov_matrix, weights))

        downside_returns = np.minimum(0, returns.values)
        downside_stddevs = np.sqrt(np.mean(np.square(downside_returns).dot(weights.T), axis=0))
        sortino_ratios = portfolio_returns / downside_stddevs

        best_idx = np.argmax(sortino_ratios)
        return weights[best_idx]

    def adjust_portfolio(self, target_weights):
        current_date = self.algorithm.time.strftime('%Y-%m-%d %H:%M:%S')
        self.algorithm.debug(f"{current_date}: Adjusting portfolio")
        
        current_symbols = set(self.algorithm.portfolio.keys())
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
        for symbol in self.algorithm.portfolio.keys():
            holding_percentage = self.algorithm.portfolio[symbol].holdings_value / self.algorithm.portfolio.total_portfolio_value * 100
            if holding_percentage > 1e-4:
                sum_of_all_holdings += holding_percentage
                holdings[symbol.id.to_string().split(" ")[0]] = round(holding_percentage, 2)
        current_date = self.algorithm.time.strftime('%Y-%m-%d %H:%M:%S')
        self.algorithm.debug(f"{current_date}: Updated holdings [{sum_of_all_holdings:.2f}%]:")
        for symbol, percentage in holdings.items():
            self.algorithm.debug(f"  {symbol}: {percentage}%")
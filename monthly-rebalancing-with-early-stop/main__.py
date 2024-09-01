#region imports
from AlgorithmImports import *
import numpy as np
from collections import deque
import statsmodels.api as sm
import statistics as stat
import pickle
#endregion

class MonthlyRebalancingWithEarlyStop(QCAlgorithm):
    def initialize(self):
        self.set_start_date(2019, 3, 1)   # Set Start Date
        self.set_end_date(2024, 8, 1)     # Set End Date
        self.initial_cash = 1000000
        self.set_cash(self.initial_cash)  # Set Strategy Cash
        self.set_security_initializer(BrokerageModelSecurityInitializer(
            self.BrokerageModel, FuncSecuritySeeder(self.GetLastKnownPrices)
        ))

        self.p_lookback = 252
        self.p_num_coarse = 200
        self.p_num_fine = 70
        self.p_num_long = 5
        self.p_adjustment_step = 1.0
        self.p_n_portfolios = 1000
        self.p_short_lookback = 63
        self.p_rand_seed = 13
        self.p_adjustment_frequency = 'monthly'  # Can be 'monthly', 'weekly', 'bi-weekly'
        
        # Change resolution to minute
        self.universe_settings.resolution = Resolution.DAILY
        # self.set_benchmark(self.add_equity('SPY').symbol)

        self._momp = {}          # Dict of Momentum indicator keyed by Symbol
        self._lookback = self.p_lookback     # Momentum indicator lookback period
        self._num_coarse = self.p_num_coarse # Number of symbols selected at Coarse Selection
        self._num_fine = self.p_num_fine     # Number of symbols selected at Fine Selection
        self._num_long = self.p_num_long     # Number of symbols with open positions

        self._rebalance = False
        self.current_holdings = set()  # To track current holdings

        self.target_weights = {}  # To store target weights
        self.adjustment_step = self.p_adjustment_step  # Adjustment step for gradual transition
        self._short_lookback = self.p_short_lookback

        self.first_trade_date = None
        self.next_adjustment_date = None

        # Metrics for no trades and profit tracking
        self.no_trade_days = 0
        self.highest_profit = 0
        self.lowest_profit = float('inf')
        self.monthly_starting_equity = 0
        self.last_logged_month = None  # 用于记录上次输出日志的月份
        self.global_stop_loss_triggered = False  # 标志是否已经触发了全局止损
        self.halved_lookback = False  # Track whether the lookback has been halved
        self.add_universe(self._coarse_selection_function, self._fine_selection_function)

    def _coarse_selection_function(self, coarse):
        '''Drop securities which have no fundamental data or have too low prices.
        Select those with highest by dollar volume'''
        if self.next_adjustment_date and self.time < self.next_adjustment_date:
            return Universe.UNCHANGED

        self._rebalance = True

        if not self.first_trade_date:
            self.first_trade_date = self.time
            self.next_adjustment_date = self.get_next_adjustment_date(self.time)
            self._rebalance = True

        selected = sorted([x for x in coarse if x.has_fundamental_data and x.price > 5],
            key=lambda x: x.dollar_volume, reverse=True)

        return [x.symbol for x in selected[:self._num_coarse]]

    def _fine_selection_function(self, fine):
        '''Select security with highest market cap'''
        selected = sorted(fine, key=lambda f: f.market_cap, reverse=True)
        return [x.symbol for x in selected[:self._num_fine]]
    
    def on_data(self, data):
        for symbol, mom in self._momp.items():
            mom.update(self.time, self.securities[symbol].close)

        if self.monthly_starting_equity == 0:
            self.monthly_starting_equity = self.Portfolio.TotalPortfolioValue

        current_portfolio_value = self.Portfolio.TotalPortfolioValue
        if self.monthly_starting_equity != 0:
            current_profit_pct_to_start = ((current_portfolio_value - self.monthly_starting_equity) / self.monthly_starting_equity) * 100
        else:
            current_profit_pct_to_start = 0

        self.highest_profit = max(self.highest_profit, current_profit_pct_to_start)

        if self.highest_profit != 0:
            drop_pct = ((self.highest_profit - current_profit_pct_to_start) / self.highest_profit) * 100
        else:
            drop_pct = 0

        if current_profit_pct_to_start <= -12 and not self.global_stop_loss_triggered:
            current_date = self.Time.strftime('%Y-%m-%d %H:%M:%S')
            self.debug(f"{current_date}: Liquidating all holdings due to a portfolio loss of {current_profit_pct_to_start:.2f}% (stop-loss from last adjustment).")
            self.Liquidate()
            self._rebalance = False  # Allow immediate rebalancing
            self.global_stop_loss_triggered = True
            self.highest_profit = 0
            self.monthly_starting_equity = 0
            self.next_adjustment_date = self.get_next_adjustment_date(self.time)

            # if not self.halved_lookback:
            #     self._lookback //= 2  # Halve the lookback period
            #     self.halved_lookback = True  # Mark that the lookback has been halved
            # else:
            #     self.debug(f"{current_date}: Stopping trading temporarily due to repeated stop-loss trigger.")
            self.debug(f"{current_date}: Stopping trading temporarily due to stop-loss trigger.")
            return

        if self.highest_profit > 10 and drop_pct >= 10:
            current_date = self.Time.strftime('%Y-%m-%d %H:%M:%S')
            self.debug(f"{current_date}: Liquidating all holdings due to a {drop_pct:.2f}% drop in profit (take-profit).")
            self.debug(f"{current_date}: Highest Net Profit: {self.highest_profit:.2f}% (from last adjustment)")
            self.debug(f"{current_date}: Current Net Profit: {current_profit_pct_to_start:.2f}% (from last adjustment)")
            total_profit_pct = ((current_portfolio_value - self.initial_cash) / self.initial_cash) * 100
            self.debug(f"{current_date}: Total Net Profit: {total_profit_pct:.2f}% (from inception)")
            self.Liquidate()
            self._rebalance = True  # Allow immediate rebalancing
            self.global_stop_loss_triggered = True
            self.highest_profit = 0
            self.monthly_starting_equity = 0
            self.next_adjustment_date = self.get_next_adjustment_date(self.time)

            if not self.halved_lookback:
                # self._lookback //= 2  # Halve the lookback period
                self._short_lookback //= 7
                self.halved_lookback = True  # Mark that the lookback has been halved
            return

        if self.Time.day == 1 and (self.Time.month != self.last_logged_month):
            current_date = self.Time.strftime('%Y-%m-%d %H:%M:%S')
            portfolio_value = self.Portfolio.TotalPortfolioValue
            net_profit = portfolio_value - self.initial_cash
            holdings_value = sum([sec.HoldingsValue for sec in self.Portfolio.Values if sec.Invested])
            unrealized_profit = self.Portfolio.TotalUnrealizedProfit
            return_pct = (net_profit / self.initial_cash) * 100
            self.debug(f"{current_date}: Equity: ${portfolio_value:.2f} | Holdings: ${holdings_value:.2f} | Net Profit: ${net_profit:.2f} | Unrealized: ${unrealized_profit:.2f} | Return: {return_pct:.2f}%")
            self.last_logged_month = self.Time.month

            if self.halved_lookback:
                self._lookback = self.p_lookback  # Restore the original lookback period
                self._short_lookback = self.p_short_lookback  # Restore the original short lookback period
                self.halved_lookback = False  # Reset the halved lookback flag

        if not self._rebalance:
            return

        if self._rebalance:
            self.global_stop_loss_triggered = False
            self._rebalance = False

        sorted_mom = sorted([k for k,v in self._momp.items() if v.is_ready],
            key=lambda x: self._momp[x].current.value, reverse=True)
        selected = sorted_mom[:self._num_long]
        new_holdings = set(selected)

        if new_holdings != self.current_holdings or self.first_trade_date == self.time:
            if len(selected) > 0:
                optimal_weights = self.optimize_portfolio(selected)
                self.target_weights = dict(zip(selected, optimal_weights))
                self.current_holdings = new_holdings
                self.adjust_portfolio()

        self._rebalance = False
        self.next_adjustment_date = self.get_next_adjustment_date(self.time)



    def on_securities_changed(self, changes):
        # Clean up data for removed securities and Liquidate
        for security in changes.RemovedSecurities:
            symbol = security.Symbol
            if self._momp.pop(symbol, None) is not None:
                self.Liquidate(symbol, 'Removed from universe')

        for security in changes.AddedSecurities:
            if security.Symbol not in self._momp:
                self._momp[security.Symbol] = MomentumPercent(self._lookback)

        # Warm up the indicator with history price if it is not ready
        added_symbols = [k for k, v in self._momp.items() if not v.IsReady]
        history = self.History(added_symbols, 1 + self._lookback, Resolution.DAILY)
        history = history.close.unstack(level=0)

        for symbol in added_symbols:
            ticker = symbol.ID.ToString()
            if ticker in history:
                for time, value in history[ticker].dropna().items():
                    item = IndicatorDataPoint(symbol, time, value)
                    self._momp[symbol].Update(item)

    def optimize_portfolio(self, selected_symbols):
        short_lookback = self._short_lookback
        returns = self.History(selected_symbols, short_lookback, Resolution.DAILY)['close'].unstack(level=0).pct_change().dropna()
        n_assets = len(selected_symbols)
        n_portfolios = self.p_n_portfolios

        results = np.zeros((3, n_portfolios))
        weights_record = []

        np.random.seed(self.p_rand_seed)

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

    def adjust_portfolio(self):
        current_symbols = set(self.Portfolio.Keys)
        target_symbols = set(self.target_weights.keys())

        # Liquidate removed symbols
        removed_symbols = current_symbols - target_symbols
        for symbol in removed_symbols:
            self.Liquidate(symbol)

        # Adjust holdings for selected symbols
        for symbol, target_weight in self.target_weights.items():
            current_weight = self.Portfolio[symbol].Quantity / self.Portfolio.TotalPortfolioValue if symbol in self.Portfolio else 0
            adjusted_weight = current_weight * (1 - self.adjustment_step) + target_weight * self.adjustment_step
            self.SetHoldings(symbol, adjusted_weight)
        holdings = {}
        sum_of_all_holdings = 0
        for symbol in self.portfolio.keys():
            holding_percentage = self.Portfolio[symbol].HoldingsValue / self.Portfolio.TotalPortfolioValue * 100
            if holding_percentage > 1e-4:
                sum_of_all_holdings += holding_percentage
                holdings[symbol.ID.to_string().split(" ")[0]] = round(holding_percentage, 2)
        self.debug(f"Final holdings [{sum_of_all_holdings:.2f}%]: {holdings}")

    def get_next_adjustment_date(self, current_date, initial=False):
        if self.p_adjustment_frequency == 'weekly':
            return current_date + timedelta(days=7)
        elif self.p_adjustment_frequency == 'bi-weekly':
            return current_date + timedelta(days=14)
        elif self.p_adjustment_frequency == 'monthly':
            if initial:
                next_month = current_date.replace(day=1) + timedelta(days=32)
                return next_month.replace(day=1)
            next_month = current_date.replace(day=1) + timedelta(days=32)
            return next_month.replace(day=1)
        else:
            raise ValueError(f"Unsupported adjustment frequency: {self.p_adjustment_frequency}")

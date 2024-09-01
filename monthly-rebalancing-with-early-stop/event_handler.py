from datetime import timedelta
from AlgorithmImports import *

class EventHandler:
    def __init__(self, algorithm: QCAlgorithm):
        self.algorithm = algorithm
        self.config = algorithm.config
        self.highest_profit = 0
        self.monthly_starting_equity = 0
        self.last_logged_month = None
        self.global_stop_loss_triggered = False
        self.halved_lookback = False

    def handle_data(self, data):
        self.update_metrics()
        self.check_stop_loss()
        self.check_take_profit()
        self.log_monthly_performance()

        if not self.algorithm._rebalance:
            return

        if self.algorithm._rebalance:
            current_date = self.algorithm.time.strftime('%Y-%m-%d %H:%M:%S')
            self.algorithm.debug(f"{current_date}: Rebalancing portfolio")

        selected = self.algorithm.stock_selector.get_selected_symbols()
        new_holdings = set(selected)

        if new_holdings != self.algorithm.current_holdings or self.algorithm.first_trade_date == self.algorithm.time:
            if len(selected) > 0:
                optimal_weights = self.algorithm.portfolio_optimizer.optimize_portfolio(selected)
                self.algorithm.target_weights = dict(zip(selected, optimal_weights))
                self.algorithm.current_holdings = new_holdings
                self.algorithm.portfolio_optimizer.adjust_portfolio(self.algorithm.target_weights)

        self.algorithm._rebalance = False
        self.algorithm.next_adjustment_date = self.get_next_adjustment_date(self.algorithm.time)

    def update_metrics(self):
        if self.monthly_starting_equity == 0:
            self.monthly_starting_equity = self.algorithm.Portfolio.TotalPortfolioValue

        current_portfolio_value = self.algorithm.Portfolio.TotalPortfolioValue
        if self.monthly_starting_equity != 0:
            self.current_profit_pct_to_start = ((current_portfolio_value - self.monthly_starting_equity) / self.monthly_starting_equity) * 100
        else:
            self.current_profit_pct_to_start = 0

        self.highest_profit = max(self.highest_profit, self.current_profit_pct_to_start)

        if self.highest_profit != 0:
            self.drop_pct = ((self.highest_profit - self.current_profit_pct_to_start) / self.highest_profit) * 100
        else:
            self.drop_pct = 0

    def check_stop_loss(self):
        if self.current_profit_pct_to_start <= self.config.stop_loss_threshold and not self.global_stop_loss_triggered:
            current_date = self.algorithm.time.strftime('%Y-%m-%d %H:%M:%S')
            self.algorithm.debug(f"{current_date}: Stop-Loss Triggered")
            self.algorithm.debug(f"  Liquidating all holdings due to a portfolio loss of {self.current_profit_pct_to_start:.2f}%")
            self.algorithm.Liquidate()
            self.algorithm._rebalance = False
            self.global_stop_loss_triggered = True
            self.highest_profit = 0
            self.monthly_starting_equity = 0
            self.algorithm.next_adjustment_date = self.get_next_adjustment_date(self.algorithm.time)
            self.algorithm.debug(f"{current_date}: Stopping trading temporarily due to stop-loss trigger.")

    def check_take_profit(self):
        if self.highest_profit > self.config.take_profit_threshold and self.drop_pct >= self.config.take_profit_threshold:
            current_date = self.algorithm.time.strftime('%Y-%m-%d %H:%M:%S')
            self.algorithm.debug(f"{current_date}: Take-Profit Triggered")
            self.algorithm.debug(f"  Liquidating all holdings due to a {self.drop_pct:.2f}% drop in profit")
            self.algorithm.debug(f"  Highest Net Profit: {self.highest_profit:.2f}% (from last adjustment)")
            self.algorithm.debug(f"  Current Net Profit: {self.current_profit_pct_to_start:.2f}% (from last adjustment)")
            total_profit_pct = ((self.algorithm.Portfolio.TotalPortfolioValue - self.config.initial_cash) / self.config.initial_cash) * 100
            self.algorithm.debug(f"  Total Net Profit: {total_profit_pct:.2f}% (from inception)")
            self.algorithm.liquidate()
            self.algorithm._rebalance = True
            self.global_stop_loss_triggered = True
            self.highest_profit = 0
            self.monthly_starting_equity = 0
            self.algorithm.next_adjustment_date = self.get_next_adjustment_date(self.algorithm.time)

            if not self.halved_lookback:
                self.algorithm.stock_selector._short_lookback //= 7
                self.halved_lookback = True

    def log_monthly_performance(self):
        current_date = self.algorithm.Time
        if current_date.day == 1 and (current_date.month != self.last_logged_month):
            formatted_date = current_date.strftime('%Y-%m-%d %H:%M:%S')
            portfolio_value = self.algorithm.Portfolio.TotalPortfolioValue
            net_profit = portfolio_value - self.config.initial_cash
            holdings_value = sum([sec.HoldingsValue for sec in self.algorithm.Portfolio.Values if sec.Invested])
            unrealized_profit = self.algorithm.Portfolio.TotalUnrealizedProfit
            return_pct = (net_profit / self.config.initial_cash) * 100
            self.algorithm.debug(f"{formatted_date}: Monthly Performance Summary")
            self.algorithm.debug(f"  Equity: ${portfolio_value:.2f}")
            self.algorithm.debug(f"  Holdings: ${holdings_value:.2f}")
            self.algorithm.debug(f"  Net Profit: ${net_profit:.2f}")
            self.algorithm.debug(f"  Unrealized Profit: ${unrealized_profit:.2f}")
            self.algorithm.debug(f"  Return: {return_pct:.2f}%")
            self.last_logged_month = current_date.month

            if self.halved_lookback:
                self.algorithm.stock_selector._lookback = self.config.lookback
                self.algorithm.stock_selector._short_lookback = self.config.short_lookback
                self.halved_lookback = False

    def get_next_adjustment_date(self, current_date, initial=False):
        if self.config.adjustment_frequency == 'weekly':
            return self.algorithm.trading_calendar.add_days(current_date, 7)
        elif self.config.adjustment_frequency == 'bi-weekly':
            return self.algorithm.trading_calendar.add_days(current_date, 14)
        elif self.config.adjustment_frequency == 'monthly':
            if initial:
                next_month = current_date.replace(day=1) + timedelta(days=32)
                next_month = next_month.replace(day=1)
            else:
                next_month = current_date.replace(day=1) + timedelta(days=32)
                next_month = next_month.replace(day=1)
            return self.algorithm.trading_calendar.add_days(next_month, 0)  # This ensures we get the next trading day
        else:
            raise ValueError(f"Unsupported adjustment frequency: {self.config.adjustment_frequency}")
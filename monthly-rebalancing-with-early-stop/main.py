#region imports
from AlgorithmImports import *
import numpy as np
from collections import deque
import statistics as stat
import pickle
import random
### modules
from config import Config
from stock_selector import StockSelector
from portfolio_optimizer import PortfolioOptimizer
from event_handler import EventHandler
#endregion

class MonthlyRebalancingWithEarlyStop(QCAlgorithm):
    def initialize(self):
        self.config = Config()
        self.set_start_date(*self.config.start_date)
        self.set_end_date(*self.config.end_date)
        self.set_cash(self.config.initial_cash)
        self.set_security_initializer(BrokerageModelSecurityInitializer(
            self.brokerage_model, FuncSecuritySeeder(self.get_last_known_prices)
        ))

        self.universe_settings.resolution = Resolution.DAILY

        self.stock_selector = StockSelector(self)
        self.portfolio_optimizer = PortfolioOptimizer(self)
        self.event_handler = EventHandler(self)

        self.current_holdings = set()
        self.target_weights = {}
        self.first_trade_date = None
        self.next_adjustment_date = None

        np.random.seed(self.config.rand_seed)
        random.seed(self.config.rand_seed)

        self.add_universe(self.stock_selector.coarse_selection, self.stock_selector.fine_selection)

    def on_data(self, data):
        self.stock_selector.update_momentum(data)
        self.event_handler.handle_data(data)

    def on_securities_changed(self, changes):
        self.stock_selector.handle_securities_changed(changes)
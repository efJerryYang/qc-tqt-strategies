from AlgorithmImports import *

class StockSelector:
    def __init__(self, algorithm: QCAlgorithm):
        self.algorithm = algorithm
        self.config = algorithm.config
        self._momp = {}
        self._lookback = self.config.momentum_lookback
        self._num_coarse = self.config.num_coarse
        self._num_fine = self.config.num_fine
        self._num_long = self.config.num_long

    def coarse_selection(self, coarse: List[CoarseFundamental]):
        if self.algorithm.next_adjustment_date and self.algorithm.time < self.algorithm.next_adjustment_date:
            return Universe.UNCHANGED

        self.algorithm._rebalance = True

        if not self.algorithm.first_trade_date:
            self.algorithm.first_trade_date = self.algorithm.time
            self.algorithm.next_adjustment_date = self.algorithm.event_handler.get_next_adjustment_date(self.algorithm.time)
            self.algorithm._rebalance = True

        selected = sorted([x for x in coarse if x.has_fundamental_data and x.price > 5],
            key=lambda x: x.dollar_volume, reverse=True)

        return [x.symbol for x in selected[:self._num_coarse]]

    def fine_selection(self, fine: List[FineFundamental]):
        selected = sorted(fine, key=lambda f: f.market_cap, reverse=True)
        return [x.symbol for x in selected[:self._num_fine]]

    def handle_securities_changed(self, changes: SecurityChanges):
        for security in changes.removed_securities:
            symbol = security.symbol
            if self._momp.pop(symbol, None) is not None:
                self.algorithm.liquidate(symbol, 'Removed from universe')

        for security in changes.added_securities:
            if security.symbol not in self._momp:
                self._momp[security.symbol] = MomentumPercent(self._lookback)

        added_symbols = [k for k, v in self._momp.items() if not v.is_ready]
        history = self.algorithm.history(added_symbols, 1 + self._lookback, Resolution.DAILY)
        history = history.close.unstack(level=0)

        for symbol in added_symbols:
            ticker = symbol.id.to_string()
            if ticker in history:
                for time, value in history[ticker].dropna().items():
                    item = IndicatorDataPoint(symbol, time, value)
                    self._momp[symbol].update(item)

    def get_selected_symbols(self) -> List[Symbol]:
        sorted_mom = sorted([k for k,v in self._momp.items() if v.is_ready],
            key=lambda x: self._momp[x].current.value, reverse=True)
        return sorted_mom[:self._num_long]

    def update_momentum(self, data: List[IndicatorDataPoint]):
        for symbol, mom in self._momp.items():
            mom.update(self.algorithm.time, self.algorithm.securities[symbol].close)
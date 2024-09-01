class Config:
    def __init__(self):
        self.start_date = (2019, 3, 1)
        self.end_date = (2024, 8, 1)
        self.initial_cash = 1000000
        self.momentum_lookback = 252
        self.num_coarse = 200
        self.num_fine = 70
        self.num_long = 5
        self.adjustment_step = 1.0
        self.simulation_count = 1000
        self.short_lookback = 63
        self.rand_seed = 114514
        self.adjustment_frequency = 'monthly'
        self.take_profit_threshold = 10
        self.stop_loss_threshold = -12
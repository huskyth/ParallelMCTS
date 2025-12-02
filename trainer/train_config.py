class TrainConfig:
    def __init__(self):
        self.epoch = 10000000
        self.test_rate = 10
        self.greedy_times = 5
        self.dirichlet_rate = 1 - 0.25
        self.dirichlet_probability = 0.3

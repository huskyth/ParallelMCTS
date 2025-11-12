class TrainConfig:
    def __init__(self):
        self.epoch = 100
        self.test_rate = 5
        self.greedy_times = 5
        self.dirichlet_rate = 1 - 0.25
        self.dirichlet_probability = 0.3

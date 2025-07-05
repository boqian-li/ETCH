import math

class Summary():
    def __init__(self, momentum=0.9):
        self.items = []
        self.running_stats = {}
        self.counters = {}
        self.momentum = momentum

    def register(self, keys):
        for k in keys:
            self.items.append(k)
            self.running_stats[k] = 0.0
            self.counters[k] = 0

    def update(self, stats):
        for k, v in stats.items():
            if self.counters[k] == 0:
                self.running_stats[k] = v
            else:
                self.running_stats[k] = self.momentum * self.running_stats[k] + (1-self.momentum) * v
            self.counters[k] += 1

    def get_item(self, k):
        return self.running_stats[k]

    def get(self):
        return '\t'.join(f'{k}: {self.get_item(k):.4f}' for k in self.items)
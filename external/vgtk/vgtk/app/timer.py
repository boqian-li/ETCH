import time

class Timer():
	def __init__(self):
		self._time = time.time()
		self._checkpoints = {}

	def set_point(self, pid='default'):
		self._checkpoints[pid] = time.time()

	def get_point(self, pid='default'):
		return time.time() - self._checkpoints[pid]

	def reset_point(self, pid='default'):
		_time = time.time() - self._checkpoints[pid]
		self._checkpoints[pid] = time.time()
		return _time
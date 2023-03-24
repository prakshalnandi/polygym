import numpy as np
from collections import namedtuple

Sequence = namedtuple(
    "Sequence", ("states", "actions", "next_states", "done")
)


class Buffer:
    def __init__(self, size: int):
        
        self.size = int(size)
        self.records = None
        self.entries = 0

    def initialize(self, sequence: Sequence):
        
        self.records = Sequence(*[np.zeros([self.size, s.size], dtype=s.dtype) for s in sequence])

    def add(self, *args):
        
        if not self.records:
            self.initialize(Sequence(*args))

        location = (self.entries) % self.size
        for i, entry in enumerate(args):
            self.records[i][location, :] = entry

        self.entries = self.entries + 1

from collections import namedtuple
from hashlib import sha256
from json import JSONEncoder
from sys import version
from typing import TypeAlias

import numpy as np
from numpy import double
from numpy.random import MT19937, Generator

# seed = int(datetime.now().timestamp() * 1000)
seed = sha256(version.encode()).digest()
seed = int.from_bytes(seed, 'big')
mt = MT19937(seed)
rand = Generator(mt)


# 座標
Coord = namedtuple("Coord", ["x", "y"], defaults=(double(0), double(0)))


class AdvancedJSONEncoder(JSONEncoder):
    def default(self, obj):
        if hasattr(obj, '__json__'):
            return obj.__json__()
        # print
        if isinstance(obj, np.ndarray):
            return obj.tolist()

        if isinstance(obj, double):
            return str(obj)

        return JSONEncoder.default(self, obj)

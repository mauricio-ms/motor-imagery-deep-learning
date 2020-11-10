import json
from json import JSONEncoder

import numpy as np


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


class LabelsCounts:
    def __init__(self):
        self.__counts = {}

    def put(self, key, counts):
        if key not in self.__counts:
            self.__counts[key] = counts
        else:
            self.__counts[key] = self.__counts[key] + counts

    def __repr__(self):
        return json.dumps(self.__counts, sort_keys=True, indent=4, cls=NumpyArrayEncoder)

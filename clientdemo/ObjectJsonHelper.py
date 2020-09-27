#!/usr/bin/python3
from collections import namedtuple
from json import JSONEncoder


class ModelEncoder(JSONEncoder):
    def default(self, o):
        return o.__dict__


def model_decoder(ages):
    return namedtuple('X', ages.keys())(*ages.values())

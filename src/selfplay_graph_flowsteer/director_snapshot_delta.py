"""Lossless JSON updates, scoped to one Director run."""

import copy
import json


def diff(old, new, path=()):
    if type(old) is not type(new):
        return [["set", list(path), new]]
    if isinstance(old, dict):
        ops = [["delete", list(path + (k,))] for k in sorted(old.keys() - new.keys())]
        for k in sorted(new):
            ops += (
                diff(old[k], new[k], path + (k,))
                if k in old
                else [["set", list(path + (k,)), new[k]]]
            )
        return ops
    return [] if old == new else [["set", list(path), new]]


def apply(state, seq, packet):
    if packet["base"] != seq or packet["seq"] != seq + 1:
        raise ValueError("missing, duplicate or out-of-order snapshot")
    if "full" in packet:
        return copy.deepcopy(packet["full"]), packet["seq"]
    state = copy.deepcopy(state)
    for op in packet["delta"]:
        kind, path = op[:2]
        if not path:
            if kind != "set":
                raise ValueError("invalid root operation")
            state = copy.deepcopy(op[2])
            continue
        parent = state
        for key in path[:-1]:
            parent = parent[key]
        if kind == "set":
            parent[path[-1]] = copy.deepcopy(op[2])
        elif kind == "delete":
            del parent[path[-1]]
        else:
            raise ValueError("unknown operation")
    return state, packet["seq"]


class SnapshotCodec:
    def __init__(self):
        self.state, self.seq = None, -1

    def encode(self, state):
        packet = {"base": self.seq, "seq": self.seq + 1}
        if self.seq < 0:
            packet["full"] = state
        else:
            packet["delta"] = diff(self.state, state)
        restored, seq = apply(self.state, self.seq, packet)
        if restored != state:
            raise ValueError("delta reconstruction failed")
        self.state, self.seq = restored, seq
        return json.dumps(packet, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

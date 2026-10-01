#!/usr/bin/env python3
"""ax-1b/chu — space form of the Axb predictive-echo solve.

Declared form:
    ax - 1b / (c h u) = 0
    x = pinv(A) @ (b / (c * h * u))

c, h, u are risk divisors taken only from the predicted echo.
Real return corrects the hypothesis. It never sets the step.

Subsumption (space):
    1 Provide air   — c  cabin / O2 margin
    2 Eat           — h  heat and power margin
    3 Win           — mission delta along b
    4 Talk          — push path, then report. u is the uplink hold.
"""

from __future__ import annotations

import numpy as np

# Divisors never fall below 1. A missing push forces u large so the step dies.
U_NO_PUSH = 8.0
C_ABORT = 1.85
H_ABORT = 1.85


def dir_vec(b):
    b = np.asarray(b, dtype=float)
    n = np.linalg.norm(b)
    return b / n if n > 1e-9 else np.zeros_like(b)


class GroundStation:
    """Holds goal and the last pushed path. Robot never receives geometry."""

    def __init__(self, goal):
        self.goal = np.asarray(goal, dtype=float)
        self.ground_ok = True
        self.last_path = None
        self.log = []

    def push(self, path):
        self.last_path = {
            "x": np.asarray(path["x"], dtype=float).tolist(),
            "c": float(path["c"]),
            "h": float(path["h"]),
            "u": float(path["u"]),
            "layer": path["layer"],
        }
        self.log.append(self.last_path)
        return True


class EchoHypothesis:
    """Internal scatterer belief only. No world map."""

    def __init__(self):
        self.scatterers = []

    def predict(self, pose):
        pose = np.asarray(pose, dtype=float)
        echoes = []
        for s in self.scatterers:
            s = np.asarray(s, dtype=float)
            echoes.append({"range_m": float(np.linalg.norm(s - pose))})
        echoes.sort(key=lambda e: e["range_m"])
        return echoes

    def correct(self, pose, real_echoes):
        pose = np.asarray(pose, dtype=float)
        self.scatterers = []
        for e in real_echoes[:4]:
            bearing = dir_vec(e.get("bearing", [1.0, 0.0, 0.0]))
            self.scatterers.append(pose + bearing * float(e["range_m"]))

    def surprise(self, pose, real_echoes):
        pred = self.predict(pose)
        if not pred and not real_echoes:
            return 0.0
        pr = pred[0]["range_m"] if pred else 9.0
        rr = real_echoes[0]["range_m"] if real_echoes else 9.0
        return abs(pr - rr) / max(rr, 0.25)


class Sandbox:
    """Deep-copy belief. Risk divisors come only from the predicted echo."""

    def __init__(self, pose, hyp, cabin=1.0, heat=1.0, link=1.0):
        self.pose = np.asarray(pose, dtype=float).copy()
        self.hyp = hyp
        self.cabin = cabin
        self.heat = heat
        self.link = link

    def probe(self, action):
        pose = self.pose + np.asarray(action, dtype=float)
        predicted = self.hyp.predict(pose)
        nearest = predicted[0]["range_m"] if predicted else np.inf
        # c rises as the predicted hull/debris return closes inside 4 m
        c = max(1.0, self.cabin) * (1.0 + max(0.0, 1.0 - nearest / 4.0))
        # h rises if the step is large relative to a 0.6 m power budget
        step = float(np.linalg.norm(action))
        h = max(1.0, self.heat) * (1.0 + max(0.0, step / 0.6 - 1.0))
        # u is the uplink hold. Must be >= 1. Weak predicted link shrinks x.
        u = max(1.0, self.link)
        return {"predicted": predicted, "c": c, "h": h, "u": u, "nearest": nearest}


def solve_ax_1b_chu(A, b, c, h, u):
    """ax - 1b/(c h u) = 0  →  x = pinv(A) @ (b / (c h u))"""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    chu = max(c, 1.0) * max(h, 1.0) * max(u, 1.0)
    rhs = b / chu
    return np.linalg.pinv(A) @ rhs, chu


def layer_of(c, h, pushed):
    if c >= C_ABORT:
        return "air"
    if h >= H_ABORT:
        return "eat"
    if not pushed:
        return "talk-hold"
    return "win"


def step(pose, goal, hyp, station, cabin=1.0, heat=1.0, link=1.0, n=8):
    pose = np.asarray(pose, dtype=float)
    goal = np.asarray(goal, dtype=float)
    b = goal - pose
    A = np.eye(3)
    best = None
    rng = np.random.default_rng(7)
    trials = [dir_vec(b)]
    for _ in range(n - 1):
        trials.append(dir_vec(b + rng.normal(0, 0.35, size=3)))
    for trial in trials:
        box = Sandbox(pose, hyp, cabin=cabin, heat=heat, link=link)
        # provisional action length 0.4 m, then the solve shrinks it
        probe = box.probe(trial * 0.4)
        x, chu = solve_ax_1b_chu(A, trial, probe["c"], probe["h"], probe["u"])
        # clamp step so a bad divisor cannot fling the craft
        nrm = np.linalg.norm(x)
        if nrm > 0.45:
            x = x * (0.45 / nrm)
        progress = float(np.dot(dir_vec(b), dir_vec(x))) if np.linalg.norm(x) > 1e-9 else -1
        score = progress - 0.35 * (probe["c"] + probe["h"] + probe["u"] - 3.0)
        cand = {"x": x, "c": probe["c"], "h": probe["h"], "u": probe["u"], "chu": chu, "score": score}
        if best is None or cand["score"] > best["score"]:
            best = cand
    layer = layer_of(best["c"], best["h"], pushed=False)
    if layer == "air":
        best["x"] = np.zeros(3)  # inhibit motion; hold cabin
    elif layer == "eat":
        best["x"] = best["x"] * 0.25
    # Talk before move: push, then u is allowed to stay at the probed value.
    station.push({**best, "layer": "talk"})
    layer = layer_of(best["c"], best["h"], pushed=station.last_path is not None)
    best["layer"] = layer
    new_pose = pose + best["x"]
    return new_pose, best


def demo():
    goal = np.array([6.0, 1.0, 0.2])
    pose = np.array([0.0, 0.0, 0.0])
    station = GroundStation(goal)
    hyp = EchoHypothesis()
    # true scatterers exist only for the ear simulator
    truth = [np.array([2.2, 0.4, 0.0]), np.array([4.5, -0.3, 0.5])]
    rows = []
    for i in range(10):
        pose, decided = step(pose, goal, hyp, station, cabin=1.0, heat=1.0, link=1.05)
        real = []
        for s in truth:
            real.append({
                "range_m": float(np.linalg.norm(s - pose)),
                "bearing": (s - pose).tolist(),
            })
        real.sort(key=lambda e: e["range_m"])
        surprise = hyp.surprise(pose, real)
        hyp.correct(pose, real)
        rows.append({
            "i": i,
            "pose": np.round(pose, 3).tolist(),
            "x": np.round(decided["x"], 3).tolist(),
            "c": round(decided["c"], 3),
            "h": round(decided["h"], 3),
            "u": round(decided["u"], 3),
            "chu": round(decided["chu"], 3),
            "layer": decided["layer"],
            "surprise": round(float(surprise), 3),
            "pushed": station.last_path is not None,
        })
        if np.linalg.norm(goal - pose) < 0.5:
            break
    return rows


if __name__ == "__main__":
    for row in demo():
        print(row)

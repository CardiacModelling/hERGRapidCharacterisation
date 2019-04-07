#!/usr/bin/env python2
import numpy as np

protocol_leak_check = {
        # protocol: (t_start, t_end)  # second
        'staircaseramp': None,  # Not doing for this
        'pharma': (0.175, 0.2),
        'apab': (0.065, 0.085),
        'apabv3': (0.065, 0.085),
        'ap05hz': (0.065, 0.085),
        'ap1hz': (0.065, 0.085),
        'ap2hz': (0.065, 0.085),
        'sactiv': (0.175, 0.2),  # TODO
        'sinactiv': (0.175, 0.2),
        }


def I_releak(g, d, v):
    # New leak corrected current from a leak corrected trace
    return g * (v - v[0]) + d


# TODO can try np.abs(I_win1 / I_win2 - constant)?
def score_leak(g, d, v, t, t_win):
    # function to be minimised for re-leak estimation
    i = np.argmin(np.abs(t - t_win[0]))
    f = np.argmin(np.abs(t - t_win[1]))
    return np.abs(np.mean(I_releak(g, d, v)[i:f]))


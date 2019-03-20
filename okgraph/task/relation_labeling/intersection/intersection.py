from okgraph.sliding_windows import SlidingWindows


def task(windows: [SlidingWindows], tail=15):
    l = []

    for i in range(len(windows)):
        l.append({k: windows[i].results[k] for k in list(windows[i].results)[:tail]})

    r = {}
    for i in range(len(l)):
        if i == 0:
            r = set(l[i].keys()) & set(l[i + 1].keys())
        elif i + 1 < len(l):
            r = set(r) & set(l[i + 1].keys())

    return r

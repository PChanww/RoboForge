"""Small workcell route planner. All lengths are metres, in world XY."""
from __future__ import annotations

import heapq
import math
import numpy as np

BOUNDS = [-1.2, 1.2, -.6, 1.5]
START = np.array([0., .205])
BELT = [-.32, .055, -.145, -.005]
CLEARANCE = .16


def rect_distance(point, rect):
    x, y = point
    return math.hypot(max(rect[0]-x, 0., x-rect[1]), max(rect[2]-y, 0., y-rect[3]))


def plan_navigation(target_xy, bin_size_m=(.21, .22, .056)):
    target = np.asarray(target_xy, dtype=float)
    size = np.asarray(bin_size_m, dtype=float)
    issues = []
    if target.shape != (2,) or not np.isfinite(target).all():
        return {"ready": False, "issues": ["目标箱坐标需要两个有限数值。"]}
    if size.shape != (3,) or not np.isfinite(size).all() or not (.16 <= size[0] <= .30 and .16 <= size[1] <= .30 and .04 <= size[2] <= .09):
        return {"ready": False, "issues": ["箱宽深支持16～30 cm，墙高支持4～9 cm。"]}
    bx, by = target
    box = [bx-size[0]/2, bx+size[0]/2, by-size[1]/2, by+size[1]/2]
    if box[0] < BOUNDS[0] or box[1] > BOUNDS[1] or box[2] < BOUNDS[2] or box[3] > BOUNDS[3]:
        issues.append("箱体外轮廓需完整位于地图场区内。")
    if box[0] < BELT[1]+.03 and box[1] > BELT[0]-.03 and box[2] < BELT[3]+.03 and box[3] > BELT[2]-.03:
        issues.append("目标箱与传送带重叠，请移到空地。")
    if rect_distance(START, box) < CLEARANCE+.025:
        issues.append("目标箱与机器人初始工作区太近，请留出移动空间。")
    result = {"ready": False, "issues": issues, "bounds": BOUNDS, "obstacles": [BELT, box], "target_xy": target.tolist()}
    if issues:
        return result

    def free(p):
        return (BOUNDS[0]+CLEARANCE <= p[0] <= BOUNDS[1]-CLEARANCE and
                BOUNDS[2]+CLEARANCE <= p[1] <= BOUNDS[3]-CLEARANCE and
                all(rect_distance(p, rect) > CLEARANCE for rect in (BELT, box)))

    def clear(a, b):
        n = max(2, int(np.linalg.norm(np.asarray(b)-a)/.015)+1)
        return all(free(p) for p in np.linspace(a, b, n))

    step = .04
    origin = np.array([BOUNDS[0]+CLEARANCE, BOUNDS[2]+CLEARANCE])
    def key(p): return tuple(np.rint((np.asarray(p)-origin)/step).astype(int))
    def pos(k): return origin+np.asarray(k)*step

    start_key = key(START)
    queue, costs, parents = [(0., start_key)], {start_key: 0.}, {}
    while queue:
        cost, node = heapq.heappop(queue)
        if cost > costs[node]+1e-9:
            continue
        for dx, dy in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
            other = (node[0]+dx,node[1]+dy)
            a, b = pos(node), pos(other)
            if not free(b) or not clear(a, b):
                continue
            new = cost+step*math.hypot(dx, dy)
            if new < costs.get(other, math.inf):
                costs[other], parents[other] = new, node
                heapq.heappush(queue, (new, other))

    radius = max(.34, float(max(size[:2])/2+.19))
    options = []
    for angle in np.linspace(0, 2*math.pi, 16, endpoint=False):
        dock = target+radius*np.array([math.cos(angle),math.sin(angle)])
        node = key(dock)
        if node not in costs or not free(dock) or not clear(pos(node), dock):
            continue
        options.append((costs[node], node, dock))
    if not options:
        result["issues"] = ["该位置没有足够的停靠和放置空间，请移动箱子。"]
        return result
    _, node, dock = min(options, key=lambda row: row[0])
    route = [dock, pos(node)]
    while node in parents:
        node = parents[node]
        route.append(pos(node))
    route.append(START.copy())
    route.reverse()
    # Keep only corners whose connecting segments retain collision clearance.
    smooth, i = [route[0]], 0
    while i < len(route)-1:
        j = len(route)-1
        while j > i+1 and not clear(route[i], route[j]):
            j -= 1
        smooth.append(route[j]); i = j
    to_bin = target-dock
    final_yaw = math.atan2(to_bin[1], to_bin[0])+math.pi/2
    final_yaw = math.atan2(math.sin(final_yaw), math.cos(final_yaw))
    result.update(ready=True, planned_route=[np.asarray(p).tolist() for p in smooth], dock_xy=dock.tolist(),
                  length_m=sum(float(np.linalg.norm(np.asarray(b)-a)) for a,b in zip(smooth,smooth[1:])),
                  clearance_m=CLEARANCE, final_yaw=final_yaw, heading_policy="face_bin_before_placement",
                  final_yaw_basis="body forward is local -Y; physical docking alignment tracks the actual bin bearing")
    return result

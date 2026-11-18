"""HMM + Viterbi route reconstruction.

The independent (classifier-only) prediction can assign two consecutive
observations to road segments that are not connected in the network.  We wrap the
classifier in a hidden Markov model:

* **hidden states**  = the candidate road segment (arc id) selected at each
  observation,
* **emission**       = log probability that the candidate is correct, taken from
  the trained classifier (the "selected model's outputs"),
* **transition**     = a function of the agreement between the straight-line GPS
  distance and the distance along the road network, so that jumps to disconnected
  or inconsistent segments are penalised.

Viterbi decoding returns the most likely sequence of arcs.
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np

from .geo import LocalProjection

EPS = 1e-12


def build_network(nodes, arcs):
    """Precompute road-network structures used by the transition model.

    Returns a dict with (all coordinates projected to a local metre grid):
        node_xy      (V,2)
        arc_nodes    (E,2)
        arc_len      (E,)
        node_adj     node -> list of (neighbor_node, arc_id)
        arc_neighbors arc -> set of adjacent arc ids (share a node)
    """
    proj = LocalProjection(nodes[:, 0].mean(), nodes[:, 1].mean())
    nx, ny = proj.xy(nodes[:, 0], nodes[:, 1])
    a1 = arcs[:, 0]
    a2 = arcs[:, 1]
    arc_len = np.sqrt((nx[a2] - nx[a1]) ** 2 + (ny[a2] - ny[a1]) ** 2)

    node_adj = defaultdict(list)
    arc_neighbors = defaultdict(set)
    for ai, (u, v) in enumerate(arcs):
        u = int(u); v = int(v)
        node_adj[u].append((v, ai))
        node_adj[v].append((u, ai))
    # arc neighbors via shared nodes
    node_arcs = defaultdict(list)
    for ai, (u, v) in enumerate(arcs):
        node_arcs[int(u)].append(ai)
        node_arcs[int(v)].append(ai)
    for arcs_at_node in node_arcs.values():
        for a in arcs_at_node:
            for b in arcs_at_node:
                if a != b:
                    arc_neighbors[a].add(b)
    return dict(
        node_xy=np.column_stack([nx, ny]), arc_nodes=arcs, arc_len=arc_len,
        node_adj=node_adj, arc_neighbors=arc_neighbors,
    )


def _shortest_path(net, src_node, dst_node, limit):
    """Bounded Dijkstra on the node graph; returns distance or inf if > limit."""
    if src_node == dst_node:
        return 0.0
    node_adj = net["node_adj"]
    arc_len = net["arc_len"]
    import heapq
    dist = {int(src_node): 0.0}
    pq = [(0.0, int(src_node))]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, np.inf):
            continue
        if u == int(dst_node):
            return d
        if d > limit:
            return np.inf
        for v, ai in node_adj.get(u, ()):
            nd = d + float(arc_len[ai])
            if nd > limit:
                continue
            if nd < dist.get(int(v), np.inf):
                dist[int(v)] = nd
                heapq.heappush(pq, (nd, int(v)))
    return np.inf


class HMMDecoder:
    def __init__(self, beta: float = 20.0, disconnected_penalty: float = 1e-6,
                 path_slack: float = 1.5, extra_margin: float = 100.0,
                 switch_penalty: float = 1.0):
        self.beta = beta
        self.disconnected_penalty = disconnected_penalty
        self.path_slack = path_slack
        self.extra_margin = extra_margin
        # small log-penalty for changing arc, discouraging junction oscillation
        self.switch_penalty = switch_penalty

    def transition_logprob(self, net, a, b, ta, tb, gps_dist, arc_nodes):
        """Log transition probability from arc a (proj frac ta) to arc b (frac tb)."""
        ln_eps = np.log(self.disconnected_penalty)
        if a < 0 or b < 0:
            return ln_eps
        arc_len = net["arc_len"]
        if a == b:
            route_dist = abs(tb - ta) * float(arc_len[a])
            base = -abs(gps_dist - route_dist) / self.beta
        elif b in net["arc_neighbors"][a]:
            # find a shared node and route distance through it
            shared = set(arc_nodes[a]) & set(arc_nodes[b])
            if not shared:
                return ln_eps
            sn = int(next(iter(shared)))
            ua = int(arc_nodes[a][0]); va = int(arc_nodes[a][1])
            ub = int(arc_nodes[b][0]); vb = int(arc_nodes[b][1])
            # distance along a from projection to shared node
            da = abs(ta - (0.0 if ua == sn else 1.0)) * float(arc_len[a])
            db = abs(tb - (0.0 if ub == sn else 1.0)) * float(arc_len[b])
            route_dist = da + db
            base = -abs(gps_dist - route_dist) / self.beta - self.switch_penalty
        else:
            limit = gps_dist * self.path_slack + self.extra_margin
            # route via node graph: try each endpoint combination
            ua, va = int(arc_nodes[a][0]), int(arc_nodes[a][1])
            ub, vb = int(arc_nodes[b][0]), int(arc_nodes[b][1])
            # distance within a from projection to each endpoint
            ta_r = float(arc_len[a]) * ta
            tb_r = float(arc_len[b]) * tb
            best = np.inf
            for s in (ua, va):
                for e in (ub, vb):
                    d = _shortest_path(net, s, e, limit)
                    if d < np.inf:
                        # add distance from projection to s on a, and e to projection on b
                        da = abs(ta_r - (0.0 if s == ua else float(arc_len[a])))
                        db = abs(tb_r - (0.0 if e == ub else float(arc_len[b])))
                        best = min(best, da + d + db)
            if best == np.inf or best > limit:
                return ln_eps
            route_dist = best
            base = -abs(gps_dist - route_dist) / self.beta - self.switch_penalty

        return base

    def decode(self, net, track, cand_arcs, cand_proj, emission_logprob):
        """Viterbi decoding.

        Parameters
        ----------
        net : network dict from build_network
        track : (N,3)
        cand_arcs : (N,K) candidate arc ids (-1 padding)
        cand_proj : (N,K) projection fraction on the candidate arc
        emission_logprob : (N,K) log emission probability per candidate

        Returns
        -------
        pred_arcs : (N,) int64
        """
        N, K = cand_arcs.shape
        proj = LocalProjection(track[:, 0].mean(), track[:, 1].mean())
        px, py = proj.xy(track[:, 0], track[:, 1])
        gps_dist = np.sqrt(np.diff(px) ** 2 + np.diff(py) ** 2)

        arc_nodes = net["arc_nodes"]
        log_dp = np.full((N, K), -np.inf)
        backptr = np.zeros((N, K), dtype=np.int32)
        log_dp[0] = emission_logprob[0]

        for i in range(1, N):
            gd = gps_dist[i - 1]
            for k in range(K):
                a = int(cand_arcs[i, k])
                if a < 0:
                    continue
                ta = cand_proj[i, k]
                best_log = -np.inf
                best_prev = 0
                for kp in range(K):
                    prev_log = log_dp[i - 1, kp]
                    if prev_log == -np.inf:
                        continue
                    b = int(cand_arcs[i - 1, kp])
                    tb = cand_proj[i - 1, kp]
                    tr = self.transition_logprob(net, b, a, tb, ta, gd, arc_nodes)
                    total = prev_log + tr
                    if total > best_log:
                        best_log = total
                        best_prev = kp
                log_dp[i, k] = best_log + emission_logprob[i, k]
                backptr[i, k] = best_prev

        pred = np.empty(N, dtype=np.int64)
        pred[-1] = int(cand_arcs[-1, int(np.argmax(log_dp[-1]))])
        prev = int(np.argmax(log_dp[-1]))
        for i in range(N - 2, -1, -1):
            prev = int(backptr[i + 1, prev])
            pred[i] = int(cand_arcs[i, prev])
        return pred

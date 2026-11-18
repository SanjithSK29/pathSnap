"""End-to-end inference pipeline: raw trip -> predicted route.

The pipeline is used both by the evaluation scripts and the FastAPI backend:

    load trip -> generate candidates -> features -> classifier scores
               -> (optional) HMM/Viterbi decode -> predicted arc per observation
"""
from __future__ import annotations

import numpy as np

from .candidates import generate_candidates
from .features import candidate_projection, pair_features_with_relative
from .hmm import HMMDecoder, build_network

RADIUS = 60.0
K = 10


def candidate_emission(model, trip, cand):
    """Classifier log-probabilities per (point, candidate) as (N, K) matrix."""
    X, point_idx, arcs = pair_features_with_relative(trip, cand)
    proba = model.predict_proba(X)[:, 1]
    proba = np.clip(proba, 1e-12, 1 - 1e-12)
    N, K = cand.arcs.shape
    em = np.full((N, K), -1e9, dtype=np.float32)  # padding -> impossible
    em[np.nonzero(cand.arcs >= 0)] = np.log(proba)
    return em


def predict_trip(model, trip, radius=RADIUS, k=K, use_hmm=True,
                 hmm: HMMDecoder | None = None):
    """Predict one arc per observation.

    Returns
    -------
    result : dict
        cand (CandidateSet), classifier_arcs (N,), final_arcs (N,),
        proj (N,K), emission (N,K)
    """
    cand = generate_candidates(trip, radius=radius, k=k)
    X, point_idx, arcs = pair_features_with_relative(trip, cand)
    proba = model.predict_proba(X)[:, 1]

    N = trip.n_points
    from .evaluate import point_predictions
    classifier_arcs, _ = point_predictions(point_idx, arcs, proba, N)

    final_arcs = classifier_arcs
    proj = candidate_projection(trip, cand.arcs)
    emission = None
    if use_hmm:
        proba_c = np.clip(proba, 1e-12, 1 - 1e-12)
        emission = np.full((N, k), -1e9, dtype=np.float32)
        emission[np.nonzero(cand.arcs >= 0)] = np.log(proba_c)
        net = build_network(trip.nodes, trip.arcs)
        if hmm is None:
            hmm = HMMDecoder()
        final_arcs = hmm.decode(net, trip.track, cand.arcs, proj, emission)

    return dict(
        cand=cand, classifier_arcs=classifier_arcs, final_arcs=final_arcs,
        proj=proj, emission=emission,
    )


def reconstruct_route(final_arcs):
    """Compress the per-observation arc sequence into a route (dedupe runs)."""
    if len(final_arcs) == 0:
        return np.array([], dtype=np.int64)
    keep = np.empty(len(final_arcs), dtype=bool)
    keep[0] = True
    keep[1:] = final_arcs[1:] != final_arcs[:-1]
    return final_arcs[keep].astype(np.int64)

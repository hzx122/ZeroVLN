"""Search a physically verified graph under cumulative route budgets.

The caller supplies the existing measured edge proof. Multiple nondominated
labels preserve a longer arrival when it leaves more budget for later stairs.
This module has no scene endpoints, annotation policy or clearance settings.
"""
import heapq
import json
import math
import os
import time
from collections import defaultdict
from functools import lru_cache

import numpy as np


def _insert_frontier_rows(costs, ids, size, candidate, label_id, known_undominated=False):
    """Exact stable compaction; no reassociation or resource quantization."""
    width = costs.shape[1]
    # Complete the rejection pass before mutating either array. A rejected
    # epsilon-equivalent candidate must leave all old label identities alive.
    if not known_undominated:
        for row in range(size):
            dominates = True
            for column in range(width):
                if not costs[row, column] <= candidate[column] + 1e-9:
                    dominates = False
                    break
            if dominates:
                return False, size, np.empty(0, dtype=np.int64)
    removed = np.empty(size, dtype=np.int64)
    removed_count = 0
    retained = 0
    for row in range(size):
        dominated = True
        for column in range(width):
            if not candidate[column] <= costs[row, column] + 1e-9:
                dominated = False
                break
        if dominated:
            removed[removed_count] = ids[row]
            removed_count += 1
        else:
            if retained != row:
                for column in range(width):
                    costs[retained, column] = costs[row, column]
                ids[retained] = ids[row]
            retained += 1
    for column in range(width):
        costs[retained, column] = candidate[column]
    ids[retained] = label_id
    return True, retained + 1, removed[:removed_count]


def _frontier_dominator_index(costs, size, candidate):
    for row in range(size):
        dominates = True
        for column in range(costs.shape[1]):
            if not costs[row, column] <= candidate[column] + 1e-9:
                dominates = False
                break
        if dominates:
            return row
    return -1


def _frontier_dominates_rows(costs, size, candidate):
    return _frontier_dominator_index(costs, size, candidate) >= 0


@lru_cache(maxsize=1)
def _compiled_frontier_scan():
    if _compiled_frontier_insert() is None:
        return None
    from numba import njit
    return njit(cache=False, fastmath=False, nogil=True)(_frontier_dominator_index)


@lru_cache(maxsize=1)
def _compiled_frontier_dominates():
    scan = _compiled_frontier_scan()
    if scan is None:
        return None
    from numba import njit

    def query(costs, size, candidate, minimum, witness, absent, use_cache, counts):
        counts[0] += 1
        if use_cache:
            for column in range(len(minimum)):
                if minimum[column] > candidate[column] + 1e-9:
                    counts[3] += 1
                    return -1, 3
            if len(witness):
                dominated = True
                for column in range(len(witness)):
                    if not witness[column] <= candidate[column] + 1e-9:
                        dominated = False
                        break
                if dominated:
                    counts[1] += 1
                    return -2, 1
            if len(absent):
                contained = True
                for column in range(len(absent)):
                    if not candidate[column] <= absent[column]:
                        contained = False
                        break
                if contained:
                    counts[2] += 1
                    return -1, 2
        counts[4] += 1
        index = scan(costs, size, candidate)
        counts[5] += index + 1 if index >= 0 else size
        return index, 0

    return njit(cache=False, fastmath=False, nogil=True)(query)


@lru_cache(maxsize=1)
def _compiled_frontier_insert():
    # Habitat's existing runtime contains Numba. Other callers retain the
    # exact NumPy implementation when Numba is absent or NumPy-incompatible.
    # Cache=False prevents writes alongside frozen source on the shared disk.
    import os
    if os.environ.get('NUMBA_DISABLE_JIT') == '1':
        return None
    try:
        from numba import njit
    except ImportError:
        return None
    return njit(cache=False, fastmath=False, nogil=True)(_insert_frontier_rows)


class _ParetoFrontier:
    """Preserve original dominance/tolerance/order in contiguous arrays."""

    def __init__(self, width, *, query_cache=True):
        self.costs = np.empty((8, width), dtype=np.float64)
        self.ids = np.empty(8, dtype=np.int64)
        self.size = 0
        self.kernel = _compiled_frontier_insert()
        self.linear_kernel = _compiled_frontier_scan() if self.kernel is not None else None
        self.dominance_kernel = _compiled_frontier_dominates() if self.kernel is not None else None
        self.query_cache = query_cache
        self.minimum = self.witness = self.absent_query = None
        self._query_counts = np.zeros(6, dtype=np.int64)
        self._empty_query = np.empty(0, dtype=np.float64)

    @property
    def query_counts(self):
        # Count large-frontier queries only; small frontiers keep the original
        # single compiled call without bookkeeping or cache checks.
        return dict(zip(('calls', 'witness_hits', 'absent_hits', 'minimum_hits',
                         'full_scans', 'rows_scanned'), map(int, self._query_counts)))

    def __bool__(self):
        return self.size > 0

    def contains_dominator(self, candidate):
        candidate = np.asarray(candidate, dtype=np.float64)
        counts = self._query_counts
        # These are query shortcuts, not a frontier-size or search-label cap.
        # Small frontiers retain the compiled linear scan without cache checks.
        if self.size < 256:
            if self.linear_kernel is not None:
                return self.linear_kernel(self.costs, self.size, candidate) >= 0
            return bool(np.any(np.all(self.costs[:self.size] <= candidate+1e-9, axis=1)))
        cache = self.query_cache
        if self.dominance_kernel is not None:
            empty = self._empty_query
            index, hit = self.dominance_kernel(
                self.costs, self.size, candidate,
                empty if self.minimum is None else self.minimum,
                empty if self.witness is None else self.witness,
                empty if self.absent_query is None else self.absent_query, cache, counts)
            if cache and hit == 0:
                if index >= 0:
                    self.witness = self.costs[index].copy()
                else:
                    self.absent_query = candidate.copy()
            return index != -1
        counts[0] += 1
        if cache:
            if self.minimum is not None and np.any(self.minimum > candidate+1e-9):
                counts[3] += 1
                return False
            if self.witness is not None and np.all(self.witness <= candidate+1e-9):
                counts[1] += 1
                return True
            # Exact monotonic containment of an already empty query orthant.
            # Do not add another epsilon here: dominance's tolerance is not
            # transitive. A smaller candidate has a no-larger candidate+epsilon.
            if self.absent_query is not None and np.all(candidate <= self.absent_query):
                counts[2] += 1
                return False
        counts[4] += 1
        found = np.flatnonzero(np.all(self.costs[:self.size] <= candidate+1e-9, axis=1))
        index = int(found[0]) if len(found) else -1
        counts[5] += index + 1 if index >= 0 else self.size
        if cache:
            if index >= 0:
                self.witness = self.costs[index].copy()
            else:
                self.absent_query = candidate.copy()
        return index >= 0

    def _accepted_query_version(self, candidate):
        if not self.query_cache:
            return
        # Stable compaction can invalidate every old witness and empty query.
        # The accepted candidate is itself an active, exact positive witness.
        self.witness = candidate.copy()
        self.absent_query = None
        # Historical component minima are conservative lower bounds even if a
        # minimizing row is later removed; no rescan or rounded resource needed.
        self.minimum = candidate.copy() if self.minimum is None else np.minimum(self.minimum, candidate)

    def insert(self, candidate, label_id):
        candidate = np.asarray(candidate, dtype=np.float64)
        known_undominated = False
        if self.query_cache and self.size >= 256:
            # Reuse exact query evidence before the old rejection pass. No
            # frontier mutation occurs between this proof and compaction.
            if self.contains_dominator(candidate):
                return False, ()
            known_undominated = True
        if self.kernel is not None:
            # One spare row lets the kernel append after stable compaction.
            if self.size == len(self.ids):
                costs = np.empty((self.size*2, self.costs.shape[1]), dtype=np.float64)
                ids = np.empty(self.size*2, dtype=np.int64)
                costs[:self.size], ids[:self.size] = self.costs, self.ids
                self.costs, self.ids = costs, ids
            accepted, self.size, removed = self.kernel(
                self.costs, self.ids, self.size, candidate, label_id, known_undominated)
            if accepted:
                self._accepted_query_version(candidate)
            return accepted, removed.tolist() if accepted else ()
        rows = self.costs[:self.size]
        # Keep the same arithmetic as dominates(a,b): a <= b + 1e-9.
        if not known_undominated and np.any(np.all(rows <= candidate+1e-9, axis=1)):
            return False, ()
        removed_mask = np.all(candidate <= rows+1e-9, axis=1)
        removed = self.ids[:self.size][removed_mask].tolist()
        if removed:
            survivors = ~removed_mask
            count = int(survivors.sum())
            self.costs[:count] = rows[survivors]
            self.ids[:count] = self.ids[:self.size][survivors]
            self.size = count
        if self.size == len(self.ids):
            costs = np.empty((self.size*2, self.costs.shape[1]), dtype=np.float64)
            ids = np.empty(self.size*2, dtype=np.int64)
            costs[:self.size], ids[:self.size] = self.costs, self.ids
            self.costs, self.ids = costs, ids
        self.costs[self.size] = candidate
        self.ids[self.size] = label_id
        self.size += 1
        self._accepted_query_version(candidate)
        return True, removed


def _completion_bounds(neighbors, end, costs, caps):
    """Exact scalar relaxations on the supplied directed completion-cost graph.

    Costs may be physical proofs or caller-proved optimistic edge costs.
    Each component is minimized independently, so these bounds can only
    underestimate a feasible completion's joint costs. Edges individually
    over a budget cannot be part of any feasible nonnegative-cost route.
    Stop each relaxation beyond its original cap; omitted nodes then have no
    completion within that cap. No label cap or resource rounding is used.
    """
    reverse = defaultdict(list)
    for source, targets in neighbors.items():
        for target in targets:
            reverse[target].append(source)
    bounds = []
    for component, cap in enumerate(caps):
        best = {end: 0.}
        queue = [(0., end)]
        while queue:
            value, node = heapq.heappop(queue)
            if value != best[node]:
                continue
            for source in reverse.get(node, ()):
                proof = costs(source, node)
                if proof is None or any(v > limit+1e-9 for v, limit in zip(proof, caps)):
                    continue
                candidate = value+proof[component]
                if candidate <= cap+1e-9 and candidate < best.get(source, math.inf):
                    best[source] = candidate
                    heapq.heappush(queue, (candidate, source))
        bounds.append(best)
    return bounds


def shortest_physical_path(neighbors, start, end, edge_length, *, max_length,
                           heuristic=lambda node: 0., edge_length_bound=None,
                           diagnostics=None):
    """Existing scalar native A*: one best distance per node, measured edges.

    Return the path and reached distances. The caller's admissible heuristic
    and length bound cannot establish an edge; only ``edge_length`` can.
    Resource-constrained callers may use this as an optimality certificate,
    but must check the complete cumulative resources before accepting it.
    """
    costs, previous = {start: 0.}, {}
    queue = [(float(heuristic(start)), 0., start)]
    expanded = 0
    while queue:
        _, cost, node = heapq.heappop(queue)
        if cost != costs.get(node):
            continue
        expanded += 1
        if node == end:
            path = [node]
            while node in previous:
                node = previous[node]
                path.append(node)
            if diagnostics is not None:
                diagnostics.update(expanded_nodes=expanded, reached_nodes=len(costs))
            return path[::-1], costs
        for target in sorted(neighbors.get(node, ())):
            if edge_length_bound is not None:
                bound = float(edge_length_bound(node, target))
                if not math.isfinite(bound) or bound < 0:
                    raise ValueError('edge length bound must be finite and nonnegative')
                if cost+bound > max_length or cost+bound >= costs.get(target, math.inf):
                    continue
            length = edge_length(node, target)
            if length is None:
                continue
            length = float(length)
            if not math.isfinite(length) or length < 0:
                raise ValueError('edge costs must be finite and nonnegative')
            next_cost = cost+length
            if next_cost > max_length or next_cost >= costs.get(target, math.inf):
                continue
            costs[target], previous[target] = next_cost, node
            heapq.heappush(queue, (next_cost+float(heuristic(target)), next_cost, target))
    if diagnostics is not None:
        diagnostics.update(expanded_nodes=expanded, reached_nodes=len(costs))
    return None, costs


class _ResourceMinimumGraph:
    """Exact directed CSR costs, updated only by newly measured edge proofs.

    Reuse the existing SciPy sparse Dijkstra backend used by rule_credit_state.
    Stored zero-cost edges remain edges; infinity represents physical failure
    or an individually over-budget edge. No coordinate/cost quantization.
    """
    def __init__(self, neighbors, start, end, bound_costs, caps):
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import dijkstra
        self.csr_matrix, self.dijkstra = csr_matrix, dijkstra
        self.nodes = sorted({start, end, *neighbors,
                             *(b for targets in neighbors.values() for b in targets)})
        self.node_indices = {node: i for i, node in enumerate(self.nodes)}
        self.start, self.end = start, end
        self.caps = np.asarray(caps, dtype=np.float64)
        self.edge_rows = {}
        indices, indptr, values = [], [0], []
        for a in self.nodes:
            for b in sorted(set(neighbors.get(a, ()))):
                self.edge_rows[a, b] = len(values)
                proof = bound_costs(a, b)
                values.append([math.inf]*len(caps) if proof is None else proof)
                indices.append(self.node_indices[b])
            indptr.append(len(values))
        self.values = np.asarray(values, dtype=np.float64).reshape(-1, len(caps))
        self.indices = np.asarray(indices, dtype=np.int32)
        self.indptr = np.asarray(indptr, dtype=np.int32)

    def measured(self, a, b, proof):
        self.values[self.edge_rows[a, b]] = math.inf if proof is None else proof

    def shortest(self, component, cap):
        valid = np.all(self.values <= self.caps+1e-9, axis=1)
        data = np.where(valid, self.values[:, component], math.inf)
        graph = self.csr_matrix((data, self.indices, self.indptr),
                               shape=(len(self.nodes), len(self.nodes)))
        distances, previous = self.dijkstra(graph, directed=True,
            indices=self.node_indices[self.start], limit=cap+1e-9, return_predecessors=True)
        cursor = self.node_indices[self.end]
        minimum = float(distances[cursor])
        if not math.isfinite(minimum):
            return None, minimum
        path = [self.end]
        seen = {cursor}
        while cursor != self.node_indices[self.start]:
            cursor = int(previous[cursor])
            if cursor < 0 or cursor in seen:
                raise ValueError('invalid sparse shortest-path predecessor proof')
            seen.add(cursor)
            path.append(self.nodes[cursor])
        return path[::-1], minimum


def resource_constrained_path(neighbors, start, end, edge, *, resource_limits,
                              max_length=25., heuristic=lambda node: 0., diagnostics=None,
                              use_completion_bounds=False, edge_length_bound=None,
                              completion_edge_bound=None, use_shortest_certificate=False,
                              use_lazy_resource_minima=False, feasible_path_accept=None,
                              verified_proposals_only=False):
    if verified_proposals_only and feasible_path_accept is None:
        raise ValueError('proposal-only construction requires a full route verifier')
    limits = tuple(float(v) for v in resource_limits)
    if (not limits or any(not math.isfinite(v) or v < 0 for v in limits)
            or not math.isfinite(max_length) or max_length < 0):
        raise ValueError('finite nonnegative route budgets required')
    # Each label is (node, distance, cumulative resources, predecessor label).
    labels = [(start, 0., (0.,)*len(limits), None)]
    root_frontier = _ParetoFrontier(len(limits)+1)
    root_frontier.insert((0.,)+(0.,)*len(limits), 0)
    active = {start: root_frontier}
    live_labels = [True]
    queue = [(float(heuristic(start)), 0)]
    cache = {}
    expanded = 0
    pruned = 0
    optimistic_pruned = 0
    length_bounds = {}

    def checked_cost(proof):
        if proof is None:
            return None
        length, resources = proof
        values = (float(length),)+tuple(float(v) for v in resources)
        if (len(values) != len(limits)+1
                or any(not math.isfinite(v) or v < 0 for v in values)):
            raise ValueError('edge costs and resources must be finite and nonnegative')
        return values

    def costs(a, b):
        key = (a, b)
        if key not in cache:
            cache[key] = checked_cost(edge(a, b))
        return cache[key]

    # Caller-proved componentwise lower costs may include unverified edges.
    # This optimistic graph can only lower completion costs. It never proves
    # a physical edge: forward search and the final route still use `edge`.
    def bound_costs(a, b):
        # Newly measured proofs replace optimistic costs in subsequent bound
        # refinements. A measured physical failure removes the edge as well.
        if (a, b) in cache or completion_edge_bound is None:
            return costs(a, b)
        return checked_cost(completion_edge_bound(a, b))
    started = time.monotonic()
    bounds = (_completion_bounds(neighbors, end, bound_costs, (max_length,)+limits)
              if use_completion_bounds and start != end else [])
    if diagnostics is not None:
        diagnostics.update(frontier_backend=('numba_exact' if root_frontier.kernel is not None else 'numpy_exact'),
                           optimistic_frontier_pruned_edges=0,
                           completion_bounds=bool(bounds),
                           completion_bound_kind=('physical' if completion_edge_bound is None else 'admissible_edge_lower_bound'),
                           completion_bound_seconds=time.monotonic()-started,
                           completion_bound_nodes=[len(b) for b in bounds])

    progress_enabled = os.environ.get('RULE_NATIVE_SEARCH_PROGRESS') == '1'
    next_progress = time.monotonic() + 15.

    def frontier_metrics():
        return dict(max_frontier_labels=max(v.size for v in active.values()),
                    frontier_query_counts={key: sum(v.query_counts[key] for v in active.values())
                        for key in root_frontier.query_counts})

    if progress_enabled:
        print(json.dumps(dict(event='native_resource_search_progress', phase='completion_bounds_done',
            seconds=time.monotonic()-started, completion_nodes=[len(b) for b in bounds],
            examined_edges=len(cache))), flush=True)

    def completion_possible(node, distance, used):
        return not bounds or all(value+b.get(node, math.inf) <= cap+1e-9
                                 for value, b, cap in zip((distance,)+used, bounds,
                                                         (max_length,)+limits))

    if not completion_possible(start, 0., (0.,)*len(limits)):
        if diagnostics is not None:
            diagnostics.update(outcome='no_resource_bounded_path', expanded_labels=0,
                generated_labels=1, examined_edges=len(cache), reached_nodes=0,
                resource_limits=list(limits), completion_pruned_labels=1)
        return None

    def dominates(a, b):
        return all(x <= y+1e-9 for x, y in zip(a, b))

    def path_values(path):
        return tuple(sum(costs(a, b)[i] for a, b in zip(path, path[1:]))
                     for i in range(len(limits)+1))

    if use_shortest_certificate:
        certificate_started = time.monotonic()
        certificate = {}
        def scalar_length(a, b):
            values = costs(a, b)
            # An individually over-budget edge cannot belong to any feasible
            # nonnegative-resource route. No cumulative-arrival pruning here.
            return (None if values is None or values[0] > max_length+1e-9
                    or not dominates(values[1:], limits) else values[0])
        shortest, distances = shortest_physical_path(neighbors, start, end, scalar_length,
            max_length=max_length+1e-9, heuristic=heuristic,
            edge_length_bound=edge_length_bound, diagnostics=certificate)
        shortest_used = (tuple(sum(costs(a, b)[i+1] for a, b in zip(shortest, shortest[1:]))
                               for i in range(len(limits))) if shortest is not None else None)
        feasible = shortest is not None and dominates(shortest_used, limits)
        route_accepted = feasible and (feasible_path_accept is None or feasible_path_accept(shortest))
        certificate.update(seconds=time.monotonic()-certificate_started,
            outcome=('feasible_shortest_path' if route_accepted else
                     'route_contract_rejected' if feasible else
                     'no_physical_path' if shortest is None else 'cumulative_budget_exceeded'),
            length=distances.get(end), resources=None if shortest_used is None else list(shortest_used))
        if diagnostics is not None:
            diagnostics['shortest_certificate'] = certificate
        if progress_enabled:
            print(json.dumps(dict(event='native_resource_search_progress',
                phase='shortest_certificate_done', **certificate)), flush=True)
        if route_accepted or shortest is None:
            if diagnostics is not None:
                diagnostics.update(outcome='resource_bounded_path' if feasible else 'no_resource_bounded_path',
                    expanded_labels=0, generated_labels=1, examined_edges=len(cache),
                    resource_limits=list(limits), completion_pruned_labels=0,
                    optimistic_frontier_pruned_edges=0, **frontier_metrics())
                if feasible:
                    diagnostics.update(length=distances[end], resources=list(shortest_used))
            return shortest
        # The scalar optimum exhausted a cumulative budget. Keep every
        # nondominated arrival in the original Pareto search; shared exact
        # measured-edge proofs avoid repeating its physical work.

    if feasible_path_accept is not None:
        # Construction callers may accept any fully validated route. Scalar
        # resource penalties propose routes quickly; they NEVER prune Pareto
        # labels or establish infeasibility. Every cumulative original budget
        # and the caller's full execution contract must pass before adoption.
        caps = (max_length,)+limits
        profiles = [(10.,)*len(limits)]
        profiles.extend(tuple(100. if i == j else 10. for i in range(len(limits)))
                        for j in range(len(limits)))
        profiles.append((100.,)*len(limits))
        proposals = []
        for penalties in dict.fromkeys(profiles):
            proposal_started = time.monotonic()
            weights = (1.,)+penalties
            score_cap = sum(w*(cap+1e-9) for w, cap in zip(weights, caps))
            if not math.isfinite(score_cap):
                continue
            def weighted_cost(a, b):
                values = costs(a, b)
                if values is None or not dominates(values, caps):
                    return None
                return sum(w*v for w, v in zip(weights, values))
            def weighted_heuristic(node):
                return float(heuristic(node)) + (sum(p*b.get(node, math.inf)
                    for p, b in zip(penalties, bounds[1:])) if bounds else 0.)
            proposal, _ = shortest_physical_path(neighbors, start, end, weighted_cost,
                max_length=score_cap, heuristic=weighted_heuristic,
                edge_length_bound=edge_length_bound)
            values = None if proposal is None else path_values(proposal)
            feasible = values is not None and dominates(values, caps)
            accepted = feasible and feasible_path_accept(proposal)
            evidence = dict(penalties=list(penalties), seconds=time.monotonic()-proposal_started,
                outcome=('verified_feasible_route' if accepted else
                         'route_contract_rejected' if feasible else
                         'cumulative_budget_exceeded' if proposal is not None else 'no_weighted_proposal'),
                length=None if values is None else values[0],
                resources=None if values is None else list(values[1:]))
            proposals.append(evidence)
            if diagnostics is not None:
                diagnostics['feasible_route_proposals'] = proposals
            if progress_enabled:
                print(json.dumps(dict(event='native_resource_search_progress',
                    phase='feasible_route_proposal_done', **evidence)), flush=True)
            if accepted:
                if diagnostics is not None:
                    diagnostics.update(outcome='resource_bounded_path', length=values[0],
                        resources=list(values[1:]), resource_limits=list(limits),
                        expanded_labels=0, generated_labels=1, examined_edges=len(cache),
                        completion_pruned_labels=0, optimistic_frontier_pruned_edges=0,
                        shortest_path_required=False, **frontier_metrics())
                return proposal
        if verified_proposals_only:
            # This is a construction failure, not a proof of physical
            # infeasibility. The model may retry/continue from the real state.
            # No unverified route is returned or executed.
            if diagnostics is not None:
                diagnostics.update(outcome='verified_proposals_exhausted',
                    search_complete=False, physical_feasibility='unknown',
                    expanded_labels=0, generated_labels=1, examined_edges=len(cache),
                    resource_limits=list(limits), **frontier_metrics())
            return None
        # The default still preserves the original complete search.

    if use_lazy_resource_minima and bounds and completion_edge_bound is not None:
        # Independent resource minima remain necessary conditions, never a
        # claim of joint feasibility. Reuse sparse scalar Dijkstra on the
        # optimistic graph, then measure its unknown edges and repeat.
        # Each nonterminal pass resolves at least one previously unknown edge;
        # no iteration cap, cost rounding or approximate rejection is used.
        minima = []
        caps = (max_length,)+limits
        minimum_graph = _ResourceMinimumGraph(neighbors, start, end, bound_costs, caps)
        for component, cap in enumerate(limits, 1):
            minimum_started = time.monotonic()
            iterations = 0
            before_edges = len(cache)
            while True:
                iterations += 1
                optimistic, minimum_cost = minimum_graph.shortest(component, cap)
                if optimistic is None:
                    minimum = dict(component=component, cap=cap, iterations=iterations,
                        outcome='minimum_exceeds_budget_or_disconnected',
                        measured_edges=len(cache)-before_edges,
                        seconds=time.monotonic()-minimum_started)
                    minima.append(minimum)
                    if diagnostics is not None:
                        diagnostics.update(lazy_resource_minima=minima,
                            outcome='no_resource_bounded_path', expanded_labels=0,
                            generated_labels=1, examined_edges=len(cache),
                            resource_limits=list(limits), completion_pruned_labels=1,
                            optimistic_frontier_pruned_edges=0, **frontier_metrics())
                    if progress_enabled:
                        print(json.dumps(dict(event='native_resource_search_progress',
                            phase='lazy_resource_minimum_done', **minimum)), flush=True)
                    return None
                unresolved = [(a, b) for a, b in zip(optimistic, optimistic[1:])
                              if (a, b) not in cache]
                if not unresolved:
                    values = path_values(optimistic)
                    minimum = dict(component=component, cap=cap, iterations=iterations,
                        outcome='verified_independent_minimum', minimum=minimum_cost,
                        measured_edges=len(cache)-before_edges,
                        path_length=values[0], path_resources=list(values[1:]),
                        seconds=time.monotonic()-minimum_started)
                    minima.append(minimum)
                    if progress_enabled:
                        print(json.dumps(dict(event='native_resource_search_progress',
                            phase='lazy_resource_minimum_done', **minimum)), flush=True)
                    if (feasible_path_accept is not None and dominates(values, caps)
                            and feasible_path_accept(optimistic)):
                        if diagnostics is not None:
                            diagnostics.update(lazy_resource_minima=minima,
                                outcome='resource_bounded_path', length=values[0],
                                resources=list(values[1:]), resource_limits=list(limits),
                                expanded_labels=0, generated_labels=1, examined_edges=len(cache),
                                completion_pruned_labels=0, optimistic_frontier_pruned_edges=0,
                                shortest_path_required=False, **frontier_metrics())
                        return optimistic
                    break
                for a, b in unresolved:
                    minimum_graph.measured(a, b, costs(a, b))
                if progress_enabled and time.monotonic() >= next_progress:
                    print(json.dumps(dict(event='native_resource_search_progress',
                        phase='lazy_resource_minimum', component=component, iterations=iterations,
                        examined_edges=len(cache), seconds=time.monotonic()-minimum_started)), flush=True)
                    next_progress = time.monotonic()+15.
        # Use all newly proved costs for nodewise completion bounds as well.
        bounds = _completion_bounds(neighbors, end, bound_costs, caps)
        if diagnostics is not None:
            diagnostics.update(lazy_resource_minima=minima,
                               completion_bound_nodes=[len(b) for b in bounds])
        if not completion_possible(start, 0., (0.,)*len(limits)):
            if diagnostics is not None:
                diagnostics.update(outcome='no_resource_bounded_path', expanded_labels=0,
                    generated_labels=1, examined_edges=len(cache), resource_limits=list(limits),
                    completion_pruned_labels=1, optimistic_frontier_pruned_edges=0,
                    **frontier_metrics())
            return None

    while queue:
        _, label_id = heapq.heappop(queue)
        node, distance, used, previous = labels[label_id]
        if not live_labels[label_id]:
            continue
        expanded += 1
        if progress_enabled and expanded % 1024 == 0 and time.monotonic() >= next_progress:
            print(json.dumps(dict(event='native_resource_search_progress', phase='forward',
                seconds=time.monotonic()-started, expanded_labels=expanded,
                generated_labels=len(labels), examined_edges=len(cache), queue_labels=len(queue),
                **frontier_metrics())), flush=True)
            next_progress = time.monotonic() + 15.
        if node == end:
            path = []
            cursor = label_id
            while cursor is not None:
                at, _, _, cursor = labels[cursor]
                path.append(at)
            if diagnostics is not None:
                diagnostics.update(outcome='resource_bounded_path', expanded_labels=expanded,
                    generated_labels=len(labels), examined_edges=len(cache), length=distance,
                    resources=list(used), resource_limits=list(limits), completion_pruned_labels=pruned,
                    optimistic_frontier_pruned_edges=optimistic_pruned, **frontier_metrics())
            return path[::-1]
        for target in sorted(neighbors.get(node, ())):
            if edge_length_bound is not None and (node, target) not in cache:
                key = (node, target)
                if key not in length_bounds:
                    bound = float(edge_length_bound(node, target))
                    if not math.isfinite(bound) or bound < 0:
                        raise ValueError('edge length bound must be finite and nonnegative')
                    length_bounds[key] = bound
                lower_distance = distance+length_bounds[key]
                arrivals = active.get(target)
                # The native caller's Euclidean edge length is exact; the
                # unmeasured resources have the universal lower bound zero.
                # An existing label that dominates this optimistic arrival
                # necessarily dominates any nonnegative measured arrival.
                if (lower_distance > max_length+1e-9 or (arrivals is not None
                        and arrivals.contains_dominator((lower_distance,)+used))):
                    optimistic_pruned += 1
                    continue
            values = costs(node, target)
            if values is None:
                continue
            next_distance = distance+values[0]
            next_used = tuple(a+b for a, b in zip(used, values[1:]))
            if next_distance > max_length+1e-9 or not dominates(next_used, limits):
                continue
            if not completion_possible(target, next_distance, next_used):
                pruned += 1
                continue
            candidate = (next_distance,)+next_used
            arrivals = active.get(target)
            if arrivals is None:
                arrivals = active[target] = _ParetoFrontier(len(limits)+1)
            next_id = len(labels)
            accepted, removed = arrivals.insert(candidate, next_id)
            if not accepted:
                continue
            for stale_id in removed:
                live_labels[stale_id] = False
            labels.append((target, next_distance, next_used, label_id))
            live_labels.append(True)
            heapq.heappush(queue, (next_distance+float(heuristic(target)), next_id))
    if diagnostics is not None:
        diagnostics.update(outcome='no_resource_bounded_path', expanded_labels=expanded,
            generated_labels=len(labels), examined_edges=len(cache),
            reached_nodes=sum(bool(v) for v in active.values()), resource_limits=list(limits),
            completion_pruned_labels=pruned,
                    optimistic_frontier_pruned_edges=optimistic_pruned, **frontier_metrics())
    return None

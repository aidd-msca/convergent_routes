import networkx as nx
import numpy as np

import itertools
import random
from tqdm import tqdm
import collections
from networkx.readwrite import json_graph
import json
import time
import copy
import logging

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def route_scorer(graph, pred_nodes):
    pred_nodes = set(n for e in pred_nodes for n in e)
    graph = graph.subgraph(pred_nodes)
    targetm = [n for n in nx.get_node_attributes(graph, "target")]
    score = []
    for tm in targetm:
        T = nx.dfs_tree(graph, source=tm)
        scr = [
            graph.nodes[n]["model_p"] for n in T.nodes() if "model_p" in graph.nodes[n]
        ]
        score.append(np.prod(scr))
    if len(score) > 0:
        score = sum(score) / len(score)
        return score
    else:
        return 0


class ConvergentRoutes:
    def __init__(self, max_len_route=None, keep_max_len=False) -> None:
        if max_len_route is not None:
            self.max_path_len = (max_len_route * 2) - 1
        self.keep_max_len = keep_max_len
        self.log = logging.getLogger("convergentsearch")

    def check_route(self, edges, convergent_search):
        if edges == []:
            return False
        multipred = collections.defaultdict(set)
        all_nodes = []
        for src, tgt in edges:
            all_nodes.append(src)
            all_nodes.append(tgt)
            if type(tgt) == str:
                multipred[tgt].add(src)
                if len(multipred[tgt]) > 1:
                    return False

        sg = convergent_search.subgraph(all_nodes).copy()

        for n in sg.nodes():
            if len(sg.out_edges(n)) == 0:
                if "buildingblock" not in sg.nodes[n]:
                    return False
        return True

    def get_nodes(self, edges):
        return [n for e in edges for n in e]

    def check_issues(self, nodes, graph, connections=None):
        sg = graph.subgraph(nodes)

        issue = False
        for n in sg.nodes():
            if len(sg.out_edges(n)) == 0:
                if type(n) == str:
                    if "buildingblock" not in sg.nodes[n]:
                        issue = True
                if issue:
                    return issue
            if len(sg.in_edges(n)) > 1:
                for n_ in sg.nodes():
                    i = 0
                    for path in nx.all_simple_paths(sg, source=n_, target=n):
                        i += 1
                        if i > 1:
                            issue = True
                            return issue

        return issue

    def prune_graph(self, graph):

        complete = False
        graph = graph.copy()
        if not self.keep_max_len:
            incomplete_paths = [
                list(graph.predecessors(n))
                for n in graph.nodes
                if (graph.out_degree(n) == 0) & ("buildingblock" not in graph.nodes[n])
            ]
        else:
            incomplete_paths = [
                list(graph.predecessors(n))
                for n in graph.nodes
                if (graph.out_degree(n) == 0)
                & (
                    ("buildingblock" not in graph.nodes[n])
                    or ("maxlen" not in graph.nodes[n])
                )
            ]

        incomplete_paths = [
            n for pth in incomplete_paths for n in pth if type(n) == int
        ]

        incomplete_paths.extend(
            [
                n
                for n in graph.nodes
                if (graph.out_degree(n) == 0)
                & ("buildingblock" not in graph.nodes[n])
                & (type(n) == int)
            ]
        )
        if not incomplete_paths:
            complete = True
            return graph, complete
        graph.remove_nodes_from(incomplete_paths)
        graph.remove_nodes_from(list(nx.isolates(graph)))

        rxn_nodes = []
        duplicated_predictions = []
        for node in graph.nodes:
            if type(node) == int:
                rxn = set(n for n in graph.successors(node))
                rxn.update(set(n for n in graph.predecessors(node)))
                if rxn not in rxn_nodes:
                    rxn_nodes.append(rxn)
                else:
                    duplicated_predictions.append(node)
        graph.remove_nodes_from(duplicated_predictions)

        return graph, complete

    def route_complete(self, nodes, graph):
        route = graph.subgraph(nodes)
        route = nx.to_numpy_array(route, nodelist=nodes)
        str_nodes = [True if type(n) == str else False for n in nodes]
        route = route[str_nodes]
        return np.all(route.sum(axis=1) <= 1)

    def traverse_graph(self, orig_graph, target_molecules, max_time_pm):

        orig_graph_pruned = orig_graph.copy()
        complete = False
        while not complete:
            orig_graph_pruned, complete = self.prune_graph(orig_graph_pruned)
        graph = orig_graph_pruned.copy()

        rm_tm = [tm for tm in target_molecules if tm not in graph]
        if len(rm_tm) == len(target_molecules):
            return None
        elif len(rm_tm) > 0:
            for tm in rm_tm:
                target_molecules.remove(tm)

        sparse_array_o = nx.adjacency_matrix(graph)

        nodes = list(graph.nodes())

        i = 0
        max_i = 1000
        max_time = 1
        sparse_array = sparse_array_o.copy()
        routes = []

        origins = collections.defaultdict(set)
        for n in nodes:
            if type(n) == int:
                for tm in target_molecules:
                    if nx.has_path(graph, tm, n):
                        origins[n].add(tm)
        tm_p = np.array([1 for tm in target_molecules])
        start_time_overall = time.time()
        max_time_overall = max_time_pm * len(target_molecules)
        while (i < max_i) and (time.time() < start_time_overall + max_time_overall):
            self.log.debug(f"Iteration - {i}", terminator="\r")

            def get_targetmolecule_prob(tm_p):
                p = tm_p - (tm_p.max() + 1)
                return p / p.sum()

            rand_cpd = np.random.choice(
                target_molecules, size=1, p=get_targetmolecule_prob(tm_p)
            ).tolist()[0]
            tm_p[target_molecules.index(rand_cpd)] += 1
            rt_i = []
            orig = True
            previously_seen = []
            pending_explore = [rand_cpd]
            additional_paths = set([rand_cpd])
            next_cpds = True
            sparse_array = sparse_array_o.copy()
            start_time = time.time()

            while next_cpds and (time.time() < start_time + max_time):
                rt_i.append(rand_cpd)
                pending_explore.remove(rand_cpd)
                ar = sparse_array[[nodes.index(rand_cpd)]]
                if len(ar.indices) == 0:
                    if (len(pending_explore) == 0) & (len(additional_paths) == 0):
                        next_cpds = False
                    elif (len(pending_explore) == 0) & (len(additional_paths) > 0):
                        next_cpds = False
                        continue

                    else:
                        rand_cpd = np.random.choice(pending_explore, size=1).tolist()[0]
                    continue

                def norm_probs(p):
                    p = [p_ / sum(p) if p_ not in previously_seen else 0 for p_ in p]
                    return p

                if not orig:
                    p = [
                        (
                            graph.nodes[nodes[a]]["model_p"]
                            if (a not in previously_seen)
                            and (nodes[a] not in rt_i)
                            and (len(origins[nodes[a]].difference(set(rt_i))) > 0)
                            else 0
                        )
                        for a in ar.indices
                    ]
                else:
                    p = [
                        (
                            graph.nodes[nodes[a]]["model_p"]
                            if (a not in previously_seen)
                            else 0
                        )
                        for a in ar.indices
                    ]

                if sum(p) == 0:
                    next_cpds = False
                    continue

                try:
                    random_pred = np.random.choice(
                        ar.indices, size=1, p=norm_probs(p)
                    ).tolist()[0]
                except:
                    self.log.warning("Could not select random sample", ar.indices, norm_probs(p))
                    next_cpds = False
                    continue

                if orig:
                    previously_seen.extend(ar.indices)
                    previously_seen.remove(random_pred)

                rt_i.append(rand_cpd)
                random_pred = nodes[random_pred]
                rt_i.append(random_pred)

                ar = sparse_array[[nodes.index(random_pred)]]
                rand_cpds = [nodes[a] for a in ar.indices]
                pending_explore.extend(rand_cpds)
                additional_paths.update(set(rand_cpds))

                if len(pending_explore) == 0:
                    rand_cpd = np.random.choice(
                        list(additional_paths), size=1
                    ).tolist()[0]
                    additional_paths.remove(rand_cpd)
                    pending_explore.append(rand_cpd)
                else:
                    rand_cpd = np.random.choice(pending_explore, size=1).tolist()[0]

            routes.append(rt_i)
            i += 1

        red_search = graph.subgraph([r_ for r in routes for r_ in r])

        return red_search

    def select_path(self, graph, target_molecules, node, edges, return_score=False):
        found_tm = set()
        scores = []

        for tm in target_molecules:
            if tm not in graph:
                continue
            score = 0
            max_path_len = self.max_path_len
            edge = []

            edge_counter = {}
            for src, tgt in edges:
                if type(src) == str:
                    edge_counter[src] = tgt

            if not nx.has_path(graph, tm, node):
                continue

            if type(tm) == int:
                tgts = list(nx.get_node_attributes(graph, "target").keys())
                tm_depth = min(
                    nx.shortest_path_length(graph, tgt, tm)
                    for tgt in tgts
                    if nx.has_path(graph, tgt, tm)
                )
                max_path_len = max_path_len - tm_depth

            simple_paths = list(
                nx.all_simple_edge_paths(graph, tm, node, cutoff=max_path_len)
            )
            simple_paths_idx = np.array(list(range(len(simple_paths))))
            if len(simple_paths) == 0:
                scores.append(score)
                continue

            def calc_score(e):
                n = set(
                    graph.nodes[n]["model_p"]
                    for n in self.get_nodes(e)
                    if type(n) == int
                )
                scr = np.prod(list(n))
                return scr

            scr = np.array([calc_score(p) for p in simple_paths])

            simple_paths_idx = simple_paths_idx[scr > 0]
            scr = scr[scr > 0]
            scores_idx = np.argsort(scr)[::-1]
            simple_paths_idx = simple_paths_idx[scores_idx]
            simple_paths = [simple_paths[i] for i in simple_paths_idx[:10]]
            random.shuffle(simple_paths)

            for e in simple_paths:
                n = set(
                    graph.nodes[n]["model_p"]
                    for n in self.get_nodes(e)
                    if type(n) == int
                )
                scr = np.prod(list(n))

                if not return_score:
                    repeated = False
                    for src, tgt in e:
                        if src in edge_counter:
                            if edge_counter[src] != tgt:
                                repeated = True
                                break

                    if repeated:
                        continue
                edge = e
                score = scr
                if not return_score:
                    found_tm.add(tm)
                break
            if not return_score:
                edges.extend(edge)
            scores.append(score)
        if not return_score:
            return edges, found_tm
        else:
            return scores

    def parse_graph(self, graph, target_molecules, max_time_pm):
        if graph is None:
            return []
        end_nodes = list(nx.get_node_attributes(graph, "buildingblock").keys())
        if self.keep_max_len:
            en = [
                n
                for n in nx.get_node_attributes(graph, "maxlen").keys()
                if n not in end_nodes
            ]
            end_nodes.extend(en)
        end_nodes = np.array(end_nodes)
        target_molecules_orig = copy.deepcopy(target_molecules)

        routes = []
        searched = []
        scores = []
        for n in tqdm(end_nodes):
            scores.append(
                sum(self.select_path(graph, target_molecules, n, [], return_score=True))
                / len(target_molecules)
            )
        scores = np.array(scores)
        end_nodes_filtered = end_nodes[scores > 0]
        scores = scores[scores > 0]
        scores_idx = np.argsort(scores)[::-1]

        start_time = time.time()
        max_time = max_time_pm * len(target_molecules)
        routes = []
        for i in tqdm(range(100)):
            for node in end_nodes_filtered[scores_idx]:
                edges = []

                stack = [[edges, set(target_molecules_orig), node]]
                searched = []
                while stack and (time.time() < start_time + max_time):
                    edges, target_molecules, node = stack[-1]

                    edges, found_tm = self.select_path(
                        graph, target_molecules, node, edges
                    )
                    searched.append(node)

                    target_molecules.difference_update(found_tm)
                    pred_nodes = set(n for n in self.get_nodes(edges) if type(n) == int)

                    pred_nodes_cnt = collections.defaultdict(int)

                    for src, tgt in set(edges):
                        if type(src) == int:
                            pred_nodes_cnt[src] += 1

                    pred_nodes = set(
                        pn
                        for pn in pred_nodes
                        if pred_nodes_cnt[pn] < graph.out_degree(pn)
                    )

                    for p in pred_nodes:
                        target_molecules.add(p)
                        pths = nx.shortest_path(graph, p)
                        nodes = [
                            n
                            for n in end_nodes_filtered
                            if (n not in searched) and (n in pths)
                        ]

                        max_score = scores[np.isin(end_nodes_filtered, nodes)]

                        if max_score.size == 0:
                            continue
                        max_score = np.argmax(max_score)
                        node = nodes[max_score]

                        stack.append([edges, target_molecules, node])
                        break
                    else:
                        stack.pop()
                if edges:
                    routes.append(edges)
        return routes

    def create_convergent_route(self, edges, score, target_molecules):
        route = nx.DiGraph(score=score)

        mol_nodes = set(n for e in edges for n in e if type(n) != int)

        route.add_nodes_from(mol_nodes)
        for tm in target_molecules:
            if tm in route.nodes():
                route.nodes[tm]["target"] = True

        for a, b in itertools.combinations(edges, 2):
            if (a[-1] == b[0]) and (type(a[0]) != int) and (type(b[-1]) != int):
                route.add_edge(a[0], b[-1])
            if (a[0] == b[-1]) and (type(a[-1]) != int) and (type(b[0]) != int):
                route.add_edge(b[0], a[-1])
        routes = []
        for r in nx.weakly_connected_components(route):
            routes.append(route.subgraph(r).copy())

        return routes

    def store_graphs(self, graphs, store_fp):
        graphs = [json_graph.adjacency_data(graph) for graph in graphs]
        s = json.dumps(graphs)
        with open(store_fp, "w") as f:
            f.write(s)

    def get_routes(
        self,
        convergent_search,
        target_molecules,
        route_scorer=route_scorer,
        i=0,
        store_fp=None,
        max_time_pm=300,
    ):

        reduced_convergent_search = self.traverse_graph(
            convergent_search, target_molecules, max_time_pm
        )

        routes = self.parse_graph(
            reduced_convergent_search, target_molecules, max_time_pm
        )

        self.log.info(f"Identified {len(routes)} routes")

        ranked_routes = {}
        parsed = []
        self.log.info("Processing routes")

        for r in routes:
            score = route_scorer(convergent_search, r)
            for cr in self.create_convergent_route(r, score, target_molecules):
                if sorted(list(cr.edges())) in parsed:
                    continue
                parsed.append(sorted(list(cr.edges())))
                ranked_routes[cr] = score
        self.log.info(f"Identified {len(ranked_routes)} unique routes")
        if store_fp is not None:
            self.store_graphs(ranked_routes.keys(), store_fp)

        return ranked_routes

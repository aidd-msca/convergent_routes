import networkx as nx
import json
import collections
import pandas as pd
import numpy as np
from tqdm import tqdm
from networkx.readwrite import json_graph
import os
import logging

class ConvergentEvaluation:
    def __init__(self, ytr_routes, experiment_name):
        self.results = collections.defaultdict(list)
        self.individual_results = collections.defaultdict(list)
        self.search_results = collections.defaultdict(list)
        self.ytr_routes = ytr_routes
        self.routes_store = []
        self.experiment_name = experiment_name
        self.log = logging.getLogger("convergentsearch")

    def load_truth(self, fp):
        with open(fp, "r") as f:
            routes = json.load(f)
        return routes

    def load_truths(self, fp):
        pass

    def evaluate_search(self, rt, targetm, i):
        self.overlap_targetmolecules.update(list(rt))
        self.results[f"solvedtargetmolecules_top{i+1}"].append(
            len(set.intersection(set(targetm), self.overlap_targetmolecules))
        )

    def evaluate_route(self, rt, scr, route_tr, i, targetm, buildingblocks):
        route_found = False
        if (sorted(rt.nodes()) == sorted(route_tr.nodes())) and (
            sorted(rt.edges()) == sorted(route_tr.edges())
            and (not self.route_found_overall)
        ):
            self.log.info(f"Route found at top-{i + 1}")
            route_found = True
            self.route_found = True
        self.results[f"route_top{i+1}"].append(route_found)

        intermediates_pred = [n for n in rt.nodes() if len(rt.in_edges(n)) > 1]
        intermediates_ytr = [
            n for n in route_tr.nodes() if len(route_tr.in_edges(n)) > 1
        ]

        intermediates_found = False
        if sorted(intermediates_pred) == sorted(intermediates_ytr):
            intermediates_found = True
        self.results[f"intermediates_top{i+1}"].append(intermediates_found)

        self.results[f"score_top{i+1}"].append(scr)

        tp = len(set.intersection(set(route_tr.edges), set(rt.edges())))
        fp = len(set(rt.edges).difference(set(route_tr.edges())))
        fn = len(set(route_tr.edges).difference(set(rt.edges())))
        f1_edges = (2 * tp) / ((2 * tp) + fp + fn)
        self.results[f"modifiedf1_top{i+1}"].append(f1_edges)

        route_tr_nodes = set(route_tr.nodes)
        route_tr_nodes.difference_update(set(targetm))
        rt_nodes = set(rt.nodes)
        rt_nodes.difference_update(set(targetm))

        tp = len(set.intersection(route_tr_nodes, rt_nodes))
        fp = len(rt_nodes.difference(route_tr_nodes))
        fn = len(route_tr_nodes.difference(rt_nodes))
        f1_nodes = (2 * tp) / ((2 * tp) + fp + fn)
        self.results[f"modifiedf1nodes_top{i+1}"].append(f1_nodes)
        self.results[f"modifiedf1combined_top{i+1}"].append((f1_edges + f1_nodes)/2)

        self.results[f"ntargetmols_top{i+1}"].append(
            sum(1 for n in rt.nodes() if n in targetm)
        )
        self.results[f"nbuildingblocks_top{i+1}"].append(
            sum(1 for n in rt.nodes() if n in buildingblocks)
        )

    def evaluate_individual_route(self, rt, route_tr, tm, i):
        rt = nx.dfs_tree(rt, tm)
        buildingblocks = [
            n for n in route_tr.nodes() if len(route_tr.out_edges(n)) == 0
        ]
        route_found = False

        if (sorted(rt.nodes()) == sorted(route_tr.nodes())) and (
            sorted(rt.edges()) == sorted(route_tr.edges())
            and (not self.route_found_indiv)
        ):
            self.log.info(f"Route found at top-{i + 1}")
            route_found = True
            self.route_found_indiv = True
        self.individual_results[f"route_top{i+1}"].append(route_found)

        tp = len(set.intersection(set(route_tr.edges), set(rt.edges())))
        fp = len(set(rt.edges).difference(set(route_tr.edges())))
        fn = len(set(route_tr.edges).difference(set(rt.edges())))
        tn = 0
        self.individual_results[f"modifiedf1_top{i+1}"].append(
            (2 * tp) / ((2 * tp) + fp + fn)
        )

        tp = len(set.intersection(set(route_tr.nodes), set(rt.nodes())))
        fp = len(set(rt.nodes).difference(set(route_tr.nodes())))
        fn = len(set(route_tr.nodes).difference(set(rt.nodes())))
        tn = 0
        self.individual_results[f"modifiedf1nodes_top{i+1}"].append(
            (2 * tp) / ((2 * tp) + fp + fn)
        )

        self.individual_results[f"nbuildingblocks_top{i+1}"].append(
            sum(1 for n in rt.nodes() if n in buildingblocks)
        )

    def prep_results(self, i):
        for j in range(i + 1, 1000):
            self.results[f"route_top{j+1}"].append(np.nan)
            self.results[f"intermediates_top{j+1}"].append(np.nan)
            self.results[f"score_top{j+1}"].append(np.nan)
            self.results[f"modifiedf1_top{j+1}"].append(np.nan)
            self.results[f"modifiedf1nodes_top{j+1}"].append(np.nan)
            self.results[f"modifiedf1combined_top{j+1}"].append(np.nan)
            self.results[f"ntargetmols_top{j+1}"].append(np.nan)
            self.results[f"nbuildingblocks_top{j+1}"].append(np.nan)
            self.results[f"solvedtargetmolecules_top{j+1}"].append(np.nan)

    def prep_results_inidvidual(self, i):
        for j in range(i + 1, 1000):
            self.individual_results[f"route_top{j+1}"].append(np.nan)
            self.individual_results[f"modifiedf1_top{j+1}"].append(np.nan)
            self.individual_results[f"modifiedf1nodes_top{j+1}"].append(np.nan)
            self.individual_results[f"nbuildingblocks_top{j+1}"].append(np.nan)

    def evaluate_convergent_search(self, routes, route_tr, target_molecules):
        if routes == {}:
            i = -1
            self.results[f"ntargetmolecules"].append(np.nan)
            self.results[f"nbuildingblocks"].append(np.nan)

            self.results[f"longest_path"].append(np.nan)

            self.results[f"n_routes"].append(np.nan)
            self.prep_results(i)
            return
        route_tr = nx.adjacency_graph(route_tr)
        buildingblocks = [
            n for n in route_tr.nodes() if len(route_tr.out_edges(n)) == 0
        ]
        self.results[f"ntargetmolecules"].append(len(target_molecules))
        self.results[f"nbuildingblocks"].append(len(buildingblocks))

        self.results[f"longest_path"].append(
            max(
                len(path) - 1
                for p in target_molecules
                for path in nx.all_simple_paths(
                    route_tr, source=p, target=buildingblocks
                )
            )
        )
        self.results[f"n_routes"].append(len(routes))
        i = -1
        self.overlap_targetmolecules = set()
        self.route_found_overall = False
        for i, (rt, scr) in tqdm(
            enumerate(
                {
                    k: v
                    for k, v in sorted(
                        routes.items(),
                        key=lambda item: item[1]
                        * sum(
                            1 for n in item[0].nodes if len(item[0].in_edges(n)) == 0
                        ),
                        reverse=True,
                    )
                }.items()
            )
        ):
            if i == 1000:
                break
            self.evaluate_route(rt, scr, route_tr, i, target_molecules, buildingblocks)
            self.evaluate_search(rt, target_molecules, i)
        self.prep_results(i)

        for tm in target_molecules:
            j = -1

            route_tr_tm = nx.dfs_tree(route_tr, tm)
            buildingblocks = [
                n for n in route_tr_tm.nodes() if len(route_tr_tm.out_edges(n)) == 0
            ]

            self.individual_results[f"longest_path"].append(
                max(
                    len(path) - 1
                    for path in nx.all_simple_paths(
                        route_tr_tm, source=tm, target=buildingblocks
                    )
                )
            )
            self.route_found_indiv = False
            for i, (rt, scr) in tqdm(
                enumerate(
                    {
                        k: v
                        for k, v in sorted(
                            routes.items(), key=lambda item: item[1], reverse=True
                        )
                    }.items()
                )
            ):
                if i == 1000:
                    break
                if tm in rt:
                    j += 1
                    self.evaluate_individual_route(rt, route_tr_tm, tm, j)

            self.prep_results_inidvidual(j)

    def run_evaluation(self, predicted_routes, ytr_idx=None, target_molecules=None):
        self.log.info("Initializing route evaluation")
        if type(self.ytr_routes) == str:
            self.ytr_routes = self.load_truth(self.ytr_routes)
        if ytr_idx is None:
            for route, ytr in zip(predicted_routes, self.ytr_routes):
                self.evaluate_convergent_search(route, ytr, target_molecules)
        else:
            self.evaluate_convergent_search(
                predicted_routes, self.ytr_routes[ytr_idx], target_molecules
            )

    def store_results(self, results_fp):
        results = pd.DataFrame.from_dict(self.results)
        individual_results = pd.DataFrame.from_dict(self.individual_results)
        if results_fp is not None:
            results.to_csv(
                os.path.join(results_fp, f"test_results_{self.experiment_name}.tsv"),
                sep="\t",
                index=False,
            )
            individual_results.to_csv(
                os.path.join(
                    results_fp, f"test_results_individual_{self.experiment_name}.tsv"
                ),
                sep="\t",
                index=False,
            )

    def store_route(self, predicted_routes, n):
        predicted_routes = [
            k
            for k, v in sorted(
                predicted_routes.items(), key=lambda item: item[1], reverse=True
            )
        ]

        self.routes_store.append(
            [json_graph.adjacency_data(g_) for g_ in predicted_routes[:n]]
        )

    def store_routes(self, store_fp):
        s = json.dumps(self.routes_store)
        with open(
            os.path.join(store_fp, f"routes_{self.experiment_name}.json"), "w"
        ) as f:
            f.write(s)

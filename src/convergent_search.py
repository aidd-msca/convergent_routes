from ssbenchmark.model_zoo import ModelZoo
from ssbenchmark.ssmodels.model_chemformer import model_chemformer

import networkx as nx
from utils import canonicalize_smiles
import time
import logging


def ranker(nodes, k):
    nodes = {n: d["p"] for n, d in nodes.items() if "p" in d}
    nxt_cpd = sorted(nodes.items(), key=lambda item: item[1], reverse=True)[:k]
    nxt_cpd, _ = zip(*nxt_cpd)
    return nxt_cpd


def scorer(graph, n, target_molecules):
    paths = [
        nx.all_shortest_paths(graph, tm, n)
        for tm in target_molecules
        if nx.has_path(graph, tm, n)
    ]
    score = []
    len_path = 1000
    for pth in paths:
        score_tm = 0
        for p in pth:
            if len(p) == 0:
                continue
            s = 1

            for n in p:
                if "model_p" in graph.nodes[n]:
                    s *= graph.nodes[n]["model_p"]

            if s > score_tm:
                score_tm = s
            if len(p) < len_path:
                len_path = len(p)
        score.append(score_tm)
    if len(score) == 0:
        score = 0
    else:
        score = sum(score) / len(
            score
        )  # Average across all targetmolecules, singular path routes potentially more likely

    return score, len_path


class ConvergentSearch:
    def __init__(
        self,
        buildingblocks,
        k,
        single_step_model,
        single_step_module_path,
        single_step_use_gpu,
        single_step_settings,
        max_time=60 * 90,
        max_iterations=200,
        ranker=ranker,
        scorer=scorer,
        max_len_route=10,
        target_molecule_limit=300,
    ) -> None:
        self.convergent_search = nx.DiGraph()

        self.buildingblocks = buildingblocks
        self.k = k  # How many compounds to explore per call
        self.ranker = ranker
        self.scorer = scorer
        self.max_time = max_time
        self.max_iterations = max_iterations
        self.max_len_route = (max_len_route * 2) - 1

        self.model = ModelZoo(
            single_step_model,
            single_step_module_path,
            single_step_use_gpu,
            single_step_settings,
        )

        self.log = logging.getLogger("convergentsearch")

        self.log.info(
            f"Running search: \n \
                Explore {self.k} compounds per iteration with a maximum route length of {max_len_route}"
        )

        self.target_molecule_limit = target_molecule_limit

    def set_k(self):
        if self.k is not None:
            return self.k
        else:
            return len(self.target_molecules) * 4

    def setup(self, target_molecules):
        if len(target_molecules) > self.target_molecule_limit:
            self.log.warn(
                f"Search contains {len(target_molecules)} molecules, skipping search"
            )
            self.valid = False
        else:
            self.convergent_search.add_nodes_from(target_molecules, target=True)
            self.target_molecules = target_molecules
            self.prediction_counter = 0
            self.log.info(f"Search set up for {len(target_molecules)} molecules")
            self.valid = True

    def clear_search(self):
        self.convergent_search.clear()

    def update_scores(self, reactants):
        for rct in reactants:
            score, len_path = self.scorer(
                self.convergent_search, rct, self.target_molecules
            )
            self.convergent_search.nodes[rct]["p"] = score
            if rct in self.buildingblocks:
                self.convergent_search.nodes[rct]["buildingblock"] = True
            if len_path >= self.max_len_route:
                self.convergent_search.nodes[rct]["maxlen"] = True

    def clean_cycles(self):
        for c in nx.simple_cycles(self.convergent_search):
            if type(c[-1]) == str:
                continue
            if self.convergent_search.has_edge(c[-1], c[0]):
                self.convergent_search.remove_edge(c[-1], c[0])

    def setup_oneiteration(self):
        nodes = {
            n: d
            for n, d in self.convergent_search.nodes.items()
            if ("p" in d)
            and (("buildingblock" not in d) & ("maxlen" not in d))
            and self.convergent_search.out_degree(n) == 0
        }
        if len(nodes) == 0:
            return None
        nxt_cpd = self.ranker(nodes, self.k)
        return nxt_cpd

    def get_edges_probabilites(self, cpds, reactants, model_p):
        reactants_canon = [[canonicalize_smiles(r_) for r_ in r] for r in reactants]
        molecule_nodes = []
        probabilities = {}
        prediction_nodes = []
        rcts = []
        for i, (c, p, model_prb) in enumerate(zip(cpds, reactants_canon, model_p)):
            added_reactants = []
            for j, (p_, prb) in enumerate(zip(p, model_prb)):
                if p_ is None:
                    continue
                if p_ in added_reactants:
                    continue
                if sum(1 for rp in p_.split(".") if rp == c) == 0:
                    prediction_nodes.append((c, self.prediction_counter))
                    probabilities[self.prediction_counter] = prb
                    mn = []
                    for rp in p_.split("."):
                        if rp in self.convergent_search:
                            if nx.has_path(self.convergent_search, rp, c):
                                mn = []
                                break
                        mn.append((self.prediction_counter, rp))

                    molecule_nodes.extend(mn)
                    rcts.extend([rp for p, rp in mn])
                    self.prediction_counter += 1
                    added_reactants.append(p_)

        return rcts, molecule_nodes, probabilities, prediction_nodes

    def run_oneiteration(self, cpds, iteration):
        reactants, model_p = self.model.model_call(cpds)
        (
            reactants,
            molecule_nodes,
            probabilities,
            prediction_nodes,
        ) = self.get_edges_probabilites(cpds, reactants, model_p)
        self.convergent_search.add_edges_from(prediction_nodes)
        nx.set_node_attributes(self.convergent_search, probabilities, "model_p")
        self.convergent_search.add_edges_from(molecule_nodes)
        self.clean_cycles()
        self.update_scores(reactants)

    def run_search(self, batch_size=32):
        start_time = time.time()
        iteration = 0
        self.run_oneiteration(self.target_molecules, iteration)
        max_time = self.max_time * len(self.target_molecules)
        max_iterations = max(
            (self.max_iterations * len(self.target_molecules)) / batch_size,
            (self.max_len_route + 1) / 2,
        )

        self.log.info(
            f"Search settings: {int(max_time)} seconds, {int(max_iterations)} iterations"
        )
        while (time.time() < start_time + max_time) & (iteration < max_iterations - 1):
            iteration += 1
            self.log.info(f"Iteration {iteration}")
            cpds = self.setup_oneiteration()
            if cpds is None:
                break
            self.run_oneiteration(cpds, iteration)

        return self.convergent_search

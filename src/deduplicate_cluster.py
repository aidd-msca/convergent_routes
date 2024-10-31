import networkx as nx
import os

import json

from tqdm import tqdm
from networkx.readwrite import json_graph

from rdkit import Chem
from convergent_routes.src.utils import get_route_descriptors, check_buildingblocks
import networkx as nx
import json

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import numpy as np

from tqdm import tqdm

import logging

import hydra


def check_nodes(gA, gB):
    return sorted(list(gA.nodes())) == sorted(list(gB.nodes()))


def gather_files(root, log=None):
    routes = []
    for f in os.listdir(root):
        assert f.endswith(".json")
        fp = os.path.join(root, f)
        if not os.path.exists(fp):
            if log is not None:
                log.info(f"File Not Found: {fp}")
            continue
        with open(fp, "r") as f:
            routes.extend(json.load(f))
    return routes


def identify_duplicated(routes):
    nodes_all = []
    edges_all = []
    equal_subgraphs = []

    for rn, g in tqdm(enumerate(routes)):
        if type(g) == dict:
            g = nx.adjacency_graph(g)
        buildingblocks = [n for n in g.nodes() if check_buildingblocks(n, g)]
        if buildingblocks == []:
            equal_subgraphs.append(rn)
            nodes_all.append([])
            edges_all.append([])
            continue

        nodes = sorted(list(g.nodes()))
        edges = sorted(list(g.edges()))

        if (nodes not in nodes_all) & (edges not in edges_all):
            nodes_all.append(nodes)
            edges_all.append(edges)
        else:
            idx = nodes_all.index(nodes)
            comp_g = routes[idx]
            if type(comp_g) == dict:
                comp_g = nx.adjacency_graph(comp_g)
            if edges_all[idx] == edges:
                equal_subgraphs.append(rn)
                nodes_all.append([])
                edges_all.append([])
                comp_g.graph["document_id"] = (
                    comp_g.graph["document_id"] + "&&" + g.graph["document_id"]
                )  # Graph can only be serialized if all attributes are strings
                routes[idx] = json_graph.adjacency_data(comp_g)
    return equal_subgraphs


def remove_identified(to_remove, routes):
    for tr in to_remove[::-1]:
        routes.pop(tr)
    return routes


def identify_stereoisomers(routes):
    fpgen = AllChem.GetMorganGenerator(radius=2)
    only_stereo = []

    for i, G in enumerate(tqdm(routes)):
        if type(G) == dict:
            G = nx.adjacency_graph(G)
        g = G.copy()
        nodes = g.nodes()
        products = [
            n
            for n in nodes
            if (len(g.in_edges(n)) == 0) and (g.nodes[n]["product"] == True)
        ]

        mols = [Chem.MolFromSmiles(n) for n in products]

        if len(mols) <= 1:
            only_stereo.append(i)
            continue

        fps = [fpgen.GetFingerprint(x) for x in mols]

        tanimoto_sim = [DataStructs.BulkTanimotoSimilarity(fp, fps) for fp in fps]
        tanimoto_sim = np.array(tanimoto_sim)
        tanimoto_sim_ = tanimoto_sim[np.tril_indices_from(tanimoto_sim, k=-1)]
        if len(tanimoto_sim_[tanimoto_sim_ == 1]) > 0:
            to_remove = []
            for a, b in zip(*np.where(np.tril(tanimoto_sim, k=-1) == 1)):
                if a == b:
                    continue
                a_ = list(g.successors(products[a]))
                b_ = list(g.successors(products[b]))
                if a_ == b_:
                    to_remove.append(products[a])
            [g.remove_node(p) for p in set(to_remove)]
            products, intermediates, buildingblocks = get_route_descriptors(g)

            if len(intermediates) == 0:
                only_stereo.append(i)
    return only_stereo


def count_reactions(
    routes,
):
    count = 0
    for g in routes:
        c = {}
        if type(g) == dict:
            g = nx.adjacency_graph(g)
        precedent = [d["precedent"] for e1, e2, d in g.edges.data()]
        precedent = [p_.split(" ") for p in precedent for p_ in p.split("_")]
        for p in precedent:
            smiles, cnt = p
            cnt = float(cnt)
            if smiles not in c:
                c[smiles] = cnt
            else:
                c[smiles] = c[smiles] + cnt
        count += sum(v for v in c.values())
    return count


def store_routes(routes, cleaned_root, clean_fp):
    s = json.dumps(routes)
    with open(f"{cleaned_root}/{clean_fp}.json", "w") as f:
        f.write(s)


@hydra.main(
    version_base=None,
    config_path="./experiments/deduplicate_routes/",
    config_name="config",
)
def run(args):
    log = logging.getLogger(__name__)

    routes = gather_files(args.routes_root, log)
    log.info(f"{len(routes)} routes identified")

    log.info(f"{count_reactions(routes)} original reactions involved in routes")

    if args.remove_stereoisomers:
        stereoisomers = identify_stereoisomers(routes)
        log.info(f"{len(stereoisomers)} stereoisomers identified")
        routes = remove_identified(stereoisomers, routes)
        log.info(f"{len(routes)} routes identified after stereoisomer removal")

    log.info(f"{count_reactions(routes)} original reactions involved in routes")

    if args.remove_duplicates:
        duplicated = identify_duplicated(routes)
        log.info(f"{len(duplicated)} duplicated routes identified")
        routes = remove_identified(duplicated, routes)
        log.info(f"{len(routes)} routes identified after depuplication")

    log.info(f"{len(routes)} routes identified after cleaning")

    store_routes(routes, args.store_root, args.store_fp)


if __name__ == "__main__":
    run()

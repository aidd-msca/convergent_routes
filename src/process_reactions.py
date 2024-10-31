import hydra
import pandas as pd
import networkx as nx
import json
import logging
import collections

from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover


from networkx.readwrite import json_graph
from itertools import combinations
from rdkit import RDLogger

from convergent_routes.src.utils import check_intermediates

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

RANDOM_SEED = 42


def convert_to_mol(r, remover=SaltRemover()):
    """Convert SMILES string to RDKit molecule then back to SMILES string.
    This is to ensure that the SMILES string is canonicalized and atom mapping removed.
    """
    try:
        r = Chem.MolFromSmiles(r)
    except:
        return None

    if r is None:
        return None
    atom_mapped = False
    for a in r.GetAtoms():
        if a.HasProp("molAtomMapNumber"):
            atom_mapped = True
        a.ClearProp("molAtomMapNumber")
    if atom_mapped:
        r = Chem.MolFromSmiles(Chem.MolToSmiles(r))
        if r is None:
            return None
    r = remover.StripMol(r, dontRemoveEverything=False)
    return Chem.MolToSmiles(r)


def canonicalize_smiles(smiles):
    """Canonicalize SMILES string."""
    if smiles is None:
        return None
    smiles = convert_to_mol(smiles)
    return smiles


def read_data(files):
    df = pd.DataFrame()
    for fp in files:
        df = pd.concat(
            [
                df,
                pd.read_csv(
                    fp,
                    sep="\t",
                ),
            ]
        )
    return df


def get_reaction_parts(smiles, log=None):
    try:
        rct, _, prd = smiles.split("|")[0].strip().split(">")
        return rct, prd
    except:
        if log is not None:
            log.error("Reaction parts splitting failed")
        return None, None


def get_reactions(df, additional_data=[]):
    data = []
    df.loc[:, "reactants"] = df.reactants.apply(canonicalize_smiles)
    df.loc[:, "products"] = df.products.apply(canonicalize_smiles)

    rcts = df.reactants.tolist()
    prds = df.products.tolist()

    data_cols_all = [
        "smiles",
        "reaction_name",
        "reaction_source",
        "limiting_reactant",
    ]  # TODO: This can be set as an argument
    df = df.rename(
        columns={"rxn_name": "reaction_name", "reaction_type": "reaction_source"}
    )  # TODO: This needs to be hnadled before starting the script
    data_cols = [dc for dc in data_cols_all if dc in df.columns]

    data_cols.extend(additional_data)

    df = df[data_cols].to_dict("list")

    for dc in data_cols_all:
        if dc not in df.keys():
            df[dc] = [None] * len(prds)
            data_cols.append(dc)

    reaction_count = []
    data = collections.defaultdict(list)
    for i, (rct, prd) in enumerate(zip(rcts, prds)):
        rct = rcts[i]
        prd = prds[i]
        if rct is None or prd is None:
            continue

        precedent = df["smiles"][i] if df["smiles"][i] is not None else f"{prd}>>{rct}"
        rxn_n = df["reaction_name"][i]
        rxn_t = df["reaction_source"][i]
        lim_rct = df["limiting_reactant"][i]
        rxn_count = 0
        if len(prd.split(".")) > 1:
            for prd_smi in prd.split("."):
                if prd_smi == "":
                    continue
                for rct_smi in rct.split("."):
                    if rct_smi == "":
                        continue
                    multiproduct = True
                    lim_r = True if lim_rct == rct_smi else False
                    data["products"].append(prd_smi)
                    data["reactants"].append(rct_smi)
                    data["multiproducts"].append(multiproduct)
                    data["precedent"].append(precedent)
                    for dc in data_cols:
                        if dc == "limiting_reactant":
                            data[dc].append(lim_r)
                        else:
                            data[dc].append(df[dc][i])
                    rxn_count += 1
        else:
            for rct_smi in rct.split("."):
                if rct_smi == "":
                    continue
                multiproduct = False
                lim_r = True if lim_rct == rct_smi else False
                data["products"].append(prd)
                data["reactants"].append(rct_smi)
                data["multiproducts"].append(multiproduct)
                data["precedent"].append(precedent)
                for dc in data_cols:
                    if dc == "limiting_reactant":
                        data[dc].append(lim_r)
                    else:
                        data[dc].append(df[dc][i])
                rxn_count += 1
        if rxn_count > 0:
            reaction_count.extend([1 / rxn_count] * rxn_count)
    return data, reaction_count


def create_graph(reactions, document_id=None, reaction_count=None):
    graph = nx.DiGraph(document_id=document_id)

    graph.add_nodes_from(reactions["products"], product=True, reactant=False)

    for r, lr in zip(reactions["reactants"], reactions["limiting_reactant"]):
        if not graph.has_node(r):
            graph.add_node(r, product=False, reactant=True, limiting_reactant=lr)
        else:
            graph.nodes[r]["reactant"] = True
            if lr:
                graph.nodes[r]["limiting_reactant"] = lr

    for i, (prd, rct, rxn_cnt) in enumerate(
        zip(reactions["products"], reactions["reactants"], reaction_count)
    ):
        if prd == rct:
            continue
        if not graph.has_edge(prd, rct):
            kwds = {}
            for k in reactions.keys():
                if k == "precedent":
                    kwds[k] = {reactions[k][i]: rxn_cnt}
                elif k == "reaction_source":
                    kwds[k] = [reactions[k][i]]
                else:
                    kwds[k] = reactions[k][i]
            graph.add_edge(
                prd,
                rct,
                **kwds,
            )

        else:
            if reactions["precedent"][i] in graph.edges[(prd, rct)]["precedent"]:
                graph.edges[(prd, rct)]["precedent"][
                    reactions["precedent"][i]
                ] += rxn_cnt
            else:
                graph.edges[(prd, rct)]["precedent"][
                    reactions["precedent"][i]
                ] = rxn_cnt

            if (
                reactions["reaction_source"][i]
                not in graph.edges[(prd, rct)]["reaction_source"]
            ):
                graph.edges[(prd, rct)]["reaction_source"].append(
                    reactions["reaction_source"][i]
                )

    return graph


def check_cycle_reactions(cpd, cyc_paths):
    if cpd is None:
        return False
    for cp in cyc_paths:
        for c in cpd.split("."):
            if cp == c:
                return True
    return False


def remove_cycles(graph, df_, log=None):
    for cyc_nodes in nx.simple_cycles(graph):
        cyc_df = df_[
            df_.apply(
                lambda x: check_cycle_reactions(x.reactants, cyc_nodes)
                & check_cycle_reactions(x.products, cyc_nodes),
                axis=1,
            )
        ]

        count_occ = collections.defaultdict(list)
        for idx, row in cyc_df.iterrows():
            count_occ[f"{row.products}>>{row.reactants}"].append(idx)

        min_occ = len(cyc_df) + 1
        to_remove = ""
        for rxn, occ in count_occ.items():
            occ = len(occ)
            if occ < min_occ:
                to_remove = rxn
            elif occ == min_occ:
                if log is not None:
                    log.info("Equal occurence of reaction")
                to_remove = ""
        if to_remove == "":
            continue
        prd, rct = rxn.split(">>")
        graph.remove_edges_from([[prd, r] for r in rct.split(".")])
        graph.remove_edges_from([[p, rct] for p in prd.split(".")])
        df_ = df_.drop(count_occ[to_remove])
    return graph, df_


def rescue_cycle(graph, df_, log=None):
    graph.remove_edges_from(nx.selfloop_edges(graph))
    for i in range(3):  # run cycle removal thrice, if not, remove
        graph, df_ = remove_cycles(graph, df_, log=log)
        if nx.is_directed_acyclic_graph(graph):
            graph.graph["contained_cycle"] = True
            if log is not None:
                log.info("Cycles addressed")
            return graph
    if log is not None:
        log.info(f"Cycle could not be removed: {list(nx.simple_cycles(graph))}")
    return None


def check_nodes(gA, gB):
    return sorted(list(gA.nodes())) == sorted(list(gB.nodes()))


def parse_graph(
    graph, skip_cycles=True, df_=None, log=None, only_convergent_routes=True
):
    nodes = graph.nodes()

    unconnected = [n for n in nodes if graph.degree(n) == 0]
    for un in unconnected:
        graph.remove_node(un)

    cr = []
    for g_ in nx.weakly_connected_components(graph):
        g = graph.subgraph(g_)
        if not nx.is_directed_acyclic_graph(g):
            if skip_cycles:
                if log is not None:
                    log.warning(f"Cycle found: {graph.graph['document_id']}")
                continue
            else:
                rescue_cycle(graph, df_, log=log)

        edges = g.edges()
        nodes = g.nodes()

        intermediates = [n for n in nodes if check_intermediates(n, g)]
        if only_convergent_routes:
            if len(intermediates) == 0:
                continue

        nd_rm = set()
        nd_kp = set()
        buildingblocks = [
            n
            for n in g.nodes()
            if (len(g.out_edges(n)) == 0) and (g.nodes[n]["product"] == False)
        ]
        if len(buildingblocks) == 0:
            if log is not None:
                log.warning(f"No building blocks found: {graph.graph['document_id']}")
            continue

        root_cpd = [
            n
            for n in g.nodes()
            if (len(g.in_edges(n)) == 0) and (g.nodes[n]["product"] == True)
        ]
        for n in root_cpd:
            dfs = [x for v in nx.dfs_successors(g, n).values() for x in v]
            if len(set(intermediates).intersection(dfs)) == 0:
                nd_rm.update(dfs)
                nd_rm.add(n)
            else:
                nd_kp.update(dfs)
                nd_kp.add(n)

        if len(nd_rm) >= 1:
            nd_rm = nd_rm - nd_kp.intersection(nd_rm)
            nd_kp = nd_kp.difference(nd_rm)

        cr.append(graph.subgraph(nd_kp))

    for gA, gB in combinations(cr, 2):
        if nx.is_isomorphic(
            gA,
            gB,
        ):
            if check_nodes(gA, gB):
                if (log is not None) & (only_convergent_routes):
                    log.warning(f"Equal subgraphs found: {graph.graph['document_id']}")

    return cr


def count_reactions(
    routes,
):
    count = 0
    for g in routes:
        c = {}
        precedent = nx.get_edge_attributes(g, "precedent")
        for e in g.edges():
            for smiles, cnt in precedent[e].items():
                if smiles not in c:
                    c[smiles] = cnt
                else:
                    c[smiles] = c[smiles] + cnt
        count += sum(v for v in c.values())
    return count


def process_document(
    df_,
    document_n,
    skip_cycles=True,
    log=None,
    only_convergent_routes=True,
    additional_data=[],
):
    reactions, reaction_count = get_reactions(df_, additional_data=additional_data)
    if reactions is None:
        return None
    
    if reactions == []:
        if log is not None:
            log.warning(f"No reactions found: {document_n}")
        return None
    if log is not None:
        log.info(f"Identified {len(reactions)} reactions")

    graph = create_graph(reactions, document_n, reaction_count)

    return parse_graph(
        graph,
        skip_cycles,
        df_=df_,
        log=log,
        only_convergent_routes=only_convergent_routes,
    )


def prep_graph(graph):
    for e in graph.edges():
        graph.edges[e]["precedent"] = ("_").join(
            [f"{k} {v}" for k, v in graph.edges[e]["precedent"].items()]
        )
        graph.edges[e]["reaction_source"] = ("&&").join(
            [str(rs) for rs in graph.edges[e]["reaction_source"]]
        )
    return json_graph.adjacency_data(graph)


def store_graphs(collate_data, args, sn):
    collate_data = [prep_graph(g_) for g_ in collate_data]
    s = json.dumps(collate_data)
    with open(f"{args.store_root}/routes_{sn}.json", "w") as f:
        f.write(s)


@hydra.main(
    version_base=None,
    config_path="./experiments/process_reactions/",
    config_name="config",
)
def run(args):
    log = logging.getLogger(__name__)
    sn = 0
    collate_data = []

    df = read_data(args.files)

    if args.process_data:
        output = df.ReactionSmiles.apply(get_reaction_parts, log=log)
        df["reactants"], df["products"] = list(zip(*output.values.tolist()))
    n_routes = 0
    if args.document_id is not None:
        for document_n, df_ in tqdm(df.groupby(args.document_id)):
            if len(df_) < 2:
                continue

            df_ = df_.reset_index(drop=True)
            routes = process_document(
                df_, document_n, skip_cycles=args.skip_cycles, log=log
            )
            if routes is None:
                continue
            collate_data.extend(routes)
            n_routes += len(routes)
            if len(collate_data) > args.max_file:
                store_graphs(collate_data=collate_data, args=args, sn=sn)
                collate_data = []
                sn += 1
    else:
        collate_data = process_document(df, "FULL", args.skip_cycles, log=log)
    store_graphs(collate_data=collate_data, args=args, sn=sn)

    log.info(f"{n_routes} routes identified")


if __name__ == "__main__":
    run()

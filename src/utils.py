import networkx as nx
import matplotlib.pyplot as plt
import json

from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover

from rdkit import RDLogger

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


def check_intermediates(n, graph, search=False):
    if len(graph.in_edges(n)) > 1:
        cs = [isinstance(p, int) for p in list(graph.predecessors(n))]
        if len(cs) - sum(cs) > 1:
            return True
        for p in list(graph.predecessors(n)):
            if type(p) == int:
                p = [p_[-1] for p_ in list(graph.out_edges(p))]
                if len(p) == 2:
                    if not nx.has_path(graph, *p):
                        return True
    return False


def check_buildingblocks(n, graph, search=False):
    if not search:
        if (
            (len(graph.out_edges(n)) == 0)
            and (graph.nodes[n]["product"] == False)
            and (len(graph.in_edges(n)) < 2)
        ):
            return True
    else:
        if (len(graph.out_edges(n)) == 0) and (len(graph.in_edges(n)) < 2):
            return True
    return False


def check_endnodes(n, graph, search=False):
    if not search:
        if (
            (len(graph.out_edges(n)) == 0)
            and (graph.nodes[n]["product"] == False)
            and (len(graph.in_edges(n)) < 2)
        ):
            return True
    else:
        if len(graph.out_edges(n)) == 0:
            return True
    return False


def check_products(n, graph, search=False):
    if not search:
        if (
            (len(graph.in_edges(n)) == 0)
            or (all(isinstance(p, int) for p in list(graph.predecessors(n))))
            and (graph.nodes[n]["product"] == True)
        ):
            return True
        else:
            return False
    else:
        if n in nx.get_node_attributes(graph, "target"):
            return True
    return False


def get_route_descriptors(graph, search=False):
    nodes = graph.nodes()
    buildingblocks = [n for n in nodes if check_buildingblocks(n, graph, search)]
    intermediates = [n for n in nodes if check_intermediates(n, graph, search)]
    products = [n for n in nodes if check_products(n, graph, search)]
    return products, intermediates, buildingblocks


def plot_convergent_route(
    g_,
    root_cpd=[],
    intermediates=[],
    endnodes=[],
    highlight=[],
    figsize=(5, 5),
    ax=None,
    graphviz_type="dot",
    shownodelabels=False,
    showrxnlabels=False,
    save_fp=None,
    title=None,
    fig=None,
    nodes=None,
    edges=None,
    node_size=300,
):
    if nodes is None:
        nodes = g_.nodes()
    if edges is None:
        edges = g_.edges()

    pos = nx.drawing.nx_agraph.graphviz_layout(g_, prog=graphviz_type)


    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    node_colors = []
    node_labels = []
    node_edge_colors = []
    alpha_n = []
    for n in list(nodes):
        if n in root_cpd:
            node_colors.append("red")
            node_labels.append(n)
        elif n in endnodes:
            node_colors.append("yellow")
            node_labels.append("")
        elif n in intermediates:
            node_colors.append("pink")
            node_labels.append("")
        else:
            node_colors.append("blue")
            node_labels.append("")
        if n in highlight:
            node_edge_colors.append("green")
        else:
            node_edge_colors.append(node_colors[-1])
        if "ChemicalSimilarity" in g_.nodes[n]:
            alpha_n.append(0.5)
            node_edge_colors[-1] = "white"
        else:
            alpha_n.append(1)

    edge_colors = []
    alpha_e = []

    for e in g_.edges.data():
        if "ChemicalSimilarity" in e[-1].keys():
            alpha_e.append(0.5)
        else:
            alpha_e.append(1)
        if ("multiproduct" in e) and (e[-1]["multiproduct"]):
            edge_colors.append("purple")
        else:
            edge_colors.append("gray")

    ec = nx.draw_networkx_edges(
        g_, pos, edgelist=list(edges), edge_color=edge_colors, alpha=alpha_e, ax=ax
    )
    nds = nx.draw_networkx_nodes(
        g_,
        pos,
        nodelist=list(nodes),
        node_size=node_size,
        node_color=node_colors,
        edgecolors=node_edge_colors,
        linewidths=2,
        alpha=alpha_n,
        ax=ax,
    )
    if shownodelabels:
        nx.draw_networkx_labels(g_, pos, font_size=9, ax=ax)

    if showrxnlabels:
        edge_labels = nx.get_edge_attributes(g_, "reaction_name")
        nx.draw_networkx_edge_labels(
            g_, pos, edge_labels=edge_labels, font_size=8, ax=ax
        )

    if title is not None:
        plt.title(title)

    plt.tight_layout()
    if save_fp is not None:
        plt.savefig(save_fp, dpi=300, bbox_inches="tight")

    if fig is not None:
        return fig


def analyse_routes(
    file, n_routes=0, n_products=[], n_intermediates=[], n_reactions=[], reactions=set()
):
    with open(file, "r") as f:
        gtest = json.load(f)
    n_routes += len(gtest)

    for g in gtest:
        g = nx.adjacency_graph(g)
        nodes = g.nodes()
        n_reactions.append(len(g.edges()))
        reactions.update(set(g.edges()))
        n_intermediates.append(
            len(
                [
                    n
                    for n in nodes
                    if (len(g.in_edges(n)) > 1) & (len(g.out_edges(n)) > 0)
                ]
            )
        )
        n_products.append(
            len(
                [
                    n
                    for n in nodes
                    if (len(g.in_edges(n)) == 0) and (g.nodes[n]["product"] == True)
                ]
            )
        )
    return n_routes, n_products, n_intermediates, n_reactions, reactions


def convert_to_mol(r, remover=SaltRemover()):
    """
    Convert SMILES string to RDKit molecule then back to SMILES string.
    This is to ensure that the SMILES string is canonicalized and atom mapping removed.
    """
    r = Chem.MolFromSmiles(r)
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

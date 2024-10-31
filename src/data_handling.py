import pandas as pd


def process_buildingblocks(fp):
    """
    Requires file path to buildingblocks, it should contain a column named buildingblocks with canonicalized molecules.
    """
    df = pd.read_csv(fp, sep="\t")
    return df.buildingblocks.to_list()

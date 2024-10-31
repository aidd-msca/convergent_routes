from convergent_search import ConvergentSearch
from analyse_search import ConvergentRoutes
from evaluate_search import ConvergentEvaluation
import pandas as pd
from data_handling import process_buildingblocks
import hydra
from tqdm import tqdm
import numpy as np
import logging


def run_multiple_searches(
    fp, group_col, molecule_col, individual=False, subsample=None
):
    df = pd.read_csv(fp, sep="\t")
    if not individual:
        if subsample is not None:
            sb_grps = np.random.choice(df[group_col].unique(), subsample)
            df = df[df[group_col].isin(sb_grps)]
        for idx, compounds in df.groupby(group_col):
            yield idx, compounds[molecule_col].tolist()
    else:
        if subsample is not None:
            df = df.sample(subsample)
        for idx, row in df.iterrows():
            yield row[group_col], [row[molecule_col]]


@hydra.main(
    version_base=None,
    config_path="./../experiments/convergent_search/",
    config_name="config",
)
def run(args):
    log = logging.getLogger("convergentsearch")
    log.info(f"Processing building blocks - {args.buildingblocks_fp}")
    buildingblocks = process_buildingblocks(args.buildingblocks_fp)

    log.info("Initializing convergent search")
    cs = ConvergentSearch(
        buildingblocks=buildingblocks,
        k=args.k,
        max_time=args.searchsettings.max_time,
        max_iterations=args.searchsettings.max_iterations,
        max_len_route=args.searchsettings.max_len_route,
        single_step_model=args.single_step_model.model_name,
        single_step_module_path=args.single_step_model.module_path,
        single_step_use_gpu=args.single_step_model.use_gpu,
        single_step_settings=args.single_step_model.model_settings,
        target_molecule_limit=args.searchsettings.target_molecule_limit,
    )
    ce = ConvergentEvaluation(args.reference_fp, args.experiment_name)
    if args.target_molecules is not None:
        cs.setup(args.target_molecules)
        if cs.valid:
            search = cs.run_search(args.batch_size)
            cr = ConvergentRoutes(
                keep_max_len=args.keep_max_len,
            )
            routes = cr.get_routes(
                search,
                args.target_molecules,
                store_fp=args.store_routes,
            )

        ce.evaluate_route(routes)
    elif args.target_molecules_fp:
        search = {}
        for idx, target_molecules in tqdm(
            run_multiple_searches(
                args.target_molecules_fp,
                args.group_col,
                args.molecule_col,
                args.individual,
                args.subsample,
            )
        ):
            cs.setup(target_molecules)
            if not cs.valid:
                routes = {}
                if args.reference_fp is not None:
                    ce.run_evaluation(routes, idx, target_molecules)

                if args.store_routes:
                    ce.store_route(routes, n=args.store_routes.n)
                    ce.store_routes(args.store_routes.store_fp)
                cs.clear_search()
                continue

            search = cs.run_search(args.batch_size)

            cr = ConvergentRoutes(
                args.searchsettings.max_len_route,
            )
            routes = cr.get_routes(
                search,
                target_molecules,
                max_time_pm=args.searchsettings.max_time,
            )
            if args.reference_fp is not None:
                ce.run_evaluation(routes, idx, target_molecules)

            if args.store_routes:
                ce.store_route(routes, n=args.store_routes.n)
                ce.store_routes(args.store_routes.store_fp)
            cs.clear_search()
        ce.store_results(args.results_fp)


if __name__ == "__main__":
    run()

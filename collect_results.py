#!/usr/bin/env python3
import argparse
import json
import os
import pprint as pp

import pandas as pd


def read_dataset_outdir(root, opts):
    experiment_dirs = [x for x in os.listdir(root)
                       if os.path.isdir(os.path.join(root, x)) \
                       and opts.run_name in x.split('_')
                       ]

    individual_frames = []

    for exp_dir in experiment_dirs:
        runsplits = exp_dir.split('_')
        if len(runsplits) == 4:
            # Inductive datasets with cross-validation
            model, run_id, fold, timestamp = runsplits
            fold = int(fold.split('-')[1])
        elif len(runsplits) == 3:
            # Transductive datasets with one predefined data split
            model, run_id, timestamp = runsplits
            fold = 0
        else:
            assert False, "Unexpected experiment directory name format."
        if opts.verbose:
            print(f" > Parsing experiment: {model, run_id, fold, timestamp}")

        results_file = os.path.join(root, exp_dir, 'results.json')
        results_tmp_file = os.path.join(root, exp_dir, 'results-tmp.json')
        if not os.path.isfile(results_file):
            msg = f"!!! Final results {results_file} not found   "
            if not os.path.isfile(results_tmp_file):
                print(f"{msg}...ignoring")
                continue
            else:
                results_file = results_tmp_file
                print(f"{msg}...using intermediate results file.")
        with open(results_file) as f:
            results = json.load(f)

        if '-' not in model or not opts.transpose_tag:
            lvls = ('model', 'fold')
            frame_index = pd.MultiIndex.from_tuples([(model, fold)], names=lvls)
        else:
            lvls = ('model', 'tag', 'fold')
            model_base, model_tag = model.rsplit("-", 1)
            frame_index = pd.MultiIndex.from_tuples([(model_base, model_tag, fold)],
                                                    names=lvls)
        individual_frames.append(pd.DataFrame(results, index=frame_index))
        # print(individual_frames[-1])

    if len(individual_frames):
        df = pd.concat(individual_frames).sort_index()
    else:
        df = pd.DataFrame()

    if opts.verbose:
        print(df)
    # print(df.groupby(level=['model']).mean())

    if not df.empty:
        mean_df = df.mean(axis=0, level=lvls[:-1])
        sem_df = df.sem(axis=0, level=lvls[:-1])
        if opts.std:
            sem_df = df.std(axis=0, level=lvls[:-1])

        num_folds = df.groupby(level=lvls[:-1]).count()['epoch']
        mean_df['num_folds'] = num_folds
        sem_df['num_folds'] = num_folds
    else:
        mean_df = df
        sem_df = df

    if opts.verbose:
        print(mean_df)
        # print(sem_df)

    return mean_df, sem_df


def main():
    parser = argparse.ArgumentParser(
        description="Collect results from cross-validation."
    )
    parser.add_argument('-d', '--dir', default='outputs',
                        help="Directory with result outputs.")
    parser.add_argument('--run_name', default='run',
                        help="Identifier for experiment runs to select.")
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Print notifications and partial results.')
    parser.add_argument('--no_save', action='store_true',
                        help='Do not write results to CSV files.')
    parser.add_argument('--sem', action='store_true',
                        help='Print unbiased standard error of the mean of the results.')
    parser.add_argument('--std', action='store_true',
                        help='Print standard deviation of the results.')
    parser.add_argument('-s', '--select', type=str, default='',
                        help='Select particular perf. metric, e.g. "test/AUROC"')
    parser.add_argument('--transpose-tag', action='store_true',
                        help='Separate tag from model name and present it as columns.')
    opts = parser.parse_args()
    print("[*] Options")
    pp.pprint(vars(opts))
    print("")
    assert not(opts.sem and opts.std), "Only one option can be set at the same time."

    datasets = [x for x in os.listdir(os.path.realpath(opts.dir))
                if os.path.isdir(os.path.join(opts.dir, x))]
    mean_frames, sem_frames, processed_datasets = [], [], []
    for ds in datasets:
        ds_dir = os.path.join(opts.dir, ds)
        if opts.verbose:
            print(f"\n\nProcessing dataset {ds} in {ds_dir}...")
        mean_df, sem_df = read_dataset_outdir(ds_dir, opts)
        if not mean_df.empty:
            processed_datasets.append(ds)
            mean_frames.append(mean_df)
            sem_frames.append(sem_df)
    results = pd.concat(mean_frames, keys=processed_datasets, names=['dataset']).sort_index()
    sems = pd.concat(sem_frames, keys=processed_datasets, names=['dataset']).sort_index()

    if opts.sem or opts.std:
        res_to_print = sems
    else:
        res_to_print = results
    pd.options.display.float_format = '{:.4f}'.format
    if opts.select:
        assert opts.select in results, f"Selected metric not found: {opts.select}"
        # unstack one of the MultiIndex levels into columns
        if 'tag' in res_to_print.index.names:
            print(res_to_print[opts.select].unstack(level='tag'))
        else:
            print(res_to_print[opts.select].unstack(level='dataset'))
    else:
        print(res_to_print)
        # print(res_to_print[["epoch", "test/AUROC"]])

    results_csv_name = os.path.join(opts.dir, f"all_results-{opts.run_name}.csv")
    sems_csv_name = os.path.join(opts.dir, f"all_results_sem-{opts.run_name}.csv")
    if not opts.no_save:
        print(f"\n[*]\nSaving fold-averaged results to: {results_csv_name}")
        print(f"Saving fold-sems to: {sems_csv_name}")
        results.to_csv(results_csv_name, sep='\t')
        sems.to_csv(sems_csv_name, sep='\t')

if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import os
import argparse
import yaml
import shutil
import json
import itertools
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from colorama import init, Fore

import add_path
from trajectory import Trajectory
import plot_utils as pu
import results_writer as res_writer
from analyze_trajectory_single import analyze_multiple_trials
from fn_constants import kNsToEstFnMapping, kNsToMatchFnMapping, kFnExt

init(autoreset=True)

rc('font', **{'family': 'serif', 'serif': ['Cardo']})
rc('text', usetex=True)

FORMAT = '.pdf'

def spec(N):
    t = np.linspace(-510, 510, N)
    return np.round(np.clip(np.stack([-t, 510-np.abs(t), t], axis=1), 0, 255)).astype("float32")/255

PALLETE = spec(20)


def parse_config_file(config_function, sort_names):
    """ Parse Config File Method

    This method is used to parse the configuration yaml file (for AV, we want to use the av_euroc_vio_mono_config.yaml
    file). This yaml file is structured as follows:

        Included Datasets:
            MH_01:
                label: MH01
            MH_03:
                label: MH03
            MH_05:
                label: MH05
        Algorithms:
            rovio:
                include algorithm: true
                label: rovio
                file name: traj_est
                tests: [0, 1, 3]
            larvio:
                include algorithm: false
                label: larvio
                file name: traj_est
            svo2:
                include algorithm: true
                label: svo2
                file name: traj_est
                tests: [3, 4]
        RelDistances: []
        Comparison Option:
            algorithms: false

    Args:
        config_function (datatype: string): This is a string which contains the filepath to the config.yaml file. For
            our case, we use want to path to the av_euroc_vio_mono_config.yaml file.

        sort_names (datatype: boolean): This is a boolean flag that is passed in as an argument to the python script.
            Enabling this means we sort the list of algorithms. This is useful if you want to create plots that compare
            different algorithms and you want the algorithm legend to be in alphabetical order.

    Returns:
        datasets_list (datatype: list): This is a list that contains the datasets that the user wants to generates
            figures for.

            datasets_list = ['MH_01', 'MH_03', 'MH_05']

        datasets_labels_dict (datatype: dict): This is a dictionary that contains the datasets and their corresponding
            labels.

            datasets_labels_dict = {'MH_01': 'MH01', 'MH_03': 'MH03', 'MH_05': 'MH05'}

        datasets_titles_dict (datatype: dict):

        algorithms_list (datatype: list):

            algorithms_list = ['larvio', 'rovio', 'svo2']

        algorithms_labels_dict (datatype: dict):

            algorithms_labels_dict = {'rovio': 'rovio', 'svo2': 'svo2'}

        algorithms_file_names_dict (datatype: dict):

            algorithms_file_names_dict = {'rovio': ['traj_est_0', 'traj_est_1', 'traj_est_3'],
                'svo2': ['traj_est_3', 'traj_est_4']}

        algorithms_tests_dict (datatype: dict):

            algorithms_tests_dict = {'rovio': [0, 1, 3], 'svo2': [3, 4]}

    """
    with open(config_function) as stream:
        yaml_config_data = yaml.safe_load(stream)

    # Comparison Options
    comparison_options = yaml_config_data['Comparison Options']

    # Datasets (i.e., MH_01, MH_03, MH_05)
    datasets_list = yaml_config_data['Included Datasets'].keys()

    # Sort the Datasets list
    if sort_names:
        datasets_list = sorted(datasets_list)

    # Datasets labels and titles (datatype: dictionaries)
    datasets_labels_dict = {}
    datasets_titles_dict = {}

    # For each dataset in the config.yaml, assign the dataset label into the datasets_labels dictionary
    for datasets_name in datasets_list:
        datasets_labels_dict[datasets_name] = yaml_config_data['Included Datasets'][datasets_name]['label']

        if 'title' in yaml_config_data['Included Datasets'][datasets_name]:
            datasets_titles_dict[datasets_name] = yaml_config_data['Included Datasets'][datasets_name]['title']

    # Algorithms (i.e., rovio, larvio, svo2)
    algorithms_list = yaml_config_data['Algorithms'].keys()

    # Sort the Algorithm list
    if sort_names:
        algorithms_list = sorted(algorithms_list)

    # Algorithm labels, filenames, and tests (datatype: dictionaries)
    algorithms_labels_dict = {}
    algorithms_file_names_dict = {}
    algorithms_tests_dict = {}
    algorithms_tests_labels_dict = {}

    # For each algorithm in the config.yaml, assign the algorithm label into the algorithms_labels dictionary and
    # specify the traj_est_<test_number> file that you want to extract data from.
    for algorithms_name in algorithms_list:

        # if we specify to include the algorithm
        include_algorithm = yaml_config_data['Algorithms'][algorithms_name]['include algorithm']
        if include_algorithm:
            algorithms_labels_dict[algorithms_name] = yaml_config_data['Algorithms'][algorithms_name]['label']

            algorithms_file_names_list = []
            algorithms_tests_list = []
            algorithms_test_labels_list = []

            for test_number in yaml_config_data['Algorithms'][algorithms_name]['tests']:
                algorithms_file_name_string = yaml_config_data['Algorithms'][algorithms_name]['file name']

                algorithms_test_label_string = yaml_config_data['Algorithms'][algorithms_name]['label'] + '-test-' + \
                    str(test_number)

                algorithms_file_names_list.append(algorithms_file_name_string)
                algorithms_test_labels_list.append(algorithms_test_label_string)
                algorithms_tests_list.append(test_number)

            algorithms_file_names_dict[algorithms_name] = algorithms_file_names_list
            algorithms_tests_dict[algorithms_name] = algorithms_tests_list
            algorithms_tests_labels_dict[algorithms_name] = algorithms_test_labels_list

    boxplot_distances_dict = []
    if 'RelDistances' in yaml_config_data:
        boxplot_distances_dict = yaml_config_data['RelDistances']
    boxplot_percentages_dict = []
    if 'RelDistancePercentages' in yaml_config_data:
        boxplot_percentages_dict = yaml_config_data['RelDistancePercentages']

    if boxplot_distances_dict and boxplot_percentages_dict:
        print(Fore.RED + "Found both both distances and percentages for boxplot distances")
        print(Fore.RED + "Will use the distances instead of percentages.")
        boxplot_percentages_dict = []

    return comparison_options, datasets_list, datasets_labels_dict, datasets_titles_dict, algorithms_list, \
        algorithms_labels_dict, algorithms_file_names_dict, algorithms_tests_dict, algorithms_tests_labels_dict, \
        boxplot_distances_dict, boxplot_percentages_dict

if __name__ == '__main__':

    # Parsing Argument Options
    parser = argparse.ArgumentParser(description='''Analyze trajectories''')

    default_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '../results')
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               '../analyze_trajectories_config')

    parser.add_argument('config', type=str,
                        help='yaml file specifying algorithms and datasets')
    parser.add_argument(
        '--output_dir',
        help="Folder to output plots and data",
        default=default_path)
    parser.add_argument(
        '--results_dir', help='base folder with the results to analyze',
        default=default_path)

    parser.add_argument(
        '--platform', help='HW platform: [laptop, nuc, odroid, up]',
        default='laptop')
    parser.add_argument(
        '--mul_trials', type=int,
        help='number of trials, None for single run', default=None)
    parser.add_argument('--no_sort_names', action='store_false', dest='sort_names',
                        help='whether to sort dataset and algorithm names')

    # odometry error
    parser.add_argument(
        '--odometry_error_per_dataset',
        help="Analyze odometry error for individual dataset. "
             "The same subtrajectory length will be used for the same dataset "
             "and different algorithms",
        action='store_true')
    parser.add_argument(
        '--overall_odometry_error',
        help="Collect the odometry error from all datasets and calculate statistics.",
        dest='overall_odometry_error',
        action='store_true')

    # RMSE (ATE)
    parser.add_argument(
        '--rmse_table', help='Output rms erros into latex tables',
        action='store_true')
    parser.add_argument(
        '--rmse_table_median_only', action='store_true', dest='rmse_median_only')
    parser.add_argument('--rmse_boxplot',
                        help='Plot the trajectories', action='store_true')

    parser.add_argument('--write_time_statistics', help='write time statistics',
                        action='store_true')

    # plot trajectories
    parser.add_argument('--plot_trajectories',
                        help='Plot the trajectories', action='store_true')
    parser.add_argument('--no_plot_side', action='store_false', dest='plot_side')
    parser.add_argument('--no_plot_aligned', action='store_false', dest='plot_aligned')
    parser.add_argument('--no_plot_traj_per_alg', action='store_false',
                        dest='plot_traj_per_alg')

    parser.add_argument('--recalculate_errors',
                        help='Deletes cached errors', action='store_true')
    parser.add_argument('--png',
                        help='Save plots as png instead of pdf',
                        action='store_true')
    parser.add_argument('--dpi', type=int, default=300)
    parser.set_defaults(odometry_error_per_dataset=False, overall_odometry_error=False,
                        rmse_table=False,
                        plot_trajectories=False, rmse_boxplot=False,
                        recalculate_errors=False, png=False, time_statistics=False,
                        sort_names=True, plot_side=True, plot_aligned=True,
                        plot_traj_per_alg=True, rmse_median_only=False)

    args = parser.parse_args()
    print("Arguments:\n{}".format(
        '\n'.join(['- {}: {}'.format(k, v)
                   for k, v in args.__dict__.items()])))

    print("Will analyze results from {0} and output will be "
          "in {1}".format(args.results_dir, args.output_dir))
    output_dir = args.output_dir

    config_fn = os.path.join(config_path, args.config)

    print("Parsing evaluation configuration {0}...".format(config_fn))

    # Parsing the Configuration yaml file
    comparison_opt, datasets, datasets_labels, datasets_titles, algorithms, algorithms_labels, \
        algorithms_file_names, algorithms_tests, algorithms_tests_labels, boxplot_distances, \
        boxplot_percentages = parse_config_file(config_fn, args.sort_names)

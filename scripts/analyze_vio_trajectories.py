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


def collect_odometry_error_per_dataset(dataset_multierror_list, dataset_names):
    """

    """
    dataset_relative_error = []

    print("\n>>> Collecting relative error (KITTI style)...")

    for dataset_idx, dataset_nm in enumerate(dataset_names):
        print("> Processing {0} for all configurations...".format(
            dataset_nm))
        dataset_mt_error = dataset_multierror_list[dataset_idx]
        print("> Found {0} configurations.".format(len(dataset_mt_error)))
        for d in dataset_mt_error:
            print('  - {0}: {1}'.format(d.alg, d.uid))

        cur_res = {'trans_err': {}, 'trans_err_perc': {},
                   'ang_yaw_err': {}, 'rot_deg_per_m': {},
                   'subtraj_len': []}

        cur_res['subtraj_len'] = dataset_mt_error[0].rel_distances
        for d in dataset_mt_error:
            assert cur_res['subtraj_len'] == d.rel_distances, \
                "inconsistent boxplot distances"
        print("Using distances {0} for relative error.".format(
            cur_res['subtraj_len']))

        print("Processing each configurations...")
        for mt_error in dataset_mt_error:
            cur_alg = mt_error.alg
            print('  - {0}'.format(cur_alg))
            cur_res['trans_err'][cur_alg] = []
            cur_res['trans_err_perc'][cur_alg] = []
            cur_res['ang_yaw_err'][cur_alg] = []
            cur_res['rot_deg_per_m'][cur_alg] = []

            for dist in cur_res['subtraj_len']:
                cur_res['trans_err'][cur_alg].append(
                    mt_error.rel_errors[dist]['rel_trans'])
                cur_res['trans_err_perc'][cur_alg].append(
                    mt_error.rel_errors[dist]['rel_trans_perc'])
                cur_res['ang_yaw_err'][cur_alg].append(
                    mt_error.rel_errors[dist]['rel_yaw'])
                cur_res['rot_deg_per_m'][cur_alg].append(
                    mt_error.rel_errors[dist]['rel_rot_deg_per_m'])

        dataset_relative_error.append(cur_res)
        print("< Finish processing {0} for all configurations.".format(
            dataset_nm))

    print("<<< ...finish collecting relative error.\n")

    return dataset_relative_error


def plot_odometry_error_per_dataset(dataset_rel_err, dataset_names, included_algorithm_names, algorithm_test_names,
                                    test_names_list, datasets_outdir, plot_settings):
    """

    """

    for dataset_idx, dataset_nm in enumerate(dataset_names):
        output_dir = datasets_outdir[dataset_nm]
        print("Plotting {0}...".format(dataset_nm))
        rel_err = dataset_rel_err[dataset_idx]
        assert sorted(test_names_list) == sorted(list(rel_err['trans_err'].keys()))
        distances = rel_err['subtraj_len']

        config_labels = []
        config_colors = []

        # for test_name in test_names_list:
        #     print(test_name)
        #     print(plot_settings['algorithms_tests_labels'])
        #     config_labels.append(plot_settings['algorithms_tests_labels'][test_name])
        #     config_colors.append(plot_settings['alg_colors'][test_name])
        #
        # fig = plt.figure(figsize=(12, 3))
        # ax = fig.add_subplot(121, xlabel='Distance traveled (m)', ylabel='Translation error (\%)')
        # pu.boxplot_compare(ax, distances, [rel_err['trans_err_perc'][test_name] for test_name in test_names_list],
        #                    config_labels, config_colors, legend=False)
        #
        # ax = fig.add_subplot(122, xlabel='Distance traveled (m)', ylabel='Rotation error (deg / m)')
        # pu.boxplot_compare(ax, distances, [rel_err['rot_deg_per_m'][test_name] for test_name in test_names_list],
        #                    config_labels, config_colors, legend=True)
        # fig.tight_layout()
        # fig.savefig(output_dir + '/' + dataset_nm +
        #             '_trans_rot_error' + FORMAT, bbox_inches="tight", dpi=args.dpi)
        # plt.close(fig)

        for algo in included_algorithm_names:
            for alg_test in range(len(algorithm_test_names[alg])):

                config_labels.append(plot_settings['algorithms_tests_labels'][algo][alg_test])
                config_colors.append(plot_settings['alg_colors'][algo][alg_test])

        print(config_labels)
        print(config_colors)

        fig = plt.figure(figsize=(12, 3))
        ax = fig.add_subplot(121, xlabel='Distance traveled (m)', ylabel='Translation error (\%)')
        pu.boxplot_compare(ax, distances, [rel_err['trans_err_perc'][test_name] for test_name in test_names_list],
                           config_labels, config_colors, legend=False)

        ax = fig.add_subplot(122, xlabel='Distance traveled (m)', ylabel='Rotation error (deg / m)')
        pu.boxplot_compare(ax, distances, [rel_err['rot_deg_per_m'][test_name] for test_name in test_names_list],
                           config_labels, config_colors, legend=True)

        fig.tight_layout()
        fig.savefig(output_dir + '/' + dataset_nm +
                    '_trans_rot_error' + FORMAT, bbox_inches="tight", dpi=args.dpi)
        plt.close(fig)

        # for algo in included_algorithm_names:
        #     pu.boxplot_compare(ax, distances, [rel_err['trans_err_perc'][algorithm_test_names[algo][alg_test]]
        #                                        for alg_test in range(len(algorithm_test_names[alg]))],
        #                        config_labels, config_colors, legend=False)
        #     ax = fig.add_subplot(
        #         122, xlabel='Distance traveled (m)', ylabel='Rotation error (deg / m)')
        #     pu.boxplot_compare(ax, distances, [rel_err['rot_deg_per_m'][algorithm_test_names[algo][alg_test]]
        #                                        for alg_test in range(len(algorithm_test_names[alg]))],
        #                        config_labels, config_colors, legend=True)
        #     fig.tight_layout()
        #     fig.savefig(output_dir + '/' + dataset_nm +
        #                 '_trans_rot_error' + FORMAT, bbox_inches="tight", dpi=args.dpi)
        #     plt.close(fig)




    #
    # for dataset_idx, dataset_nm in enumerate(dataset_names):
    #     output_dir = datasets_outdir[dataset_nm]
    #     print("Plotting {0}...".format(dataset_nm))
    #     rel_err = dataset_rel_err[dataset_idx]
    #     assert sorted(algorithm_names) == sorted(list(rel_err['trans_err'].keys()))
    #     distances = rel_err['subtraj_len']
    #
    #     config_labels = []
    #     config_colors = []
    #     for v in algorithm_names:
    #         print('alg\n')
    #         print(v)
    #         config_labels.append(plot_settings['algorithms_tests_labels'][v])
    #         config_colors.append(plot_settings['alg_colors'][v])
    #
    #     fig = plt.figure(figsize=(12, 3))
    #     ax = fig.add_subplot(
    #         121, xlabel='Distance traveled (m)',
    #         ylabel='Translation error (\%)')
    #     pu.boxplot_compare(ax, distances, [rel_err['trans_err_perc'][v] for v in algorithm_names],
    #                        config_labels, config_colors, legend=False)
    #     ax = fig.add_subplot(
    #         122, xlabel='Distance traveled (m)', ylabel='Rotation error (deg / m)')
    #     pu.boxplot_compare(ax, distances, [rel_err['rot_deg_per_m'][v] for v in algorithm_names],
    #                        config_labels, config_colors, legend=True)
    #     fig.tight_layout()
    #     fig.savefig(output_dir+'/'+dataset_nm +
    #                 '_trans_rot_error'+FORMAT, bbox_inches="tight", dpi=args.dpi)
    #     plt.close(fig)


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

    shutil.copy2(config_fn, output_dir)
    datasets_res_dir = {}
    for d in datasets:
        cur_res_dir = os.path.join(output_dir, '{}_{}_results'.format(args.platform, d))
        if os.path.exists(cur_res_dir):
            shutil.rmtree(cur_res_dir)
        os.makedirs(cur_res_dir)
        datasets_res_dir[d] = cur_res_dir
    same_subtraj = True if boxplot_distances else False

    included_alg = algorithms_tests_labels.keys()

    if comparison_opt['algorithms']:
        check_total_num_alg_test = 0
        for alg in included_alg:
            check_total_num_alg_test += len(algorithms_tests_labels[alg])
        assert len(PALLETE) > check_total_num_alg_test, "Not enough colors for all algorithms and test configurations"
    else:
        for alg in included_alg:
            assert len(PALLETE) > len(algorithms_tests_labels[alg])

    alg_colors = {}

    if comparison_opt['algorithms']:
        alg_test_idx = 0
        for alg in included_alg:
            test_color_list = []
            for test in algorithms_tests[alg]:
                test_color_list.append(PALLETE[alg_test_idx])
                alg_test_idx += 1
            alg_colors[alg] = test_color_list
    else:
        for alg in included_alg:
            alg_test_idx = 0
            test_color_list = []
            for test in algorithms_tests[alg]:
                test_color_list.append(PALLETE[alg_test_idx])
                alg_test_idx += 1
            alg_colors[alg] = test_color_list

    print(Fore.YELLOW + "=== Evaluation Configuration Summary ===")
    print(Fore.YELLOW + "Datasets to evaluate: ")
    for d in datasets:
        print(Fore.YELLOW + '- {0}: {1}'.format(d, datasets_labels[d]))
    print(Fore.YELLOW + "Algorithms to evaluate: ")
    for alg in included_alg:
        for test in range(len(algorithms_tests[alg])):
            print(Fore.YELLOW+'- {0}: {1}, {2}, {3}'.format(alg, algorithms_tests_labels[alg][test],
                                                            algorithms_file_names[alg][test],
                                                            alg_colors[alg][test]))

    plot_settings = {'datasets_labels': datasets_labels,
                     'datasets_titles': datasets_titles,
                     'algorithms_tests_labels': algorithms_tests_labels,
                     'alg_colors': alg_colors}

    if args.png:
        FORMAT = '.png'

    eval_uid = \
        '__'.join(list(itertools.chain.from_iterable(list(plot_settings['algorithms_tests_labels'].values())))) + \
        datetime.now().strftime("%Y%m%d%H%M")

    n_trials = 1
    if args.mul_trials:
        print(Fore.YELLOW +
              "We will ananlyze multiple trials #{0}".format(args.mul_trials))
        n_trials = args.mul_trials

    need_odometry_error = args.odometry_error_per_dataset or args.overall_odometry_error
    if need_odometry_error:
        print(Fore.YELLOW + "Will calculate odometry errors")

    print("#####################################")
    print(">>> Loading and calculating errors....")
    print("#####################################")
    # organize by configuration
    config_trajectories_list = []
    config_multierror_list = []
    dataset_boxdist_map = {}

    for d in datasets:
        dataset_boxdist_map[d] = boxplot_distances
    for config_i in included_alg:
        cur_trajectories_i = []
        cur_multierror_i = []
        for test_i in range(len(algorithms_tests[alg])):
            cur_trajectories_test_i = []
            cur_multierror_test_i = []
            for d in datasets:
                print(Fore.RED + "--- Processing {0}-{1}-{2}... ---".format(
                    config_i, d, algorithms_tests_labels[config_i][test_i]))

                experiment_name = args.platform + '_' + config_i + '_' + d
                trace_dir = os.path.join(args.results_dir,
                                         args.platform, config_i, experiment_name,
                                         algorithms_tests_labels[config_i][test_i])

                print(experiment_name)
                print(trace_dir)
                assert os.path.exists(trace_dir), \
                    "{0} not found.".format(trace_dir)

                trajectory_list, mt_error = analyze_multiple_trials(
                    trace_dir, algorithms_file_names[config_i][test_i], n_trials,
                    recalculate_errors=args.recalculate_errors,
                    preset_boxplot_distances=dataset_boxdist_map[d],
                    preset_boxplot_percentages=boxplot_percentages,
                    compute_odometry_error=need_odometry_error)
                if not dataset_boxdist_map[d] and trajectory_list:
                    print("Assign the boxplot distances for {0}...".format(d))
                    dataset_boxdist_map[d] = trajectory_list[0].preset_boxplot_distances
                print('Trajectory List\n')
                print(trajectory_list)
                print(mt_error)
                for traj in trajectory_list:
                    traj.alg = algorithms_tests_labels[config_i][test_i]
                    traj.dataset_short_name = d
                mt_error.saveErrors()
                mt_error.cache_current_error()
                mt_error.uid = '_'.join([args.platform, algorithms_tests_labels[config_i][test_i],
                                         d, str(n_trials)])
                mt_error.alg = algorithms_tests_labels[config_i][test_i]
                mt_error.dataset = d
                cur_trajectories_test_i.append(trajectory_list)
                cur_multierror_test_i.append(mt_error)

            cur_trajectories_i.append(cur_trajectories_test_i)
            cur_multierror_i.append(cur_multierror_test_i)
        config_trajectories_list.append(cur_trajectories_i)
        config_multierror_list.append(cur_multierror_i)

    print('Trajectories\n')
    print(config_trajectories_list)
    print(config_multierror_list)
    # organize by dataset name
    dataset_trajectories_list = []
    dataset_multierror_list = []

    for ds_idx, dataset_nm in enumerate(datasets):
        print(ds_idx)
        print(dataset_nm)
        dataset_trajs = [config_trajectories_list[0][test_i][ds_idx]
                         for test_i in range(len(algorithms_tests[alg]))]
        dataset_trajectories_list.append(dataset_trajs)
        dataset_multierrors = [config_multierror_list[0][test_i][ds_idx]
                               for test_i in range(len(algorithms_tests[alg]))]
        dataset_multierror_list.append(dataset_multierrors)

    print(dataset_trajectories_list)
    print(dataset_multierror_list)

    print("#####################################")
    print(">>> Analyze different error types...")
    print("#####################################")
    print(Fore.RED + ">>> Processing absolute trajectory errors...")
    tests_label_list = []
    if args.rmse_table:
        rmse_table = {}
        rmse_table['values'] = []
        for config_mt_error in config_multierror_list[0]:
            print('con\n')
            print(config_mt_error)
            cur_trans_rmse = []
            for mt_error_d in config_mt_error:
                print("> Processing {0}".format(mt_error_d.uid))
                if args.rmse_median_only or n_trials == 1:
                    cur_trans_rmse.append(
                        "{:3.3f}".format(
                            mt_error_d.abs_errors['rmse_trans_stats']['median']))
                else:
                    cur_trans_rmse.append(
                        "{:3.3f}, {:3.3f} ({:3.3f} - {:3.3f})".format(
                            mt_error_d.abs_errors['rmse_trans_stats']['mean'],
                            mt_error_d.abs_errors['rmse_trans_stats']['median'],
                            mt_error_d.abs_errors['rmse_trans_stats']['min'],
                            mt_error_d.abs_errors['rmse_trans_stats']['max']))
            rmse_table['values'].append(cur_trans_rmse)
            tests_label_list.append(mt_error_d.alg)
        rmse_table['rows'] = tests_label_list
        rmse_table['cols'] = datasets
        print('\n--- Generating RMSE tables... ---')
        res_writer.write_tex_table(
            rmse_table['values'], rmse_table['rows'], rmse_table['cols'],
            os.path.join(output_dir, args.platform + '_translation_rmse_' +
                         eval_uid + '.txt'))


    # TODO: If you wish to have multiple n_trials
    # if args.rmse_boxplot and n_trials > 1:
    #     print('Hello')
    #     rmse_plot_alg_test = []

    print(Fore.RED + ">>> Collecting odometry errors per dataset...")
    print(algorithms_tests_labels)
    print(alg_colors)
    if args.odometry_error_per_dataset:
        dataset_rel_err = {}
        dataset_rel_err = collect_odometry_error_per_dataset(
            dataset_multierror_list, datasets)
        print(Fore.MAGENTA +
              '--- Generating relative (KITTI style) error plots ---')
        plot_odometry_error_per_dataset(dataset_rel_err, datasets, included_alg, algorithms_tests, tests_label_list,
                                        datasets_res_dir, plot_settings)
    print(Fore.GREEN + "<<< .... processing odometry errors done.\n")

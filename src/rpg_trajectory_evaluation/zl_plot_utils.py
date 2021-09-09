#!/usr/bin/env python3

__author__ = 'Zachary Lacey'
__email__ = 'zlacey@gmail.com'
__date__ = 'September 9, 2021'

__version__ = '0.1.0'
__status__ = 'Prototype'

import os
import yaml
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd


def generate_error_tables(dataset_trajectories, evaluation_uid, dataset_names_list, included_algorithm_names,
                          algorithm_test_names_dict, datasets_output_directory, plot_idx=0):
    """ Generate Error Tables

    This method generates the figures that depicts the Mean, Median and RMSE of the translation and rotational error
    (with respect to the ground truth Motion Capture pose [position and orientation]) vs the distance traveled.

    Args:
         dataset_trajectories : `list`
            A nested list that contains the trajectory information for each test configurations in each dataset.

            - Example: dataset_trajectories = [
                [<object rovio-test-0>, <object rovio-test-1>, <object rovio-test-2>, <object rovio-test-3>],   # MH_01
                [<object rovio-test-0>, <object rovio-test-1>, <object rovio-test-2>, <object rovio-test-3>],   # MH_03
                [<object rovio-test-0>, <object rovio-test-1>, <object rovio-test-2>, <object rovio-test-3>] ]  # MH_05

        evaluation_uid : `str`
            A string which contains the evaluated uid that will be included in the output file name.

            - Example: evaluation_uid = 'rovio-test-0-1-2-3__sv02-test-3-4__%Y%m%d%H%M'

        dataset_names_list : `list`
            A list that contains the datasets that the user wants to generates figures for. The example shows the setup
            for the some of the Euroc Datasets.

            - Example: dataset_names = ['MH_01', 'MH_03', 'MH_05']

        included_algorithm_names : `list`
            A list of the algorithms that we selected to include in the config.yaml file.

        algorithm_test_names_dict : `dict`
            A nested dictionary that contains the test configurations that the user wants to plot in the generated
            figures.

            - Example: algorithm_test_names = {'rovio': [0, 1, 2, 3], 'svo2': [3, 4]}

        datasets_output_directory : `dict`
            A nested dictionary that specifies the path to the output directories of each dataset.

            - Example: dataset_output_directory = {'MH_01': './results/av_euroc_vio_mono/arm_MH_01_results',
                                        'MH_03': './results/av_euroc_vio_mono/arm_MH_03_results',
                                        'MH_05': './results/av_euroc_vio_mono/arm_MH_05_results'}

        plot_idx : `int`
            An integer that is assigned a zero value. This is used to define the first index in the trajectory list.
            In particular, the trajectories_list can be found as a single dimensional list that contains the trajectory
            object.

            - Example: trajectories_list =  [<object of trajectory>]

                       So,

                       trajectories_list[plot_idx] = <object of trajectory>

    """
    # Decimal Precision of the Error Metric values within the generated Error Metrics Table
    decimal_precision = 4

    # For each dataset
    for dataset_idx, dataset_name in enumerate(dataset_names_list):
        output_directory = datasets_output_directory[dataset_name]
        print("Creating Tables for {0}...".format(dataset_name))
        data_trajectories = dataset_trajectories[dataset_idx]

        num_distances = len(data_trajectories[0][plot_idx].rel_errors)

        merge_cells_dict = {}
        merge_test_cells_dict = {}

        # Data Frame  that holds the Error Metric results
        df = pd.DataFrame()

        # Algorithm and Test # Columns
        algorithm_col_list = []
        test_col_list = []

        first_time_bool = True
        num_tests_list = []

        # Starts on the third row (first two rows of the table are the headers)
        iteration = 2

        # For each algorithm
        for alg_idx, algo in enumerate(included_algorithm_names):
            # Lists that will contain the empty cell locations in the Dataframe that we want to eventually merge.
            merge_cell_alg_list = []
            merge_cell_alg_test_list = []

            # Number of test configuration for the current algorithm
            num_tests = len(algorithm_test_names_dict[algo])
            num_tests_list.append(num_tests)

            # If the first time in algorithm for loop
            if first_time_bool:
                prev_num_tests = num_tests_list[0]
                first_time_bool = False
            else:
                prev_num_tests = num_tests_list[alg_idx - 1]

            # First value in the current algorithm column list is the current algorithm name
            algorithm_col_list.append(algo)

            # The remaining rows values in the algorithm column list are made empty cells and we keep track of these
            # cell locations to eventually use for merging these empty cells
            alg_cells = ((num_distances * prev_num_tests + prev_num_tests) * alg_idx + 2, 0)
            merge_cell_alg_list.append(alg_cells)

            for idx in range(num_distances * num_tests + num_tests - 1):
                alg_cells = ((num_distances * prev_num_tests + 1) * alg_idx + alg_idx + idx + 3, 0)
                merge_cell_alg_list.append(alg_cells)
                algorithm_col_list.append('')

            merge_cells_dict[algo] = merge_cell_alg_list

            # Configures the Test Column (assigns empty cells to the test column in the Dataframe and keeps track of the
            # cell locations to eventually use for merging these empty cells).
            for test_idx, test in enumerate(algorithm_test_names_dict[algo]):
                merge_cell_tests_list = []

                test_col_list.append(test)
                for idx in range(num_distances):
                    test_cells = (iteration, 1)
                    iteration += 1

                    merge_cell_tests_list.append(test_cells)

                    test_col_list.append('')

                test_cells = (iteration, 1)
                iteration += 1

                merge_cell_tests_list.append(test_cells)
                merge_cell_alg_test_list.append(merge_cell_tests_list)

            merge_test_cells_dict[algo] = merge_cell_alg_test_list

        df['Algorithm'] = algorithm_col_list
        df['Test'] = test_col_list

        dist_col_list = []

        # Translation and Rotation Metric Lists
        translation_mean_list = []
        translation_median_list = []
        translation_rmse_list = []

        rotation_mean_list = []
        rotation_median_list = []
        rotation_rmse_list = []

        # For each trajectory
        for trajectories_list in data_trajectories:
            # For each distance (distances that define the sub-trajectories. i.e., for MH_01, the distances are 8.06 m,
            # 16.12 m, 24.18 m, 32.25 m, and 40.31 m)
            for dist in trajectories_list[plot_idx].rel_errors:
                dist_col_list.append(dist)
                dist_str = "{:3.1f}".format(dist).replace('.', '_')
                dist_fn = os.path.join(
                    trajectories_list[plot_idx].saved_results_dir,
                    'mt_rel_err_' + dist_str + '.yaml')

                rotation_err, translation_err = parse_mt_error_yaml(dist_fn)

                # Obtains the error metric values (rounds these values to the 4th decimal place) and places these values
                # in their respective lists so they can be stored in their respective columns in the Dataframe table.

                # Acquire the values
                rotation_err_mean = round(rotation_err['mean'], decimal_precision)
                rotation_err_median = round(rotation_err['median'], decimal_precision)
                rotation_err_rmse = round(rotation_err['rmse'], decimal_precision)

                translation_err_mean = round(translation_err['mean'], decimal_precision)
                translation_err_median = round(translation_err['median'], decimal_precision)
                translation_err_rmse = round(translation_err['rmse'], decimal_precision)

                # Add the values to the lists
                rotation_mean_list.append(rotation_err_mean)
                rotation_median_list.append(rotation_err_median)
                rotation_rmse_list.append(rotation_err_rmse)

                translation_mean_list.append(translation_err_mean)
                translation_median_list.append(translation_err_median)
                translation_rmse_list.append(translation_err_rmse)

            # Here we now also obtain the overall error metric values across the entire trajectory.
            overall_fn = os.path.join(
                trajectories_list[plot_idx].saved_results_dir,
                'mt_rel_err_overall.yaml')

            rotation_err_overall, translation_err_overall = parse_mt_error_yaml(overall_fn)

            # Obtains the error metric values (rounds these values to the 4th decimal place) and places these values
            # in their respective lists so they can be stored in their respective columns in the Dataframe table.

            # Acquire the Overall values
            rotation_err_mean_overall = round(rotation_err_overall['mean'], decimal_precision)
            rotation_err_median_overall = round(rotation_err_overall['median'], decimal_precision)
            rotation_err_rmse_overall = round(rotation_err_overall['rmse'], decimal_precision)

            translation_err_mean_overall = round(translation_err_overall['mean'], decimal_precision)
            translation_err_median_overall = round(translation_err_overall['median'], decimal_precision)
            translation_err_rmse_overall = round(translation_err_overall['rmse'], decimal_precision)

            # Add the Overall values to the lists
            rotation_mean_list.append(rotation_err_mean_overall)
            rotation_median_list.append(rotation_err_median_overall)
            rotation_rmse_list.append(rotation_err_rmse_overall)

            translation_mean_list.append(translation_err_mean_overall)
            translation_median_list.append(translation_err_median_overall)
            translation_rmse_list.append(translation_err_rmse_overall)

            dist_col_list.append('Overall')

        # Add the Error Metric Columns to the Dataframe
        df['Distance Traveled (m)'] = dist_col_list
        df['Mean (m)'] = translation_mean_list
        df['Median (m)'] = translation_median_list
        df['RMSE (m)'] = translation_rmse_list

        df['Mean (deg)'] = rotation_mean_list
        df['Median (deg)'] = rotation_median_list
        df['RMSE (deg)'] = rotation_rmse_list

        print(df)

        fig = plt.figure(figsize=(28, 12))
        ax = fig.gca()
        ax.axis('off')
        df_rows, df_cols = df.shape

        table = ax.table(cellText=np.vstack([[' ', ' ', ' ', 'Translation Error', ' ', ' ', 'Rotation Error', ' ', ' '],
                                            df.columns, df.values]),
                         cellColours=[['none'] * df_cols] * (2 + df_rows), bbox=[0, 0, 1, 1])

        table.auto_set_font_size(False)
        table.set_fontsize(16)
        table.scale(2, 2)

        # need to draw here so the text positions are calculated
        fig.canvas.draw()

        mergecells(table, (1, 0), (0, 0))

        mergecells(table, (1, 1), (0, 1))

        mergecells(table, (1, 2), (0, 2))

        merge_cells(table, [(0, 3), (0, 4), (0, 5)])
        merge_cells(table, [(0, 6), (0, 7), (0, 8)])

        for alg_idx, algo in enumerate(included_algorithm_names):
            merge_cell_alg = merge_cells_dict[algo]

            merge_cells(table, merge_cell_alg)

            for test_idx, test in enumerate(algorithm_test_names_dict[algo]):
                merge_cell_test = merge_test_cells_dict[algo][test_idx]
                merge_cells(table, merge_cell_test)

        fig.savefig(output_directory + '/' + evaluation_uid + '_' + dataset_name +
                    '_trans_rot_error_table.png')


def merge_cells(table, cells):
    """
        Merge N matplotlib.Table cells

        Args:
            table : `matplotlib.Table`
                The table that contains the Error Metric Values

            cells : `list[set]`
                List of a tuples that represent the table coordinates

                - Example: cells = [(0, 1), (0, 0), (0, 2)]

    """
    cells_array = [np.asarray(c) for c in cells]
    h = np.array([cells_array[i + 1][0] - cells_array[i][0] for i in range(len(cells_array) - 1)])
    v = np.array([cells_array[i + 1][1] - cells_array[i][1] for i in range(len(cells_array) - 1)])

    # if it's a horizontal merge, all values for `h` are 0
    if not np.any(h):
        # sort by horizontal coord
        cells = np.array(sorted(list(cells), key=lambda vertical: vertical[1]))
        edges = ['BTL'] + ['BT' for i in range(len(cells) - 2)] + ['BTR']
    elif not np.any(v):
        cells = np.array(sorted(list(cells), key=lambda horizontal: horizontal[0]))
        edges = ['TRL'] + ['RL' for i in range(len(cells) - 2)] + ['BRL']
    else:
        raise ValueError("Only horizontal and vertical merges allowed")

    for cell, e in zip(cells, edges):
        table.get_celld()[cell[0], cell[1]].visible_edges = e

    txts = [table.get_celld()[cell[0], cell[1]].get_text() for cell in cells]

    tpos = [np.array(t.get_position()) for t in txts]

    # transpose the text of the left cell
    trans = (tpos[-1] - tpos[0]) / 2
    # didn't had to check for ha because I only want ha='center'
    txts[0].set_transform(mpl.transforms.Affine2D().translate(*trans))


def mergecells(table, ix0, ix1):
    ix0, ix1 = np.asarray(ix0), np.asarray(ix1)
    d = ix1 - ix0
    if not (0 in d and 1 in np.abs(d)):
        raise ValueError("ix0 and ix1 should be the indices of adjacent cells. ix0: %s, ix1: %s" % (ix0, ix1))

    if d[0] == -1:
        edges = ('BRL', 'TRL')
    elif d[0] == 1:
        edges = ('TRL', 'BRL')
    elif d[1] == -1:
        edges = ('BTR', 'BTL')
    else:
        edges = ('BTL', 'BTR')

    # hide the merged edges
    for ix, e in zip((ix0, ix1), edges):
        table.get_celld()[ix[0], ix[1]].visible_edges = e

    txts = [table.get_celld()[ix[0], ix[1]].get_text() for ix in (ix0, ix1)]
    tpos = [np.array(t.get_position()) for t in txts]

    # center the text of the 0th cell between the two merged cells
    trans = (tpos[1] - tpos[0])/2
    if trans[0] > 0 and txts[0].get_ha() == 'right':
        # reduce the transform distance in order to center the text
        trans[0] /= 2
    elif trans[0] < 0 and txts[0].get_ha() == 'right':
        # increase the transform distance...
        trans[0] *= 2

    txts[0].set_transform(mpl.transforms.Affine2D().translate(*trans))

    # hide the text in the 1st cell
    txts[1].set_visible(False)


def parse_mt_error_yaml(multi_traj_error_yaml):
    """

    This method is for parsing the Translation and Rotation Error Metrics from the multiple trial error yaml files
    (saved results from the calculations).

    Args:
        multi_traj_error_yaml : `str`
            A String of the file path to the multiple trial error yaml files (saved results from the calculations).

    Returns:
        rotation_error : `dict`
            A dictionary that contains the rotation error metrics.

            - Example: rotation_error = { 'max':
                                          'mean':
                                          'median':
                                          'min':
                                          'num_samples':
                                          'rmse':
                                          'std':
                                        }

        translation_error : `dict`
            A dictionary that contains the translation error metrics.

            - Example: translation_error = { 'max':
                                             'mean':
                                             'median':
                                             'min':
                                             'num_samples':
                                             'rmse':
                                             'std':
                                           }

    """
    with open(multi_traj_error_yaml) as stream:
        yaml_data = yaml.safe_load(stream)

    rotation_error = yaml_data['rot']
    translation_error = yaml_data['trans']

    return rotation_error, translation_error

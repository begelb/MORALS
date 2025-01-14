from MORALS.mg_utils import MorseGraphOutputProcessor
from MORALS.data_utils import *
from MORALS.dynamics_utils import *
import argparse

np.set_printoptions(suppress=True)

import os
import warnings
warnings.filterwarnings("ignore")
from collections import defaultdict
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support

def generate_key(design, num_layers, step):
    return design + "_" + str(num_layers) + "_" + str(step)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config_dir',help='Directory of config files',type=str,default='config/')
    argparser.add_argument('--out_dir',type=str, default="humanoid_learned1k")
    argparser.add_argument('--config', type=str)
    argparser.add_argument('--test_data',type=str, default="")
    argparser.add_argument('--test_labels',type=str, default="")
    argparser.add_argument('--fine_tune',help='Evaluate for fine_tuned model', action='store_true')

    args = argparser.parse_args()

    trajectories = None
    successful_final_conditions_true = None

    output_dir = "output/" + args.out_dir + "/"
    precisions = {}
    recalls = {}

    config_fname = args.config_dir + args.config

    # for dir in tqdm(os.listdir(output_dir)):
    #     if dir.endswith('.txt'): continue
    #     config_fname = os.path.join(output_dir, dir, "config.txt")

    with open(config_fname) as f:
        config = eval(f.read())

    k = generate_key(config['experiment'], config['num_layers'], config['step'])
    if k not in precisions:
        precisions[k] = []
        recalls[k] = []
    
   # try:
    mg_out_utils = MorseGraphOutputProcessor(config, output_dir)
    # except FileNotFoundError:
    #     continue
    # except ValueError:
    #     print("ValueError from: ", config_fname)
    #     continue
    # except IndexError:
    #     print("IndexError from: ", config_fname)
    #     continue

    if mg_out_utils.get_num_attractors() <= 1:
        print('only one attractor')
        exit()
    
    dynamics = DynamicsUtils(config, args.fine_tune)
    if trajectories is None:
        print("Assumes all configs in this experiment share the same dataset")
        if args.test_data == "" or args.test_labels == "":
            trajectories = TrajectoryDataset(config)
            successful_final_conditions_true = trajectories.get_successful_final_conditions()
            unsuccessful_final_conditions_true = trajectories.get_unsuccessful_final_conditions()
        else:
            config["data_dir"] = args.test_data
            config["labels_dir"] = args.test_labels
            trajectories = TrajectoryDataset(config)
            successful_final_conditions_true = trajectories.get_successful_final_conditions()
            unsuccessful_final_conditions_true = trajectories.get_unsuccessful_final_conditions()

    encoded_successful_final_conditions = dynamics.encode(successful_final_conditions_true)
    encoded_unsuccessful_final_conditions = dynamics.encode(unsuccessful_final_conditions_true)

    '''
    success_counts = defaultdict(int)  
    failure_counts = defaultdict(int)

    for i in range(encoded_successful_final_conditions.shape[0]):
        success_counts[mg_out_utils.which_morse_node(encoded_successful_final_conditions[i])] += 1
    for i in range(encoded_unsuccessful_final_conditions.shape[0]):
        failure_counts[mg_out_utils.which_morse_node(encoded_unsuccessful_final_conditions[i])] += 1
    
    inside_nodes = set()
    for key in success_counts.keys():
        if key != -1 and success_counts[key] > failure_counts[key]:
            inside_nodes.add(key)
    '''

    not_inside_bounds_count = 0
    inside_nodes = set()
    for i in range(encoded_successful_final_conditions.shape[0]):
        point = encoded_successful_final_conditions[i]
        if mg_out_utils.check_in_bounds(point):
            inside_nodes.add(mg_out_utils.which_morse_node(encoded_successful_final_conditions[i]))
        else:
            not_inside_bounds_count += 1
    if -1 in inside_nodes:
        inside_nodes.remove(-1)
    print('Not inside bounds count ', not_inside_bounds_count)

    encoded_initial_states = dynamics.encode(np.vstack([elt[0] for elt in trajectories]))
    ground_truth = []
    predicted = []
    for i in range(len(trajectories)):
        ground_truth.append(trajectories.get_label(i))
        predicted_roa = mg_out_utils.which_morse_node(encoded_initial_states[i])
        if predicted_roa in inside_nodes:
            predicted.append(1)
        else:
            predicted.append(0)
    
    ground_truth = np.array(ground_truth)
    predicted = np.array(predicted)
    # print first few ground_truth
    print(ground_truth[:25])
    # print first few predicted
    print(predicted[:25])

    precision, recall, f1, _ = precision_recall_fscore_support(ground_truth, predicted, average='binary')
    precisions[k].append(precision)
    recalls[k].append(recall)

    # Write mean and stddev to file
    if args.fine_tune:
        output_dir += "MG_fine_tune/"
    with open(output_dir + "precision_recall.txt", "w") as f:
        for k in precisions:
            f.write(k + "," + str(np.mean(precisions[k])) + "," + str(np.std(precisions[k])) + "," + str(np.mean(recalls[k])) + "," + str(np.std(recalls[k])) + "\n")



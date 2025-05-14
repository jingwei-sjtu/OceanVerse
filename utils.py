import numpy as np
import torch
import random
import pdb

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score

def custom_mean_absolute_percentage_error(y_true, y_pred, threshold=5):
    mask = y_true >= threshold
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]
    mape = mean_absolute_percentage_error(y_true_filtered, y_pred_filtered)    
    return mape

def calculate_metrics(pred_result, ground_truth, metric_list=['MAE', 'MAPE', 'MSE', 'RMSE', 'R2', 'STD']):
    pred_result = pred_result.cpu().detach().numpy()
    ground_truth = ground_truth.cpu().detach().numpy()
    metric_result_dict = {}
    for metric in metric_list:
        if metric == 'MAE':
            result = mean_absolute_error(ground_truth, pred_result)
        elif metric == 'MAPE':
            result = custom_mean_absolute_percentage_error(ground_truth, pred_result)
        elif metric == 'MSE':
            result = mean_squared_error(ground_truth, pred_result)
        elif metric == 'RMSE':
            result = np.sqrt(mean_squared_error(ground_truth, pred_result))
        elif metric == 'R2':
            result = r2_score(ground_truth, pred_result)
        elif metric == 'STD':
            result = np.std(pred_result)
        metric_result_dict[metric] = result
    return metric_result_dict

def print_mean_std(result_list, name_list):
    for i, result_i in enumerate(result_list):
        name_i = name_list[i]
        result_mean = np.mean(np.array(result_i))
        result_std = np.std(np.array(result_i))
        if name_i == 'MAPE':
            print(f"MAPE: {result_mean * 100:.4f}±{result_std*100:.4f}")
        else:
            print(f"{name_i}: {result_mean:.4f}±{result_std:.4f}")

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def print_isolated_node_ratio(data, year):
    num_nodes = data['feature'].shape[0]
    degrees = torch.zeros(num_nodes, dtype=torch.long)
    edge_index = data['edge_index'].T
    for edge in edge_index:
        degrees[edge[0]] += 1
        degrees[edge[1]] += 1
    num_isolated_nodes = (degrees == 0).sum().item()
    isolation_ratio = num_isolated_nodes / num_nodes
    print(f"Year={year}, isolation ratio is {isolation_ratio*100:.2f}")

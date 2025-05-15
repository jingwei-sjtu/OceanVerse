import argparse
import gzip
import torch
import pdb

import torch.nn as nn
from torch.utils.data import Dataset, random_split, DataLoader
from torch.optim.lr_scheduler import LambdaLR, StepLR
from torch_geometric.loader import NeighborLoader
from models import *
from utils import *
import os
import random
from tqdm import tqdm
import pandas as pd
import xgboost as xgb



def main():
    args = get_args()
    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")

    dataset_param = {'CESM2-omip1':{'num_years': 62, 'start_year': 1948},'CESM2-omip2':{'num_years': 61, 'start_year': 1958},'GFDL_ESM4':{'num_years': 95, 'start_year': 1920}}
    num_years = dataset_param[args.dataset]['num_years']
    start_year = dataset_param[args.dataset]['start_year']
    dataset_path = f'data/{args.dataset}/'

    set_random_seed(args.seed)
    if args.seed is None: 
        args.seed = random.randint(0, 10000)
    print(f"INFO: Using seed {args.seed}")
    print(args)
    

    if args.split == 'spatial':
        indices_path = os.path.join(dataset_path, f'split/spatial_split.pt')
        indices = torch.load(indices_path)
        train_indices = indices['train_indices']
        val_indices = indices['val_indices']
        train_indices = train_indices[:num_years]
        val_indices = val_indices[:num_years]
        get_indices = get_valid_indices_spatial

    elif args.split == 'temporal':
        train_year_list = [num for num in range(start_year, start_year + int(num_years*0.7)+1)]
        val_year_list = [num for num in range(start_year + int(num_years*0.7)+1, start_year + num_years)]
        get_indices = get_valid_indices_temporal
        train_indices = torch.arange(0, 42491)
        val_indices = torch.arange(0, 42491)

    elif args.split == 'random':
        indices_path = os.path.join(dataset_path, f'split/random_split_seed{args.seed}.pt')
        indices = torch.load(indices_path)
        train_indices = indices['train_indices']
        val_indices = indices['val_indices']
        get_indices = get_valid_indices



    year_list = [num for num in range(start_year, num_years + start_year)]
    random.shuffle(year_list)
    count = 0
    if args.split == 'temporal':
        train_year_list = [num for num in range(start_year, start_year + int(num_years*0.7)+1)]
        val_year_list = [num for num in range(start_year + int(num_years*0.7)+1, start_year + num_years)]
        year_list = train_year_list

    for year in tqdm(year_list):
        file_path = os.path.join(dataset_path, 'graph/' + str(year) + '.pt')
        data = torch.load(file_path)
        # indices=torch.load('/home/shenjj/digital_twin/GFDL-ESM4_graph/split_indices/baseline_model_MLP_256_2_MLPsvd_all_9698.pt')
    
        indices = get_indices(data.y, year, train_indices, start_year)
        train_loader = NeighborLoader(data, num_neighbors=[0], batch_size=args.batch_size, input_nodes=indices)
        for i, batch in enumerate(train_loader):
            batch = batch.to(device)
            x_factor = batch.x
            x_time_series = batch.time_series_profile
            x_geo = batch.x_geo.unsqueeze(1).repeat(1, 33, 1)
            x = torch.cat([x_factor, x_time_series, x_geo], dim=2)
            x = x.reshape(-1, x.shape[2])
            y = batch.y.reshape(-1, 1)
            not_nan_indices = ~torch.isnan(y).squeeze()
            if count == 0:
                x_train = x[not_nan_indices]
                y_train = y[not_nan_indices]
            else:
                x_train = torch.cat([x_train, x[not_nan_indices]], dim=0)
                y_train = torch.cat([y_train, y[not_nan_indices]], dim=0)
    if args.split == 'temporal':
        year_list = val_year_list
    for year in tqdm(year_list):
        file_path = os.path.join(dataset_path, 'graph/' + str(year) + '.pt')
        data = torch.load(file_path)
        indices = get_indices(data.y, year, train_indices, start_year)
        val_loader = NeighborLoader(data, num_neighbors=[0], batch_size=args.batch_size, input_nodes=indices)
        for i, batch in enumerate(val_loader):
            batch = batch.to(device)
            x_factor = batch.x
            x_time_series = batch.time_series_profile
            x_geo = batch.x_geo.unsqueeze(1).repeat(1, 33, 1)
            x = torch.cat([x_factor, x_time_series, x_geo], dim=2)
            x = x.reshape(-1, x.shape[2])
            y = batch.y.reshape(-1, 1)
            not_nan_indices = ~torch.isnan(y).squeeze()
            if count == 0:
                x_val = x[not_nan_indices]
                y_val = y[not_nan_indices]
            else:
                x_val = torch.cat([x_val, x[not_nan_indices]], dim=0)
                y_val = torch.cat([y_val, y[not_nan_indices]], dim=0)
        

    x_train = x_train.cpu().numpy()
    y_train = y_train.cpu().numpy()
    x_val = x_val.cpu().numpy()
    y_val = y_val.cpu().numpy()
    train_dmatrix = xgb.DMatrix(x_train, label=y_train)
    val_dmatrix = xgb.DMatrix(x_val, label=y_val)

    evls = [(val_dmatrix, 'valid')]
    params = {
        'objective': 'reg:squarederror',  
        'colsample_bytree': 0.3,
        'learning_rate': 0.01,
        'max_depth': 5,
        'alpha': 10,
        'seed': args.seed,
    }
    num_round = args.num_epochs
    early_stopping_rounds = args.max_patience
    xg_reg = xgb.train(params, train_dmatrix, num_round, evals=evls, early_stopping_rounds=early_stopping_rounds)

    ## inference
    for i in tqdm(range(num_years)):
        year = start_year + i
        file_path = os.path.join(dataset_path, 'graph/' + str(year) + '.pt')
        data = torch.load(file_path)
        x_factor = data.x
        x_time_series = data.time_series_profile
        x_geo = data.x_geo.unsqueeze(1).repeat(1, 33, 1)
        x = torch.cat([x_factor, x_time_series, x_geo], dim=2)
        x = x.reshape(-1, x.shape[2])
        if i == 0:
            x_all = x
        else:
            x_all = torch.cat([x_all, x], dim=0)
        year = x[:, -1] * num_years + start_year
        xyz = x[:, -5:-2]
        latitude = torch.arcsin(xyz[:,-1])
        latitude = torch.rad2deg(latitude)
        longitude = torch.arctan2(xyz[:,1],xyz[:,0])
        longitude = torch.rad2deg(longitude)
        geo_information = torch.stack((latitude, longitude, year), dim=1)
        if i == 0:
            geo_all = geo_information
        else:
            geo_all = torch.cat([geo_all, geo_information], dim=0)
    
    x_pred = x_all.cpu().numpy()
    x_pred = xgb.DMatrix(x_pred)
    all_pred = xg_reg.predict(x_pred)
    all_geoinformation = geo_all.cpu().numpy()
    depths = np.arange(1, 34)  # 
    depths = np.tile(np.arange(1, 34), len(all_geoinformation))
    
    depths_column = pd.Series(depths, name='depth')
    oxygen_column = pd.Series(all_pred.flatten(), name='oxygen')
    df = pd.DataFrame(all_geoinformation, columns=['latitude', 'longitude', 'year'])
    df['depth'] = depths_column
    df['oxygen'] = oxygen_column
    df['latitude'] = df['latitude'].round(1)
    df['longitude'] = df['longitude'].round(1)
    df['year'] = df['year'].round(0)
    df['depth'] = df['depth'].round(2)

    try:
        all_df = pd.read_csv(f'infer_result/all_df_empty_{args.dataset}.csv')
    except:
        time_range = np.arange(start_year, start_year+num_years, 1)
        longitude_range = np.arange(-179.5, 180, 1)
        latitude_range = np.arange(-89.5, 90, 1)
        depth_values = df['depth'].unique()
        coordinates = [(time, depth, lat, lon) 
                    for time in time_range 
                    for depth in depth_values
                    for lat in latitude_range 
                    for lon in longitude_range 
                    ]
        all_df = pd.DataFrame(coordinates, columns=['year', 'depth', 'latitude', 'longitude'])
        all_df.to_csv(f'infer_result/all_df_empty_{args.dataset}.csv', index=False)

    result = pd.merge(all_df, df, on=['year', 'depth', 'latitude', 'longitude'], how='left')
    result_np = result['oxygen'].values
    result_np = result_np.reshape((num_years,33,180,360))
    np.save(f'infer_result/inference_by_{args.dataset}_{args.split}_XGBoost_model_{args.seed}', result_np)


def get_args():
    parser = argparse.ArgumentParser(description='Oceanverse')
    parser.add_argument('--dataset', type=str, default='CESM2-omip1', help='omip1, omip2, GFDL')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of layers in the Transformer model')
    parser.add_argument('--batch_size', type=int, default=42491, help='Batch size for training')
    parser.add_argument('--max_patience', type=int, default=10, help='Maximum patience for early stopping')
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate for training')
    parser.add_argument('--num_epochs', type=int, default=2000, help='Number of epochs for training')
    parser.add_argument('--seed', type=int, default=1041, help='Random seed for reproducibility')
    parser.add_argument('--gpu', type=int, default=0, help='Index of GPU to use for training')
    parser.add_argument('--valid_step', type=int, default=20, help='Validation step')
    parser.add_argument('--geo_dim', type=int, default=5, help='The dimension of the DO geo factor')
    parser.add_argument('--hidden_dim', type=int, default=256, help='The dimension of the hidden layer')
    parser.add_argument('--time_length', type=int, default=11, help='The length of the time series')
    parser.add_argument('--num_deep_layers', type=int, default=33, help='The number of deep layers')
    parser.add_argument('--edge_dim', type=int, default=3, help='The number of edge features')
    parser.add_argument('--input_dim', type=int, default=8, help='The number of node features')
    parser.add_argument('--tips', type=str, default='None', help='tips') 
    parser.add_argument('--model', type=str, default='MLP', help='select model from MLP, Transformer, LSTM, Oxygenerator')
    parser.add_argument('--split', type=str, default='temporal', help='temporal, spatial, random')
    parser.add_argument('--grad_lambda', type=float, default=1)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
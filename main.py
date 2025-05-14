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
import numpy as np



def get_valid_indices(data, year, split_indices, start_year):
	data = (~torch.isnan(data)).sum(dim = 1)
	valid_indices = torch.nonzero(data)[:, 0]
	split_indices = split_indices[year - start_year]
	valid_indices = torch.tensor(list(set(valid_indices.numpy()) & set(split_indices)))
	return valid_indices

def get_valid_indices_spatial(data, year, split_indices, start_year):
	data = (~torch.isnan(data)).sum(dim = 1)
	valid_indices = torch.nonzero(data)[:, 0]
	split_indices = split_indices[year - start_year]
	valid_indices = torch.tensor(list(set(valid_indices.numpy()) & set(split_indices.numpy())))

	return valid_indices


def get_valid_indices_temporal(data, year, split_indices, start_year):
	data = (~torch.isnan(data)).sum(dim = 1)
	valid_indices = torch.nonzero(data)[:, 0]
	split_indices = np.arange(0, 42491)
	valid_indices = torch.tensor(list(set(valid_indices.numpy()) & set(split_indices)))
	return valid_indices





def baseline_train(args, model, get_valid_indices, train_indices, val_indices, start_year, num_years, dataset_path,optimizer, scheduler, device='cuda'):
    criterion = args.criterion
    year_list = [num for num in range(start_year, start_year + num_years)]
    best_val_loss = 1e10
    if args.split == 'temporal':
        train_year_list = [num for num in range(start_year, start_year + int(num_years*0.7)+1)]
        val_year_list = [num for num in range(start_year + int(num_years*0.7)+1, start_year + num_years)]
    for epoch in range(args.num_epochs):
        model.train()
        iters = 0
        if args.split == 'temporal':
            year_list = train_year_list

        random.shuffle(year_list)
        for year in year_list:
            file_path = os.path.join(dataset_path, 'graph/' + str(year) + '.pt')
            data = torch.load(file_path)
            indices = get_valid_indices(data.y, year, train_indices, start_year)
            train_loader = NeighborLoader(data, num_neighbors=[0], batch_size=args.batch_size, input_nodes=indices)

            for i, batch in enumerate(train_loader):
                batch = batch.to(device)
                result = model(batch.x_geo.float(), batch.x.float(), batch.time_series_profile.float(), batch.edge_index, batch.edge_attr, batch.x_geo.float())
                y = batch.y[:len(batch.input_id)].float()
                mask = ~torch.isnan(y)
                mseLoss_list = criterion(result[:len(batch.input_id)][mask], y[mask]) 
                mean_mseloss, std_mseloss = torch.mean(mseLoss_list), torch.std(mseLoss_list)
                loss = mean_mseloss 
                if loss.isnan().any():
                    pdb.set_trace()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                iters += 1
                if iters % 10 == 0:
                    print(f'Epoch {epoch}, Year_ID {year}, Mean MSE Loss: {round(mean_mseloss.item(),3)}, output_max:{round(result[:len(batch.input_id)][mask].max().item(),3)}, output_min:{round(result[:len(batch.input_id)][mask].min().item(),3)}, lr: {scheduler.get_last_lr()}')

            scheduler.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            if args.split == 'temporal':
                year_list = val_year_list
            count = 0
            for year in tqdm(year_list):
                file_path = os.path.join(dataset_path, 'graph/' + str(year) + '.pt')
                data = torch.load(file_path)
                indices = get_valid_indices(data.y, year, val_indices, start_year)
                val_loader = NeighborLoader(data, num_neighbors=[0], batch_size=args.batch_size, input_nodes=indices)
                for i, batch in enumerate(val_loader):
                    batch = batch.to(device)
                    result = model(batch.x_geo.float(), batch.x.float(), batch.time_series_profile.float(), batch.edge_index, batch.edge_attr, batch.x_geo.float())
                    pred = result[:len(batch.input_id)].float()
                    y = batch.y[:len(batch.input_id)].float()
                    mask = ~torch.isnan(y)
                    loss = criterion(pred[mask], y[mask])
                    val_loss += loss.item()
                    count += 1

            val_loss = val_loss / count
            print(f'Epoch {epoch}, Validation Loss: {val_loss}, tips: {args.tips}')
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience = 0
                torch.save(model.state_dict(), f'model_pkl/baseline_model_{args.split}_{args.model}_{args.hidden_dim}_{args.num_layers}_{args.tips}_{args.seed}.pt')
            else:
                patience += 1
                if patience >= args.max_patience:
                    break_outer = True
                    return
    return

def oxygenerator_train(args, model, get_valid_indices, train_indices, val_indices, start_year, num_years, dataset_path, optimizer, scheduler, device='cuda'):
    criterion = args.criterion
    year_list = [num for num in range(start_year, start_year + num_years)]
    if args.split == 'temporal':
        train_year_list = [num for num in range(start_year, start_year + int(num_years*0.7)+1)]
        val_year_list = [num for num in range(start_year + int(num_years*0.7)+1, start_year + num_years)]
    patience = 0
    best_val_loss = 1e10
    for epoch_i in range(args.num_epochs):
        random.shuffle(year_list)
        train_loss = 0
        train_nit_loss = 0
        train_pho_loss = 0
        train_spatial_loss = 0
        batch_count = 0

        if args.split == 'temporal':
            year_list = train_year_list

        for year in year_list:
            file_path = os.path.join(dataset_path, 'graph/' + str(year) + '.pt')
            data = torch.load(file_path)
            model.train()
            indices = get_valid_indices(data.y, year, train_indices, start_year)
            if len(indices) == 0:
                continue
            train_loader = NeighborLoader(data, num_neighbors=[-1, -1], batch_size=args.batch_size, input_nodes=indices)
            for i, batch in enumerate(train_loader):
                batch = batch.to(device)
                batch.x.requires_grad_()
                oxygen_pred, spatial_pred = model(batch)
                gradients = torch.autograd.grad(outputs=oxygen_pred, inputs=batch.x, grad_outputs=torch.ones_like(oxygen_pred), create_graph=True)

                pho_indices = torch.nonzero(batch.x[:, :, 6])
                nit_indices = torch.nonzero(batch.x[:, :, 5])
                pho_loss = gradients[0][pho_indices, 6].var() if len(pho_indices) > 1 else 0
                nit_loss = gradients[0][nit_indices, 5].var() if len(nit_indices) > 1 else 0
                nonan_indices = torch.isnan(batch.y) == False
                
                xyz = batch.x_geo[:,:3]
                latitude = torch.arcsin(xyz[:,-1])
                latitude = torch.sin(torch.rad2deg(latitude))
                longitude = torch.arctan2(xyz[:,1],xyz[:,0])
                longitude = torch.rad2deg(longitude)
                longitude = torch.sin(longitude/360*3.1415926)
                spatial_label = torch.cat((latitude.unsqueeze(1), longitude.unsqueeze(1)), dim=1)
                spatial_loss = criterion(spatial_pred, spatial_label)
                loss = criterion(oxygen_pred[nonan_indices], batch.y[nonan_indices].to(torch.float32))
                loss = loss + args.grad_lambda * pho_loss + args.grad_lambda * nit_loss + 0.01 * spatial_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                if len(pho_indices) > 1:
                    train_pho_loss += pho_loss.item()
                if len(nit_indices) > 1:
                    train_nit_loss += nit_loss.item()
                batch_count += 1
                train_spatial_loss += spatial_loss.item()

        train_loss /= batch_count
        train_nit_loss /= batch_count
        train_pho_loss /= batch_count
        train_spatial_loss /= batch_count
        print(f"Epoch ({epoch_i}/{args.num_epochs})NITLoss is {train_nit_loss:.6f}.PHOLoss is {train_pho_loss:.6f}.Loss is {train_loss:.6f}.Spatial_pred Loss is {train_spatial_loss:.6f}")

        if epoch_i % args.valid_epoch == 0:
            model.eval()
            val_loss = 0
            if args.split == 'temporal':
                year_list = val_year_list
            with torch.no_grad():
                batch_count = 0
                for year in year_list:
                    file_path = os.path.join(dataset_path, 'graph/' + str(year) + '.pt')
                    data = torch.load(file_path)
                    indices = get_valid_indices(data.y, year, val_indices, start_year)
                    if len(indices) == 0:
                        continue
                    val_loader = NeighborLoader(data, num_neighbors=[-1, -1], batch_size=args.batch_size, input_nodes=indices)
                    for i, batch in enumerate(val_loader):
                        batch = batch.to(device)
                        oxygen_pred, _ = model(batch)
                        nonan_indices = torch.isnan(batch.y) == False
                        loss = criterion(oxygen_pred[nonan_indices], batch.y[nonan_indices].to(torch.float32))
                        val_loss += loss.item()
                        batch_count += 1
            val_loss /= batch_count
            print(f"Epoch ({epoch_i}/{args.num_epochs}). Val Loss is {val_loss:.6f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch_i
                patience = 0
                torch.save(model.state_dict(), f'model_pkl/baseline_model_{args.split}_{args.model}_{args.hidden_dim}_{args.num_layers}_{args.tips}_{args.seed}.pt')
            else:
                patience += 1
                if patience >= args.max_patience:
                    break_outer = True
                    return
            
    return

def main():
    args = get_args()
    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    if args.seed is None: 
        args.seed = random.randint(0, 10000)
    print(f"INFO: Using seed {args.seed}")
    print(args)
    dataset_param = {'CESM2-omip1':{'num_years': 62, 'start_year': 1948},'CESM2-omip2':{'num_years': 61, 'start_year': 1958},'GFDL_ESM4':{'num_years': 95, 'start_year': 1920}}
    num_years = dataset_param[args.dataset]['num_years']
    start_year = dataset_param[args.dataset]['start_year']
    dataset_path = f'data/{args.dataset}/'

    set_random_seed(args.seed)
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

    if args.model == 'MLP':
        model = Baseline_MLP(factor_dim=args.input_dim, geo_dim=args.geo_dim, hidden_dim=args.hidden_dim, time_len=args.time_length, layer_num=args.num_layers, edge_dim=args.edge_dim, meta_dim=args.geo_dim).to(device)
    elif args.model == 'Transformer':
        model = Baseline_Transformer(factor_dim=args.input_dim, geo_dim=args.geo_dim, hidden_dim=args.hidden_dim, time_len=args.time_length, layer_num=args.num_layers, edge_dim=args.edge_dim, meta_dim=args.geo_dim).to(device)
    elif args.model == 'LSTM':
        model = Baseline_BiLSTM(factor_dim=args.input_dim, geo_dim=args.geo_dim, hidden_dim=args.hidden_dim, time_len=args.time_length, layer_num=args.num_layers, edge_dim=args.edge_dim, meta_dim=args.geo_dim).to(device)
    elif args.model == 'Oxygenerator':
        args.input_dim = args.input_dim * args.num_deep_layers
        model = STModel(args.input_dim, args.hidden_dim, args.num_deep_layers, args.edge_dim, args.num_layers, time_series_dim=args.num_deep_layers).to(device)
        args.valid_epoch = 20
    for name, param in model.named_parameters():
        print('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))
    criterion = nn.MSELoss()
    args.criterion = criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    decay_factor = 0.95
    initial_lr = args.lr
    min_lr = 1e-4
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: decay_factor ** epoch if initial_lr * (decay_factor ** epoch) >= min_lr else min_lr / initial_lr)
    patience = 0
    min_val_loss = 50000000
    break_outer = False
    year_list = [num for num in range(start_year, start_year + num_years)]
    if args.model == 'Oxygenerator':

        oxygenerator_train(args, model, get_indices, train_indices, val_indices, start_year, num_years, dataset_path, optimizer, scheduler, device=device)
    else:
        baseline_train(args, model, get_indices, train_indices, val_indices, start_year, num_years, dataset_path, optimizer, scheduler, device=device)
         
    # inference
    model.load_state_dict(torch.load(f'model_pkl/baseline_model_{args.split}_{args.model}_{args.hidden_dim}_{args.num_layers}_{args.tips}_{args.seed}.pt'))

    model.eval()
    with torch.no_grad():
        year_list = [num for num in range(start_year, start_year + num_years)]
        count = 0
        for year in tqdm(year_list):
            file_path = os.path.join(dataset_path, 'graph/' + str(year) + '.pt')
            data = torch.load(file_path)
            test_loader = NeighborLoader(data, num_neighbors=[0], batch_size=args.batch_size * 10)
            if args.model == 'Oxygenerator':
                test_loader = NeighborLoader(data, num_neighbors=[-1, -1], batch_size=args.batch_size)
            for i, batch in enumerate(test_loader):
                batch = batch.to(device)
                if args.model == 'Oxygenerator':
                    result, spatial_pred = model(batch)
                else:
                    result = model(batch.x_geo.float(), batch.x.float(), batch.time_series_profile.float(), batch.edge_index, batch.edge_attr, batch.x_geo.float())
                pred = result[:len(batch.input_id)]
                y = batch.y[:len(batch.input_id)].float()
                mask = ~torch.isnan(y)
                xyz = batch.x_geo[:,:3]
                latitude = torch.arcsin(xyz[:,-1])
                latitude = torch.rad2deg(latitude)
                longitude = torch.arctan2(xyz[:,1],xyz[:,0])
                longitude = torch.rad2deg(longitude)
                years = batch.x_geo[:,4] * num_years + start_year 
                geo_mask = batch.mask[:len(batch.input_id)] == 1
                geo_information = torch.stack((latitude[:len(batch.input_id)], longitude[:len(batch.input_id)], years[:len(batch.input_id)]), dim=1)
                pred = torch.where(geo_mask, pred, torch.nan)
                if count == 0:
                    all_geoinformation = geo_information.cpu().numpy()
                    all_pred = pred.cpu().numpy()
                else:
                    all_geoinformation = np.concatenate([all_geoinformation, geo_information.cpu().numpy()])
                    all_pred = np.concatenate([all_pred, pred.cpu().numpy()])
                count +=1

    depths = np.arange(1, 34)
    depths = np.tile(np.arange(1, 34), len(all_geoinformation))
    all_geoinformation =  np.repeat(all_geoinformation, 33, axis = 0)
    depths_column = pd.Series(depths, name='depth')
    oxygen_column = pd.Series(all_pred.flatten(), name='oxygen')
    df = pd.DataFrame(all_geoinformation, columns=['latitude', 'longitude', 'year'])
    df['depth'] = depths_column
    df['oxygen'] = oxygen_column
    df['latitude'] = df['latitude'].round(1)
    df['longitude'] = df['longitude'].round(1)
    df['year'] = df['year'].round(0)
    df['depth'] = df['depth'].round(2)
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
    np.save(f'infer_result/inference_by_{args.dataset}_{args.split}_' + args.model + f'_model_{args.seed}', result_np)


def get_args():
    parser = argparse.ArgumentParser(description='Oceanverse')
    parser.add_argument('--dataset', type=str, default='CESM2-omip1', help='omip1, omip2, GFDL')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of layers in the Transformer model')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for training')
    parser.add_argument('--max_patience', type=int, default=10, help='Maximum patience for early stopping')
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate for training')
    parser.add_argument('--num_epochs', type=int, default=1000, help='Number of epochs for training')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
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
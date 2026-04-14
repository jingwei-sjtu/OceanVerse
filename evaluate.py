import numpy as np
import torch
import pandas as pd
from tqdm import tqdm

## select the dataset
# dataset = 'CESM2_omip1'
# dataset = 'CESM2_omip2'
# dataset = 'GFDL_ESM4'
dataset = 'CESM2_omip1'
DATA_DIR = 'OceanVerse'  # Replace with your actual download directory


dataset_param = {
    'CESM2_omip1': {'num_years': 62, 'start_year': 1948, 'indices': 'omip1', 'graph': 'omip1', 'gt_file': 'OMIP1_do.npy'},
    'CESM2_omip2': {'num_years': 61, 'start_year': 1958, 'indices': 'omip2', 'graph': 'omip2', 'gt_file': 'OMIP2_do.npy'},
    'GFDL_ESM4':  {'num_years': 95, 'start_year': 1920, 'indices': 'GFDL',   'graph': 'GFDL',   'gt_file': 'GFDL_do.npy'},
}

num_years = dataset_param[dataset]['num_years']
start_year = dataset_param[dataset]['start_year']
year_list = list(range(start_year, start_year + num_years))

## load graph data (year-by-year .pt files)
pwd = f'{DATA_DIR}/{dataset}/{dataset_param[dataset]["graph"]}_graph/'
all_label = []
count = 0
for year in tqdm(year_list):
    file_path = pwd + str(year) + '.pt'
    data = torch.load(file_path)
    xyz = data.x_geo[:, :3]
    latitude = torch.rad2deg(torch.arcsin(xyz[:, -1]))
    longitude = torch.rad2deg(torch.arctan2(xyz[:, 1], xyz[:, 0]))
    label = data.y
    years = data.x_geo[:, 4] * num_years + start_year
    geo_information = torch.stack((latitude, longitude, years), dim=1)  # (42491, 3)
    if count == 0:
        all_geoinformation = geo_information.cpu().numpy()
        all_label = label.cpu().numpy()
    else:
        all_geoinformation = np.concatenate([all_geoinformation, geo_information.cpu().numpy()])
        all_label = np.concatenate([all_label, label.cpu().numpy()])
    count += 1

## expand to 3D grid (lat, lon, depth, year)
depths = np.arange(1, 34)
depths = np.tile(depths, len(all_geoinformation))
all_geoinformation = np.repeat(all_geoinformation, 33, axis=0)
df = pd.DataFrame(all_geoinformation, columns=['latitude', 'longitude', 'year'])
df['depth'] = pd.Series(depths, name='depth')
df['oxygen'] = pd.Series(all_label.flatten(), name='oxygen')
df['latitude'] = df['latitude'].round(1)
df['longitude'] = df['longitude'].round(1)
df['year'] = df['year'].round(0)
df['depth'] = df['depth'].round(2)

all_df = pd.read_csv(f'infer_result/all_df_empty_{dataset}.csv')
sampled_data = pd.merge(all_df, df, on=['year', 'depth', 'latitude', 'longitude'], how='left')
sampled_data = sampled_data['oxygen'].values
sampled_data = sampled_data.reshape((num_years, 33, 180, 360))
sampled_data = torch.tensor(sampled_data)

## load prediction result and ground truth
predict_result_path = 'infer_result/your_predict_result.npy'  # TODO: update to your actual file
ground_truth_path = f'{DATA_DIR}/{dataset}/{dataset_param[dataset]["indices"]}_ground_truth/{dataset_param[dataset]["gt_file"]}'

predict_result = np.load(predict_result_path)
ground_truth = np.load(ground_truth_path)

print(f"predict_result shape: {predict_result.shape}")
print(f"ground_truth shape: {ground_truth.shape}")

ground_truth = torch.from_numpy(ground_truth)
predict_result = torch.from_numpy(predict_result)

## longitude shift (if needed)
ground_truth = torch.cat((ground_truth[:, :, :, 180:], ground_truth[:, :, :, :180]), dim=3)

## select valid data
select_indices = torch.isnan(sampled_data) & (~torch.isnan(predict_result)) & (~torch.isnan(ground_truth))
predict_result_valid = predict_result[select_indices]
ground_truth_valid = ground_truth[select_indices]

delta = predict_result_valid - ground_truth_valid
mae = torch.mean(torch.abs(delta))
mse = torch.mean(delta ** 2)
rmse = torch.sqrt(mse)
r2 = 1 - (torch.sum(delta ** 2) / torch.sum((ground_truth_valid - torch.mean(ground_truth_valid)) ** 2))

mape_threshold = 5 / 1000
mape_indices = ground_truth_valid > mape_threshold
mape = torch.mean(torch.abs(delta[mape_indices]) / ground_truth_valid[mape_indices])

print(f'MAE:  {mae.item():.6f}')
print(f'MSE:  {mse.item():.6f}')
print(f'RMSE: {rmse.item():.6f}')
print(f'R2:   {r2.item():.6f}')
print(f'MAPE: {mape.item():.6f}')

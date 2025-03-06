import pandas as pd

from base_data_loader import BaseDataLoader
import numpy as np
from barra.functions import dfs_to_df


def remove_outliers(arr):
    median = np.nanmedian(arr, axis=1)
    absolute_deviation = np.abs(arr - median[:, np.newaxis])
    mad = np.nanmedian(absolute_deviation, axis=1)
    limit_up = median + 5 * mad
    limit_down = median - 5 * mad
    arr = np.clip(arr, limit_down[:, np.newaxis], limit_up[:, np.newaxis])
    return arr



if __name__ == '__main__':
    data = BaseDataLoader.load_data('../../数据/capital.parquet', fields=['circulating_cap'])
    ln_cap = np.log(data.data[:, 0, :])
    mid_cap = np.power(ln_cap, 3)
    l = len(ln_cap[0])
    for i in range(len(ln_cap)):
        y = mid_cap[i]
        index = np.where(~np.isnan(y))[0]
        x = ln_cap[i][index]
        x = np.c_[np.ones(len(x)), x]
        y = y[index]
        w = np.linalg.pinv(x.T @ x) @ x.T @ y
        mid_cap[i][index] = y - w @ x.T
    mid_cap = remove_outliers(mid_cap)
    mid_cap = (mid_cap - np.nanmean(mid_cap, axis=1)[:, np.newaxis]) / np.nanstd(mid_cap, axis=1)[:, np.newaxis]
    size = (ln_cap + mid_cap) / 2
    data.data = np.stack((ln_cap, mid_cap, size), axis=1)
    data.cols = {'ln_cap': 0, 'mid_cap': 1, 'size': 2}
    dfs = data.to_dataframes()
    df = dfs_to_df(dfs)
    df.to_parquet('../barra_factors.parquet')

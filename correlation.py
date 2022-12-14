import numpy as np
import pandas as pd
import scipy as sp
from scipy import stats
import pingouin as pg


def spearman_corr(df: pd.DataFrame):
    corr, p_values = stats.spearmanr(df.values)
    corr_df = pd.DataFrame(corr.copy(), columns=df.columns, index=df.columns)
    
    p_df = pd.DataFrame(p_values, index=df.columns, columns=df.columns)
    
    mixed = np.tril(corr).copy()
    mixed += np.triu(p_values)
    mixed_df = pd.DataFrame(mixed, index=df.columns, columns=df.columns)
    
    return corr_df, p_df, mixed_df


def partial_spearmanr(df: pd.DataFrame):
    columns = df.columns
    r_matrix = []
    p_matrix = []
    
    for x in columns:
        r_row = []
        p_row = []
        for y in columns:
            if x is not y:
                covar = list(filter(lambda i: i not in (x, y), columns))
                stats_df = pg.partial_corr(data=df, x=x, y=y, covar=covar, method="spearman")
                r_row.append(stats_df.at["spearman", "r"])
                p_row.append(stats_df.at["spearman", "p-val"])
            else:
                covar = None
                r_row.append(1)
                p_row.append(1)
        r_matrix.append(r_row)
        p_matrix.append(p_row)
        
    pcorr, p_values = np.array(r_matrix), np.array(p_matrix)
    pcorr_df = pd.DataFrame(pcorr.copy(), columns=columns, index=columns)
    
    p_df = pd.DataFrame(p_values, index=columns, columns=columns)
    
    np.fill_diagonal(pcorr, 0) # ゼロ割防止のために対角成分を0とする
    mixed = np.tril(pcorr).copy()
    mixed += np.triu(p_values)
    mixed_df = pd.DataFrame(mixed, index=columns, columns=columns)
    
    return pcorr_df, p_df, mixed_df


def pearson_corr(df: pd.DataFrame):
    data = df.values
    cov = np.cov(data.T)  
    D = np.diag(np.power(np.diag(cov), -0.5))
    corr = np.dot(np.dot(D, cov), D)
    
    corr_df = pd.DataFrame(corr.copy(), columns=df.columns, index=df.columns)
    
    n = df.shape[0]
    np.fill_diagonal(corr, 0) # ゼロ割防止のために対角成分を0とする
    t = np.abs(corr) * np.sqrt(n-2) / np.sqrt(1-corr*corr)
    p_values = 2 * sp.stats.t.cdf(-t, n-2)
    p_df = pd.DataFrame(p_values, index=df.columns, columns=df.columns)
    
    mixed = np.tril(corr).copy()
    mixed += np.triu(p_values)
    mixed_df = pd.DataFrame(mixed, index=df.columns, columns=df.columns)
    
    return corr_df, p_df, mixed_df


def partial_corr(df: pd.DataFrame):
    data = df.values
    cov = np.cov(data.T)
    #逆行列(精度行列)を求める
    try:
        omega=np.linalg.inv(cov)
        
        # 偏相関行列
        D = np.diag(np.power(np.diag(omega), -0.5))
        partialcorr = np.dot(np.dot(D, omega), D) * ((-1) * np.ones_like(D)) ** (np.eye(D.shape[0]) + 1)
        
        pcorr_df = pd.DataFrame(partialcorr.copy(), columns=df.columns, index=df.columns)
        # t-test
        n = df.shape[0]
        q = df.shape[1]-2
        np.fill_diagonal(partialcorr, 0) # ゼロ割防止のために対角成分を0とする
        t = np.abs(partialcorr) * np.sqrt(n-q-2) / np.sqrt(1-partialcorr*partialcorr)
        p_values = 2 * sp.stats.t.cdf(-t, n-q-2)
        p_df = pd.DataFrame(p_values, index=df.columns, columns=df.columns)
        
        mixed = np.tril(partialcorr).copy()
        mixed += np.triu(p_values)
        mixed_df = pd.DataFrame(mixed, index=df.columns, columns=df.columns)
    except Exception as e:
        print(e)
    
    return pcorr_df, p_df, mixed_df


def corr_ratio(df: pd.DataFrame, column: str):
    df_total = df[df.columns[df.columns != column]]
    arr_total = df_total.values
    s_total = np.sum(np.square(arr_total - arr_total.mean(axis=0)), axis=0)

    s_within = 0
    for i in df[column].unique():
        group = df_total.iloc[list(df[column] == i)].values
        s_group = np.sum(np.square(group - group.mean(axis=0)), axis=0)
        s_within += s_group
    
    cr = np.ones_like(s_total) - s_within / s_total
    cr_df = pd.DataFrame(cr.reshape(1, -1), columns=df_total.columns, index=[column])
    
    s_between = cr * s_total
    
    # f-test
    c = len(df[column].unique())
    n = len(df)
    f = (s_between/(c-1)) / (s_within/(n-c))
    p_values = sp.stats.f.sf(f, c-1, n-c)
    p_df = pd.DataFrame(p_values.reshape(1, -1), columns=df_total.columns, index=[column])    
        
    return cr_df, p_df

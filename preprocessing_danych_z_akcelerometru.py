import numpy as np
import pandas as pd
import argparse
import os
from sklearn.cluster import KMeans
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
from scipy import signal


# funkcja wczytująca wszystkie pliki .csv w danej lokacji
def get_data(f_dir, c_red):
    dfs = []
    for i, file in enumerate(os.listdir(f_dir)):
        if file.endswith('.csv'):
            try:
                if c_red == -1:
                    temp_df = pd.read_csv(os.path.join(f_dir, file), header=None)
                    dfs.append(temp_df)
                else:
                    temp_df = pd.read_csv(os.path.join(f_dir, file), header=None)
                    dfs.append(temp_df[c_red])
            except FileNotFoundError:
                pass
            except KeyError:
                print(f'File {file} had no column {c_red}')
                return pd.DataFrame()
    if len(dfs) > 0:
        df_raw = pd.concat(dfs, axis=1, ignore_index=True)
    else:
        return pd.DataFrame()
    return df_raw


# wybiera proces któremu będą poddane dane
def process_type(op, df, reduction):
    if op == 'mmm':
        df_processed = mmm(df, reduction)
    elif op == 'der':
        df_processed = der(df, reduction)
    elif op == 'absder':
        df_processed = absder(df, reduction)
    elif op == 'integ':
        df_processed = integ(df, reduction)
    elif op == 'km':
        df_processed = k_mean(df, reduction)
    elif op == 'pca':
        df_processed = pca(df, reduction)
    elif op == 'rpca':
        df_processed = rpca(df, reduction)
    elif op == 'cwt':
        df_processed = cwt(df, reduction)
    elif op == 'max':
        df_processed = max(df, reduction)
    elif op == 'rms':
        df_processed = rms(df, reduction)
    elif op == 'crest':
        df_processed = crest(df, reduction)
    elif op == 'rcwt':
        df_processed = rcwt(df, reduction)
    elif op == 'std_scale':
        df_processed = std_scale(df, reduction)
    elif op == 'nothing':
        df_processed = df
    elif op == 'fft':
        df_processed = fft(df, reduction)
    else:
        print(f'No operation {op} available, data left unprocessed')
        return pd.DataFrame()

    return df_processed


# max - min dzielone przed mean w danym oknie
def mmm(df, red):
    reduction_df = pd.concat([df, pd.DataFrame([x // red for x in df.index], columns=['cl'])], axis=1)
    temp = reduction_df.groupby('cl')
    return (temp.max() - temp.min())/temp.mean()


# "pochodna", opisuje pochylenie względem poprzedniego
def der(df, red):
    scores = []
    for col in df.columns:
        t_scores = []
        prev = 0
        for x in df[col]:
            t_scores.append(x - prev)
            prev = x
        scores.append(t_scores)

    scores = [pd.Series(ser_sc) for ser_sc in scores]
    t_df = pd.concat(scores, axis=1)
    reduction_df = pd.concat([t_df, pd.DataFrame([x // red for x in t_df.index], columns=['cl'])], axis=1)
    temp = reduction_df.groupby('cl').max()
    return temp[1:]


# "pochodna" ale w wartości bezwzględnej
def absder(df, red):
    scores = []
    for col in df.columns:
        t_scores = []
        prev = 0
        for x in df[col]:
            t_scores.append(abs(x - prev))
            prev = x
        scores.append(t_scores)

    scores = [pd.Series(ser_sc) for ser_sc in scores]
    t_df = pd.concat(scores, axis=1)
    reduction_df = pd.concat([t_df, pd.DataFrame([x // (red//2) for x in t_df.index], columns=['cl'])], axis=1)
    temp = reduction_df.groupby('cl').max()
    return temp[1:]


# "całka" w uproszczeniu liczy pole pod wykresem w danym zakresie
def integ(df, red):
    reduction_df = pd.concat([df, pd.DataFrame([x // (red//2) for x in df.index], columns=['cl'])], axis=1)
    temp = reduction_df.groupby('cl')
    return temp.min()-np.mean(temp.mean())+(1/2)*(temp.max()-temp.min())


# K Mean
def k_mean(df, red):
    fixed_red = df.shape[0] // red
    km = KMeans(n_clusters=fixed_red).fit(df)
    return pd.DataFrame(km.cluster_centers_)


# PCA
def pca(df, red):
    fixed_red = df.shape[0] // red
    if fixed_red > df.shape[1]:
        fixed_red = df.shape[1]
        print(f"PCA components amount reduced to {df.shape[1]} to keep process")

    pca_o = decomposition.PCA(n_components=fixed_red)
    pca_list = pca_o.fit_transform(df.transpose())
    return pd.DataFrame(data=pca_list).transpose()


# PCA z innym solverem i skalowaniem danych
def rpca(df, red):
    fixed_red = df.shape[0] // red
    #if fixed_red > df.shape[1]:
    #    fixed_red = df.shape[1]
    #    print(f"PCA components amount reduced to {df.shape[1]} to keep process")
    
    df = StandardScaler().fit_transform(df)
    
    pca_o = decomposition.PCA(n_components=fixed_red, svd_solver='randomized', whiten=False)
    pca_list = pca_o.fit_transform(df.transpose())
    return pd.DataFrame(pca_list).transpose()


# cwt, redukcja to w tym przypadku skala, do danych fft - widmo amplitudowe
def cwt(dfs, red):
    df = pd.DataFrame()
    for i, d in enumerate(dfs):
        if i%2 == 1:
            df = pd.concat([df, dfs[d]], axis=1)
            
    df_out = pd.DataFrame()
    for i, col in enumerate(df):
        temp = pd.DataFrame(signal.cwt(df[col], signal.ricker, [red])[0], columns=[i])
        df_out = pd.concat([df_out, temp], axis=1)
    return df_out


# cwt do danych gdzie uzyte są wszystkie kolumny    
def rcwt(dfs, red):
    df = dfs
            
    df_out = pd.DataFrame()
    for i, col in enumerate(df):
        temp = pd.DataFrame(signal.cwt(df[col], signal.ricker, [red])[0], columns=[i])
        df_out = pd.concat([df_out, temp], axis=1)
    return df_out
    
    
# max
def max(df, red):
    reduction_df = pd.concat([df, pd.DataFrame([x // red for x in df.index], columns=['cl'])], axis=1)
    temp = reduction_df.groupby('cl')
    return temp.max()
    
    
# funkcja rms
def fun_rms(x):
    x = np.array(x)
    return np.sqrt(np.divide(np.sum(np.multiply(x, x)), len(x)))


# rms
def rms(df, red):
    reduction_df = pd.concat([df, pd.DataFrame([x // red for x in df.index], columns=['cl'])], axis=1)
    temp = reduction_df.groupby('cl')
    return temp.aggregate(fun_rms)
    

# crest
def crest(df, red):
    reduction_df = pd.concat([df, pd.DataFrame([x // red for x in df.index], columns=['cl'])], axis=1)
    temp = reduction_df.groupby('cl')
    return np.abs(temp.max()-temp.min())/temp.aggregate(fun_rms)
    

# skalowanie danych dla uzyskania średniej 0 i wariacji 1
def std_scale(df, red):
    return pd.DataFrame(data=StandardScaler().fit_transform(df))
    

def fft(df, red):
    temp = pd.DataFrame()
    for col in df:
        temp[col] = np.abs(np.fft.fft(df[col]))
    
    return temp


# zapis pliku uwzględniający proces jakiemu zostały te dane poddane, zapisuje każdą kolumnę DataFrame osobno
def save_files(df, loc, process):
    if not os.path.isdir(loc):
        os.mkdir(loc)
    for i, col in enumerate(df.columns):
        try:
            df[col].to_csv(f'{loc}{process}_{i}.csv', header=False, index=False)
            print(f'Saving {process}_{i}.csv ended succesfully')
        except FileNotFoundError or PermissionError:
            print(f'Error while saving {process}_{i}.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocessing parameters.')
    parser.add_argument('-f', '--file_dir', type=str, default='.',
                        help='Direction of files if they are not in the same folder as script.')
    parser.add_argument('-l', '--location_dir', type=str, default='processed/',
                        help='Directory to save for processed files')
    parser.add_argument('-o', '--operation', type=str, default='mmm',
                        help='Type of operation, available: mmm (default); der; absder; integ; km; pca; rpca; cwt; max; rms; crest; std_scale; nothing')
    parser.add_argument('-n', '--nan_drop', type=bool, default=True, help='Remove rows with NaNs (default: True)')
    parser.add_argument('-c', '--condensation', type=int, default=20,
                        help='How many rows of data will be reduced to single or how many rows will you get at the end')
    parser.add_argument('-m', '--condensation_mode', type=int, default=0,
                        help='Condensation mode, available: '
                             '0 - output will have -c rows condensed into single; '
                             '1 - output will have -c amount of rows')
    parser.add_argument('-r', '--column_reduction', type=int, default=-1,
                        help='Uses only X columns from each .csv file (default: uses all)')
    parser.add_argument('-s', '--prescale', type=int, default=0,
                        help='Scales data using standard scaler before operation (default: 0 - False)')
    parser.add_argument('-p', '--postscale', type=int, default=0,
                        help='Scales data using standard scaler after operation (default: 0 - False)')

    args = parser.parse_args()

    file_fol = args.file_dir
    drop_nan = args.nan_drop
    operation = args.operation
    condensation = args.condensation
    save_loc = args.location_dir
    condensation_mode = args.condensation_mode
    col_reduction = args.column_reduction
    prescale = args.prescale
    postscale = args.postscale

    # ewentualne poprawki dla podanej lokacji
    if not file_fol[-1] == '/':
        file_fol += '/'
    if not save_loc[-1] == '/':
        save_loc += '/'
    '''
    Dane są wczytywane z danej lokacji, potem usuwane są rzędy z NaN jeśli taki był argument,
    dane poddawane są zadanemu procesowi i zapisywane
    '''
    folders = ['0p', '5p', '10p', '15p', '20p', '25p', '30p', '35p', '40p', '45p']
    
    for fol in folders:
        file_folder = f'{fol}\\{file_fol}'
        save_location = f'{fol}\\{save_loc}'
        condensation = args.condensation
        if os.path.exists(file_folder):
            unprocessed_df = get_data(file_folder,  col_reduction)

            if not unprocessed_df.empty:
                if drop_nan:
                    unprocessed_df.dropna(inplace=True)

                if condensation_mode == 1:
                    condensation = unprocessed_df.shape[0] // condensation
                
                if prescale != 0:
                    unprocessed_df = process_type('std_scale', unprocessed_df, condensation)
                
                preprocessed_df = process_type(operation, unprocessed_df, condensation)
                
                if postscale != 0:
                    preprocessed_df = process_type('std_scale', preprocessed_df, condensation)
                
                if not preprocessed_df.empty:
                    save_files(preprocessed_df, save_location, operation)
                else:
                    print("Got empty DataFrame after process")
            else:
                print("Could not get any data from given direction")

        else:
            print('The folder does not exist')

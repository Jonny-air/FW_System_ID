import os
import pandas as pd
import numpy as np

#Other Setup Parameters
_model_name = 'Believer_201112_true'
_verbose = True
_env = [9.8, 1.225]

def import_data(model_name, log_type, ramp_types = None, verbose = False, env = _env): #TODO add options to pass additinal arguments to only use part of the data
    csv_path = f"./csv/{model_name}_{log_type}.csv"
    if os.path.isfile(csv_path):
        # import existing csv to dataframe
        DF = pd.read_csv(csv_path)
    else:
        print(f"{csv_path} for this model and log type was not found")
        return 0
    if ramp_types is not None:
        SelectedDF = pd.DataFrame()
        for i in ramp_types:
           SelectedDF = SelectedDF.append(DF.query(f'ramp_types == {i}'), ignore_index=True)
        DF = SelectedDF

    vas = np.array([DF.loc[:,'vas'].values]).transpose()
    va_dots = np.array([DF.loc[:,'va_dots'].values]).transpose()
    fpas = np.array([DF.loc[:,'fpas'].values]).transpose()
    fpa_dots = np.array([DF.loc[:,'fpa_dots'].values]).transpose()
    alphas = np.array([DF.loc[:,'alphas'].values]).transpose()
    rolls = np.array([DF.loc[:,'rolls'].values]).transpose()
    pitches = np.array([DF.loc[:,'pitches'].values]).transpose()
    elevs = np.array([DF.loc[:,'elevs'].values]).transpose()
    if log_type ==1 or log_type ==2:
        n_ps = np.array([DF.loc[:,'n_ps'].values]).transpose()
        return vas, va_dots, fpas, fpa_dots, alphas, rolls, pitches, elevs, n_ps
    return vas, va_dots, fpas, fpa_dots, alphas, pitches, elevs, rolls

if __name__ == '__main__':
    import_data(_model_name, 0, verbose=_verbose, env=_env)
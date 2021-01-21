import os
import pandas as pd

_model_name = "Believer_201112_true" #<name>_<committdateYYMMDD>_<statelevel=true,false>
_log_name = 'log_1'
_log_type = 0            #0 = Lift/Drag, 1 = Throttle , 2 = Evaluation Data

def remove_log(model_name, log_name, log_type):
    csv_path = f"./csv/{model_name}_{log_type}.csv"

    oldDF = pd.read_csv(csv_path)
    if not oldDF.isin([f'{log_name}']).any().any():
        print("[ERROR] No log with this name to delete...")

    clearIndexes = oldDF[oldDF['log_names'] == log_name].index
    oldDF.drop(clearIndexes, inplace=True)
    oldDF.to_csv(csv_path)
    print("All done")

if __name__ == '__main__':
    remove_log(_model_name, _log_name, _log_type)
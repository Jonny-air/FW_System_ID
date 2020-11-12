import pickle

def get_params(model_name, verbose = False):
    try:
        with open(f'./model/{model_name}.txt', 'rb') as f:
            params = pickle.load(f)
    except:
        print("[ERROR] No parameters for this model, add model first with Add_Model.py")
        return 0
    return params
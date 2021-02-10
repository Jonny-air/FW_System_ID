import pickle

def get_params(model_name, verbose = False):
    try:
        with open(f'./model/{model_name}.txt', 'rb') as f:
            params = pickle.load(f)
    except:
        print("[ERROR] No parameters for this model, add model first with Add_Model.py")
        return 0
    return params

def get_wing_area(model_name, verbose=False):
    try:
        with open(f'./model/{model_name}_W.txt', 'rb') as f:
            w_A = pickle.load(f)
    except:
        print("[ERROR] No wing area for this model, add first with Add_Model.py")
        return 0
    return w_A
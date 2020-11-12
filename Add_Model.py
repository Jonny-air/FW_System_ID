import pickle
import os

#Model Parameters
model_name = "Believer_201112_true" #<name>_<committdateYYMMDD>_<statelevel=true,false>
m = 3.204           # mass in kg
n_0 = 2.3873        # prop speed when armed in rotps
n_slope = 156.7676  # prop speed slop in rotps
thrust_incl = 0.0   # thrust inclination angle if prop is inclined in rad
D_p = 0.28          # prop_diameter in m
actuator_control_ch = 3
state_level = True

#Other Setup Parameters
overwrite_existing = True  #only set to true if you are sure what you are doing

def add_model():
    #Do work:
    #check if model already exists
    if os.path.isfile(f"./model/{model_name}.txt"):
        print("Model already exists")
        if overwrite_existing:
            print("Overwriting...")
        else:
            return 0

    params = [m, n_0, n_slope, thrust_incl, D_p, actuator_control_ch, state_level]

    with open(f'./model/{model_name}.txt', 'wb') as f:
        pickle.dump(params, f)

if __name__ == '__main__':
    add_model()
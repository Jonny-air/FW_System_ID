import os

#parameters
_log_name = 'log_BT3'

def convert(log_name):
    messages = 'vehicle_attitude,sensor_accel,actuator_controls_0,vehicle_local_position'

    log_folder = './log'
    log_location = "%s/%s.ulg" %(log_folder,log_name)
    csv_location = "./csv"
    os.system('ulog2csv %s -o %s -m %s' %(log_location, csv_location, messages))

if __name__ == '__main__':
    convert(_log_name)

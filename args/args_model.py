



import argparse
import json
import os


parser = argparse.ArgumentParser()
args_temp = parser.parse_args()

this_file_abs_path = os.path.abspath(__file__)
package_path = '/'.join(this_file_abs_path.split('/')[:-1])
with open(os.path.join(package_path, 'model.json')) as f:
    trainer_params = json.load(f)

args_dict = vars(args_temp)


#print(args_dict)
for k,v in trainer_params.items():
    args_dict[k] = v

#print("*"*100)

args_model = argparse.Namespace(**args_dict)

#print(args)
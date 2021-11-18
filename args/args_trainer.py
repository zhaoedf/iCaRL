

import argparse
import json
import os

from pytorch_lightning import Trainer



parser = argparse.ArgumentParser()
parser = Trainer.add_argparse_args(parser)

args_temp = parser.parse_args()


this_file_abs_path = os.path.abspath(__file__)
package_path = '/'.join(this_file_abs_path.split('/')[:-1])
with open(os.path.join(package_path, 'trainer.json')) as f:
    trainer_params = json.load(f)

args_dict = vars(args_temp)


#print(args_dict)
for k,v in trainer_params.items():
    args_dict[k] = v

#print("*"*100)

args_trainer = argparse.Namespace(**args_dict)

# assert args_trainer.check_val_every_n_epoch == args_trainer.max_epochs

#print(args)


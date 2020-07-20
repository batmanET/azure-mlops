"""
Created on Mon Jun  8 14:20:33 2020

@author: Arawat4
"""

import argparse
import os
import tensorflow as tf
import fileinput
import shutil
import subprocess
import object_detection
import slim

from azureml.core import Dataset, Workspace, Model
from azureml.core.authentication import ServicePrincipalAuthentication

parser = argparse.ArgumentParser()
parser.add_argument('--dataset-path', dest='ds_path')
parser.add_argument('--epochs', dest='epochs')
parser.add_argument("--model_data", dest="model_data")

args = parser.parse_args()
data_path = args.ds_path
epochs = args.epochs

cred = {"SUBSCRIPTION_ID": "", # Subscription ID
        "RESOURCE_GROUP": "", # Resource group
        "WORKSPACE_NAME": "", # Workspace name
        "TENANT_ID": "", # Tenant ID
        "CLIENT_ID": "", # Client Secret
        }

def get_workspace(workspace, subscription, resource_group, tenantId, clientId, clientSecret):

    svcprincipal = ServicePrincipalAuthentication(tenant_id=tenantId,
                                                  service_principal_id=clientId,
                                                  service_principal_password=clientSecret)
    return Workspace.get(name=workspace, 
                         subscription_id=subscription, 
                         resource_group=resource_group,
                         auth=svcprincipal)

ws = get_workspace(workspace=cred["WORKSPACE_NAME"],
                   subscription=cred["SUBSCRIPTION_ID"],
                   resource_group=cred["RESOURCE_GROUP"],
                   tenantId=cred["TENANT_ID"],
                   clientId=cred["CLIENT_ID"],
                   clientSecret=cred["CLIENT_SECRET"])

tf_path = tf.__path__
TF_MODELS_DIR = object_detection.__path__[0]

print("="*40)
print(f"Tensorflow version - {tf.__version__}")
print("="*40)

try:
    paths = os.environ['PYTHONPATH'].split(":")
    if slim.__path__[0] not in paths:
        os.environ['PYTHONPATH'] = os.environ['PYTHONPATH'] + slim.__path__[0]
except:
    os.environ['PYTHONPATH'] = slim.__path__[0]

OUTPUT_DIR = args.model_data
LOGS_DIR = 'logs'
for directory in [OUTPUT_DIR, LOGS_DIR , "temp"]:
    os.makedirs(directory, exist_ok=True)

dataset = Dataset.get_by_name(ws, name='JIM10_tf_records')
dataset.download(target_path=(data_path), overwrite=False)

ssd_config_file = 'jim10_training_pipeline.config'
ssd_config_file = os.path.join(data_path, ssd_config_file)

for line in fileinput.input(ssd_config_file, inplace=True):
    print(line.replace("PATH_TO_BE_CONFIGURED", data_path))


pipeline_config_arg = f'--pipeline_config_path={ssd_config_file}'
epochs_arg = f'--num_train_steps={epochs}'
model_dir_arg = '--model_dir=temp'

#start training
process = subprocess.Popen(['python',
                            os.path.join(TF_MODELS_DIR, 'model_main.py'),
                            pipeline_config_arg,
                            model_dir_arg,
                            epochs_arg,
                            '--sample_1_of_n_eval_examples=1',
                            '--alsologtostderr'
                             ],
                           stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT
                        )

lines_iterator = iter(process.stdout.readline, b"")
with open("training_log.txt", "wb") as file:
    while process.poll() is None:
        for line in lines_iterator:
            print(line.decode("utf-8"), flush =True)
            file.write(line + b"\n")

process = subprocess.Popen(['python',
                            os.path.join(TF_MODELS_DIR, 'export_inference_graph.py'),
                            f'--pipeline_config_path={ssd_config_file}',
                            f'--trained_checkpoint_prefix=temp/model.ckpt-{epochs}',
                            '--output_directory=jim10_detection_graph'
                            ],
                           stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT
                          )

lines_iterator = iter(process.stdout.readline, b"")
with open("graph_export_log.txt", "wb") as file:
    while process.poll() is None:
        for line in lines_iterator:
            print(line.decode("utf-8"), flush =True)
            file.write(line + b"\n")


# Write event to logs directory
shutil.move("training_log.txt",  LOGS_DIR)
shutil.move("graph_export_log.txt",  LOGS_DIR)
shutil.move("jim10_detection_graph/frozen_inference_graph.pb", OUTPUT_DIR)
shutil.move("jim10_detection_graph/pipeline.config", OUTPUT_DIR)
shutil.move(os.path.join("temp", "eval_0"), LOGS_DIR)

shutil.rmtree("jim10_detection_graph")
shutil.rmtree("temp")
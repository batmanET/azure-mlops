# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 14:22:08 2020

@author: Arawat4
"""

from azureml.core.experiment import Experiment
from azureml.core import Workspace, Datastore
from azureml.core.runconfig import RunConfiguration 
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core.environment import Environment, CondaDependencies
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.pipeline.core import Pipeline, PipelineData, PipelineParameter
from azureml.pipeline.steps import PythonScriptStep, EstimatorStep
from azureml.train.estimator import Estimator

cred = {"SUBSCRIPTION_ID": "", # Subscription ID
        "RESOURCE_GROUP": "", # Resource group
        "WORKSPACE_NAME": "", # Workspace name
        "TENANT_ID": "", # Tenant ID
        "CLIENT_ID": "", # Client Secret
        }

EXP_NAME = 'JIM10_training'
TRAINING_CLUSTER_NAME = "training-cluster"
CONVERSION_CLUSTER_NAME = "ir-conversion"
REGISTERING_CLUSTER_NAME = "reg-artifacts"
TRAIN_DIR = "train"


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

experiment = Experiment(ws, EXP_NAME)  

def allocate_cluster(ws, CLUSTER_NAME, VM_TYPE, MAX_NODES):
    found = False
    cts = ws.compute_targets
    if CLUSTER_NAME in cts and cts[CLUSTER_NAME].type == 'AmlCompute':
        found = True
        print(f'Found existing compute target {CLUSTER_NAME}.')
        compute_target = cts[CLUSTER_NAME]

    if not found:
        print(f'Creating a new compute target {CLUSTER_NAME}...')
        provisioning_config = AmlCompute.provisioning_configuration(vm_size = VM_TYPE, max_nodes = MAX_NODES)

        # Create the cluster.\n",
        compute_target = ComputeTarget.create(ws, CLUSTER_NAME, provisioning_config)
    return compute_target

training_compute_target = allocate_cluster(ws, TRAINING_CLUSTER_NAME, "STANDARD_F8S_V2", 4)
conversion_compute_target = allocate_cluster(ws, CONVERSION_CLUSTER_NAME, "STANDARD_D1", 4)
register_compute_target = allocate_cluster(ws, REGISTERING_CLUSTER_NAME, "STANDARD_D1", 4)


conda_env = Environment("tf_object_detection")
conda_dep = CondaDependencies()
conda_dep.add_conda_package("tensorflow=1.15.0")
conda_dep.add_conda_package("tf_object_detection")
conda_dep.add_conda_package("numpy")
conda_dep.add_conda_package("tf-slim")
conda_dep.add_conda_package("pillow")
conda_dep.add_conda_package("lxml")
conda_dep.add_conda_package("scikit-learn")
conda_dep.add_conda_package("numpy=1.17.4")
conda_dep.add_conda_package("pycocotools")
conda_env.python.conda_dependencies=conda_dep


use_case_tag = PipelineParameter(name="use_case", default_value="NA")
controller_tag = PipelineParameter(name="controller", default_value="NA")
epochs = PipelineParameter(name="epochs", default_value=100)

datastore = Datastore(ws, "workspaceblobstore")
training_step_output = PipelineData("tf_training_output", datastore=datastore)
ir_model_output = PipelineData("jim10_detection", datastore=datastore)


data_prep_config = RunConfiguration(conda_dependencies=conda_dep)

ir_conversion_config = RunConfiguration()
ir_conversion_config.environment.docker.enabled = True
ir_conversion_config.environment.docker.base_image = None
ir_conversion_config.environment.docker.base_dockerfile = './Dockerfile'
ir_conversion_config.environment.python.user_managed_dependencies = True


est = Estimator(source_directory=TRAIN_DIR,
                entry_script='train.py',
                use_gpu=False,
                environment_definition=conda_env,
                compute_target=training_compute_target
               )

data_step = PythonScriptStep(name="Data Preparation Step",
                            script_name="makedata.py",
                            arguments=['--use_case', use_case_tag,
                                        '--controller', controller_tag
                                        ],
                            source_directory=None,
                            compute_target=training_compute_target,
                            runconfig = data_prep_config
                            )

train_step = EstimatorStep(name="Training Step",
                            estimator=est,
                            estimator_entry_script_arguments=['--dataset-path', ".",
                                                                '--epochs', epochs,
                                                                '--model_data', training_step_output
                                                                ],
                            outputs=[training_step_output],
                            compute_target=training_compute_target
                            )

ir_step = PythonScriptStep(name="TF Model To IR Conversion Step",
                            script_name="convert_to_ir.py",
                            arguments=['--tf_model_data', training_step_output, '--ir_model_data', ir_model_output],
                            source_directory=None,
                            inputs=[training_step_output],
                            outputs=[ir_model_output],
                            compute_target=conversion_compute_target, 
                            runconfig=ir_conversion_config
                            )

register_step = PythonScriptStep(name="Register OpenVINO IR",
                                script_name="register_model.py",
                                arguments=['--use_case', use_case_tag,
                                            '--controller', controller_tag,
                                            '--ir_model_data', ir_model_output],
                                source_directory=None,
                                inputs=[ir_model_output],
                                compute_target=register_compute_target
                                )

register_step.run_after(ir_step)
ir_step.run_after(train_step)
train_step.run_after(data_step)

pipeline = Pipeline(workspace=ws, steps=[register_step])
print ("Pipeline is built")

pipeline.validate()
print("Pipeline validation complete")

pipeline.publish(name="Rig_Floor_Safety_JIM10_train",
                description="Training JIM 10 detection model",
                version="1.0")
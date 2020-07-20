import argparse

from azureml.core import Workspace, Model
from azureml.core.authentication import ServicePrincipalAuthentication

parser = argparse.ArgumentParser()
parser.add_argument("--ir_model_data", dest="ir_model_data")
parser.add_argument('--use_case', dest='use_case')
parser.add_argument('--controller', dest='controller')

args = parser.parse_args()

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

Model.register(workspace=ws,
                model_name='jim10-detection',
                model_path=f'{args.ir_model_data}',
                tags={'use_case': args.use_case,
                        'controller': args.controller,
                        'type': 'object_detection'}
                )


# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 14:20:04 2020

@author: Arawat4
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import glob
import pandas as pd
import io
import argparse
from collections import namedtuple, OrderedDict
import xml.etree.ElementTree as ET
from random import shuffle
from PIL import Image

from sklearn.model_selection import train_test_split
import tensorflow as tf
from object_detection.utils import dataset_util

from azureml.core import Workspace, Dataset, Datastore
from azureml.core.authentication import ServicePrincipalAuthentication

parser = argparse.ArgumentParser()
parser.add_argument('--use_case', dest='use_case')
parser.add_argument('--controller', dest='controller')

args = parser.parse_args()
use_case = args.use_case
controller = args.controller

DATA_DIR = "data"

os.makedirs(os.path.join(DATA_DIR, "intermediate"), exist_ok=True)

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

dataset = Dataset.get_by_name(ws, name='JIM10')
dataset.download(target_path=(f"{DATA_DIR}"), overwrite=False)

def xml_to_csv():
    xml_list = {"test": [], "train": []}
    xmls = glob.glob(f"{DATA_DIR}/*.xml")
    train, test = train_test_split(xmls, train_size=0.8)
    for set_name, annotations in zip(["train", "test"], [train, test]):
        for xml_file in annotations:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for member in root.findall('object'):
                value = (root.find('filename').text,
                         int(root.find('size')[0].text),
                         int(root.find('size')[1].text),
                         member[0].text,
                         int(member[4][0].text),
                         int(member[4][1].text),
                         int(member[4][2].text),
                         int(member[4][3].text)
                         )
                xml_list[set_name].append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_test_df = pd.DataFrame(xml_list["test"], columns=column_name)
    xml_train_df = pd.DataFrame(xml_list["train"], columns=column_name)
    
    return xml_train_df, xml_test_df

df_train, df_test = xml_to_csv()



# TO-DO replace this with label map 
def class_text_to_int(row_label):
    if row_label == 'jim10':
        return 1
    else:
        None


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.io.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


for examples, label_set in zip([df_train, df_test], ['train', 'test']):
    output_path = os.path.join(os.getcwd(), f'{DATA_DIR}/intermediate/{label_set}.record')
    writer = tf.io.TFRecordWriter(output_path)
    path = os.path.join(os.getcwd(),  f'{DATA_DIR}')
    grouped = split(examples, 'filename')
    shuffle(grouped)
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))

datastore = Datastore(ws, "workspaceblobstore")
ds_reference  = datastore.upload(f'{DATA_DIR}/intermediate',
                 target_path='JIM10_tf_records',
                 overwrite=True,
                 show_progress=True)
jim10_ds = Dataset.File.from_files(path=[(datastore, "JIM10_tf_records")])
jim10_ds = jim10_ds.register(workspace=ws, name='JIM10_tf_records',
                                create_new_version = True,
                                tags={'use_case': use_case,
                                        'controller': controller,
                                        'type': 'object_detection'})

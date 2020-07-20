import os
import subprocess
import argparse
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("--tf_model_data", dest="tf_model_data")
parser.add_argument("--ir_model_data", dest="ir_model_data")

args = parser.parse_args()

OPTIMIZER_DIR = os.path.join(os.environ["INTEL_OPENVINO_DIR"], "deployment_tools/model_optimizer")
INPUT_DIR = args.tf_model_data
OUTPUT_DIR = args.ir_model_data
LOGS_DIR = 'logs'
for directory in [OUTPUT_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

process = subprocess.Popen(['python',
                            os.path.join(OPTIMIZER_DIR, 'mo_tf.py'),
                            f'--input_model={INPUT_DIR}/frozen_inference_graph.pb',
                            f'--transformations_config={OPTIMIZER_DIR}/extensions/front/tf/ssd_v2_support.json',
                            f'--tensorflow_object_detection_api_pipeline_config={INPUT_DIR}/pipeline.config'
                            ],
                           stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT
                          )

lines_iterator = iter(process.stdout.readline, b"")
while process.poll() is None:
    for line in lines_iterator:
        print(line.decode("utf-8"), flush =True)
            
shutil.move("./frozen_inference_graph.bin", OUTPUT_DIR)
shutil.move("./frozen_inference_graph.xml", OUTPUT_DIR)
shutil.move("./frozen_inference_graph.mapping", OUTPUT_DIR)

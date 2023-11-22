import json
import os
import sys
from pathlib import Path

print("sly globals")
print(1)
import supervisely as sly
from supervisely.app.v1.app_service import AppService

print(2)
import dotenv
import ast

print(3)

import torch.cuda

print(4)
dotenv.load_dotenv("./debug.env")
dotenv.load_dotenv("./secret_debug.env")

logger = sly.logger

my_app: AppService = AppService()
api = my_app.public_api

team_id = int(os.environ["context.teamId"])
workspace_id = int(os.environ["context.workspaceId"])

model = None

task_id = my_app.task_id
print(5)
device = (
    os.environ["modal.state.device"]
    if "cuda" in os.environ["modal.state.device"] and torch.cuda.is_available()
    else "cpu"
)

selected_weights_type = str(os.environ["modal.state.modelWeightsOptions"])
pretrained_models_table = ast.literal_eval(os.environ["modal.state.models"])
print(6)

# pretrained_models_table = list(json.loads(str(os.environ['modal.state.models'])))  # debug

if selected_weights_type == "pretrained":
    print(7)
    selected_model = os.environ["modal.state.selectedModel"]
    print(8)
    model_info = None
    for row in pretrained_models_table:
        print(9)
        logger.info(row)
        print(10)
        if row["Model"] == selected_model:
            model_info = row
            break
    remote_weights_path = model_info["weightsUrl"]
else:
    remote_weights_path = os.environ["modal.state.weightsPath"]
print(11)
local_dataset_path = os.path.join(my_app.data_dir, "sly_dataset")
local_weights_path = None

batch_size = int(os.environ["modal.state.batchSize"])
print(12)

entry_point_path = Path(sys.argv[0])
root_source_dir = str(entry_point_path.parents[3])

print(root_source_dir)
print(13)
sys.path.append(os.path.join(root_source_dir, "src"))
print(14)

# DEBUG
# sly.fs.clean_dir(my_app.data_dir, ignore_errors=True)

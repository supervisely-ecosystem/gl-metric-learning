import json
import os
import sys
from pathlib import Path

import supervisely as sly
import dotenv
import ast

import torch.cuda

dotenv.load_dotenv("supervisely/serve/debug.env")
dotenv.load_dotenv("supervisely/serve/secret_debug.env")
dotenv.load_dotenv(os.path.expanduser("~/supervisely.env"))

logger = sly.logger

my_app = sly.Application()
my_server = my_app.get_server()
api = sly.Api()

team_id = int(os.environ["context.teamId"])
workspace_id = int(os.environ["context.workspaceId"])
app_data_dir = "/app/data"

model = None

task_id = sly.env.task_id()

device = (
    os.environ["modal.state.device"]
    if "cuda" in os.environ["modal.state.device"] and torch.cuda.is_available()
    else "cpu"
)

selected_weights_type = str(os.environ["modal.state.modelWeightsOptions"])
pretrained_models_table = ast.literal_eval(os.environ["modal.state.models"])

# pretrained_models_table = list(json.loads(str(os.environ['modal.state.models'])))  # debug

if selected_weights_type == "pretrained":
    selected_model = os.environ["modal.state.selectedModel"]
    model_info = None
    for row in pretrained_models_table:
        logger.info(str(row))
        if row["Model"] == selected_model:
            model_info = row
            break
    remote_weights_path = model_info["weightsUrl"]
else:
    remote_weights_path = os.environ["modal.state.weightsPath"]

local_dataset_path = os.path.join(app_data_dir, "sly_dataset")
local_weights_path = None

batch_size = int(os.environ["modal.state.batchSize"])

root_source_dir = str(Path(__file__).parents[3])

print(root_source_dir)

sys.path.append(os.path.join(root_source_dir, "src"))

# DEBUG
# sly.fs.clean_dir(my_app.data_dir, ignore_errors=True)

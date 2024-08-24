import os
import sys
from pathlib import Path

import supervisely as sly
# import dotenv
#
# dotenv.load_dotenv('./debug.env')
# dotenv.load_dotenv('./secret_debug.env')


logger = sly.logger

my_app = sly.Application()
api = sly.Api()
app_data_dir = "/app/data"


session_id = os.environ["modal.state.sessionId"]


task_id = sly.env.task_id()
team_id = int(os.environ["context.teamId"])
workspace_id = int(os.environ["context.workspaceId"])
project_id = int(os.environ["modal.state.slyProjectId"])

workspace_info = api.workspace.get_info_by_id(workspace_id)
project_info = api.project.get_info_by_id(project_id)

project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))

remote_embeddings_dir = '/GL-MetricLearning/embeddings/'

local_project_path = os.path.join(app_data_dir, 'project')

os.makedirs(local_project_path, exist_ok=True)
sly.fs.clean_dir(local_project_path)

# batch_size = 256
batch_size = 5
model_info = None

root_source_dir = str(Path(__file__).parents[3])

sys.path.append(os.path.join(root_source_dir, 'src'))

# DEBUG
# sly.fs.clean_dir(my_app.data_dir, ignore_errors=True)

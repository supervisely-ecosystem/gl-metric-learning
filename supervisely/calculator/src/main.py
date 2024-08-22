import supervisely as sly

import json

import sly_globals as g
import functions as f


@g.my_app.callback("calculate_embeddings_for_project")
@sly.timeit
def calculate_embeddings_for_project(api: sly.Api, task_id, context, state, app_logger):
    datasets_list = g.api.dataset.get_list(g.project_id)
    images_count = sum([dataset.images_count for dataset in datasets_list if dataset.images_count])
    progress = sly.Progress("processing images:", images_count)
    for current_dataset in datasets_list:
        sly.logger.debug("Processing dataset: %s", current_dataset.name)
        packed_data = {}
        i = 0
        for image_infos_batch in api.image.get_list_generator(current_dataset.id, batch_size=128):
            images_ids = [image_info.id for image_info in image_infos_batch]
            images_urls = [image_info.full_storage_url for image_info in image_infos_batch]

            sly.logger.debug(f"Downloading annotations for {i} - {i+len(images_ids)} images...")
            ann_infos = api.annotation.download_batch(current_dataset.id, images_ids)
            ann_objects = f.jsons_to_annotations(ann_infos)

            data_for_each_image = f.get_data_for_each_image(ann_objects)
            batch_for_inference = f.generate_batch_for_inference(images_urls, data_for_each_image)
            sly.logger.debug(f"Infering batch of {i} - {i+len(images_ids)} images...")
            embeddings_by_indexes = f.inference_batch(batch_for_inference)

            sly.logger.debug("Inference done. Packing data...")
            f.pack_data(packed_data, batch_for_inference, embeddings_by_indexes)

            progress.iters_done_report(len(images_ids))
            i += len(images_ids)

        sly.logger.debug("Saving packed data for dataset: %s", current_dataset.name)
        f.write_packed_data_persistent(current_dataset.id, packed_data)

    g.my_app.stop()


def main():
    sly.logger.info("Script arguments", extra={
        "context.teamId": g.team_id,
        "context.workspaceId": g.workspace_id,
        "context.sessionId": g.session_id
    })

    f.check_model_connection()
    g.my_app.run(initial_events=[{"command": "calculate_embeddings_for_project"}])


if __name__ == "__main__":
    sly.main_wrapper("main", main)

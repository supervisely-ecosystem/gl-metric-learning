import os
import urllib.request
import cv2
import requests
import numpy as np
from functools import lru_cache

import model_functions

import sly_globals as g


def download_file_by_url(url, local_path):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return 0


def download_model_and_config():
    if g.selected_weights_type == 'pretrained':
        g.local_weights_path = os.path.join(g.app_data_dir, f'{g.selected_model}.pth')
        if not os.path.exists(g.local_weights_path):
            download_file_by_url(g.remote_weights_path, g.local_weights_path)

    else:
        remote_model_weights_name = g.remote_weights_path.split('/')[-1]
        g.local_weights_path = os.path.join(g.app_data_dir, remote_model_weights_name)
        g.api.file.download(g.team_id, g.remote_weights_path, g.local_weights_path)


def inference_one_batch(data):
    return model_functions.calculate_embeddings_for_nps_batch(data)


def batch_inference(data):
    """
    :param data: np.arrays: [[img, img, img, img]]
    :return: embedding for every image [[emb1, emb2, emb3, emb4]]
    """
    splits_num = int(len(data) / g.batch_size) if int(len(data) / g.batch_size) > 0 else 1
    batches = np.array_split(data, splits_num, axis=0)
    batches = [batch for batch in batches if batch.size > 0]

    embeddings = []
    for current_batch in batches:
        temp_embedding = inference_one_batch(current_batch)
        embeddings.extend(temp_embedding)
    return embeddings


def crop_images(data):
    """
     FOR EACH image in data
     crop image if bbox is not None
    """
    for row in data:
        if row['cached_image'] is not None and row['bbox']:
            top, left, height, width = row['bbox'][0], row['bbox'][1], row['bbox'][2], row['bbox'][3]
            crop = row['cached_image'][top:top + height, left:left + width]

            if crop.shape[0] > 0 and crop.shape[1] > 0:
                row['cached_image'] = crop
            else:
                g.logger.warn(f'image not cropped: {row["url"]}\n'
                              f'reason: {crop.shape}')
                row['cached_image'] = None


@lru_cache(maxsize=32)
def url_to_image(url):
    with urllib.request.urlopen(url) as resp:
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        # return the image
        return image


def cache_images(data):
    """
     FOR EACH url in data
     download image to RAM by url
    """
    for row in data:
        try:
            image_in_memory = url_to_image(row['url'])
            row['cached_image'] = image_in_memory
        except Exception as ex:
            g.logger.warn(f'image not downloaded: {row["url"]}\n'
                          f'reason: {ex}')

            row['cached_image'] = None

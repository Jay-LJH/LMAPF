import os
import glob
import cv2
import imageio
import moviepy.editor as mp
import json

import numpy as np

from PIL import Image
from matplotlib import colors

kiva_obj_types = ".@ersw"
# 's' is dummy type
kiva_color_list = ['white', 'black', 'deepskyblue', 'orange', '', 'fuchsia']


def read_in_kiva_map(map_filepath):
    """
    Read in kiva map and return in str format
    """
    with open(map_filepath, "r") as f:
        raw_env_json = json.load(f)
        raw_env = raw_env_json["layout"]
        name = raw_env_json["name"]
    return raw_env, name


def kiva_env_str2number(env_str):
    """
    Convert kiva env in string format to np int array format.

    Args:
        env_str (List[str]): kiva env in string format

    Returns:
        env_np (np.ndarray)
    """
    env_np = []
    for row_str in env_str:
        # print(row_str)
        row_np = [kiva_obj_types.index(tile) for tile in row_str]
        env_np.append(row_np)
    return np.array(env_np, dtype=int)


def kiva_tile_to_color(tile_id):
    color_name = kiva_color_list[tile_id]
    rgb = colors.to_rgb(color_name)
    color_rgb_255 = tuple(int(val * 255) for val in rgb)
    return color_rgb_255


def convert_avi_to_gif(
        input_path,
        output_path,
        output_resolution=(640, 640),
):
    # Load the input video file
    clip = mp.VideoFileClip(input_path)

    # Convert the video to a sequence of frames
    frames = []
    for frame in clip.iter_frames():
        # Resize the frame to the desired output resolution
        frame = Image.fromarray(frame).resize(output_resolution)
        frames.append(frame)

    # Write the frames to a GIF file
    # imageio.mimsave(output_path, frames, fps=clip.fps, size=output_resolution)
    imageio.mimsave(
        output_path,
        frames,
        duration=1000 * 1/clip.fps,
        format='gif',
        palettesize=256,
    )


def create_movie(folder_path, filename):
    glob_str = os.path.join(folder_path, '*.png')
    image_files = sorted(glob.glob(glob_str))

    # Grab the dimensions of the image
    img = cv2.imread(image_files[0])
    image_dims = img.shape[:2][::-1]

    # Create a video
    avi_output_path = f"{filename}.avi"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    frame_rate = 30
    video = cv2.VideoWriter(
        avi_output_path,
        fourcc,
        frame_rate,
        image_dims,
    )

    for img_filename in image_files:
        img = cv2.imread(img_filename)
        video.write(img)

    video.release()

    # Convert video to gif
    gif_output_path = f"{filename}.gif"
    convert_avi_to_gif(avi_output_path, gif_output_path, image_dims)

def run_once(f):
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            return f(*args, **kwargs)
    wrapper.has_run = False
    return wrapper

@run_once
def print_once(*args):
    print(args)
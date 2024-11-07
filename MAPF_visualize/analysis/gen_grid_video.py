import os
import fire

from moviepy.editor import VideoFileClip, clips_array


def gen_grid_video(logdirs, margin_size=10):

    video_files = [
        "logs/solution.mp4",
        "logs/solution__1__AdobeExpress.mp4",
    ]
    videos = [
        VideoFileClip(video).resize(newsize=(720, 720)).margin(margin_size, color=(231, 233, 235))
        for video in video_files
    ]

    # video_files = []
    # for logdir in os.listdir(logdirs):
    #     if os.path.isdir(os.path.join(logdirs, logdir)):
    #         video_files.append(os.path.join(logdirs, logdir, "capture.avi"))

    # # Load the videos and resize them to have the same dimensions
    # videos = [
    #     VideoFileClip(video).margin(margin_size, color=(231, 233, 235))
    #     for video in video_files
    # ]

    # Assuming you want a 3x3 grid
    rows = 1
    cols = 2

    # Arrange the clips into an array
    video_grid = [[videos[i * cols + j] for j in range(cols)]
                  for i in range(rows)]

    # Concatenate the videos into a grid
    final_clip = clips_array(video_grid)

    # Write the result to a file
    final_clip.write_videofile(os.path.join(logdirs, "output_video.mp4"),
                               fps=24)

if __name__ == "__main__":
    fire.Fire(gen_grid_video)

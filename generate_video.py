from cv2 import VideoWriter, VideoWriter_fourcc, imread, destroyAllWindows
import os
from datetime import date

today = date.today()

image_folder = 'fragment'
video_name = "exported-%s.mp4" % today.strftime("%Y%m%d%H%M")
FPS = 10

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
frame = imread(os.path.join(image_folder, images[0]))
h, w, layers = frame.shape

video = VideoWriter(video_name, VideoWriter_fourcc(*'MP4V'), FPS, (1920, 1080))

for image in images:
    video.write(imread(os.path.join(image_folder, image)))

destroyAllWindows()
video.release()

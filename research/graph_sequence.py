import os
from PIL import Image
from PIL import ImageDraw
import imageio

images = []
total_checkpoints = int((len(os.listdir("timelapse")) / 2))

for i in range(0, total_checkpoints * 1000, 1000):
    print(i)
    img = Image.open(os.path.join("timelapse", f"checkpoint_{i}.png"))
    draw = ImageDraw.Draw(img)
    draw.text((220, 20), str(i), (0, 0, 0))
    images.append(img)

imageio.mimsave("movie.gif", images)

import glob

from PIL import Image

files = sorted(glob.glob("./*.png"))
# images = list(map(lambda file: Image.open(file), files))
images = [Image.open(file) for file in files]
images[0].save("image.gif", save_all=True, append_images=images[1:], duration=200)

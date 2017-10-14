def image_classifier():
    import os, os.path
    import numpy as np
    from PIL import Image

    path = "/home/apurvnit/Projects/cat-or-not/data"
    valid_images = [".jpg", ".gif", ".png", ".tga"]
    imgs = []
    Y = []
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        img = Image.open(os.path.join(path, f))
        img = np.array(img.resize((64, 64), Image.ANTIALIAS))
        imgs.append(img)
        Y.append(f[:f.find('.')] == 'cat')

    return np.array(imgs), np.array(Y)





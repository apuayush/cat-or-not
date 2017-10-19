
def image_classifier(path):
    import os, os.path
    import numpy as np
    from PIL import Image

    valid_images = [".jpg", ".gif", ".png", ".tga"]
    imgs = []
    c = 0
    Y = []
    for f in os.listdir(path):
        try:
            ext = os.path.splitext(f)[1]
            if ext.lower() not in valid_images:
                continue
            img = Image.open(os.path.join(path, f))
            img = np.array(img.resize((64, 64), Image.ANTIALIAS))

            # img = img.reshape(64, 64, 3)
            imgs.append(img.reshape(12288)/255)
            Y.append(f[:f.find('.')] == 'cat')

        except: pass

    # print(np.array(imgs).shape, np.array(Y).shape)
    return np.array(imgs).T, np.array(Y)


def rename(dir, pattern, titlePattern):
    import glob, os
    for pathAndFilename in glob.iglob(os.path.join(dir, pattern)):
        title, ext = os.path.splitext(os.path.basename(pathAndFilename))
        os.rename(pathAndFilename, 
                  os.path.join(dir, titlePattern % title + ext))

# rename(r'/home/apurvnit/Projects/cat-or-not data/catmapper',r'*.jpg',r'cat.(%s)')


import numpy as np

def mask2rle(img, width, height):
    rle = []
    lastColor = 0;
    currentPixel = 0;
    runStart = -1;
    runLength = 0;

    for x in range(width):
        for y in range(height):
            currentColor = img[x][y]
            if currentColor != lastColor:
                if currentColor == 255:
                    runStart = currentPixel;
                    runLength = 1;
                else:
                    rle.append(str(runStart));
                    rle.append(str(runLength));
                    runStart = -1;
                    runLength = 0;
                    currentPixel = 0;
            elif runStart > -1:
                runLength += 1
            lastColor = currentColor;
            currentPixel+=1;

    return " ".join(rle)

def rle2mask(rle, width, height):
    mask= np.zeros(width* height)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position:current_position+lengths[index]] = 255
        current_position += lengths[index]

    return mask.reshape(width, height)

def load_mask(self, height, width, annotations):
    count = len(annotations)
    if count == 0 or (count == 1 and annotations[0] == -1): # empty annotation
        mask = np.zeros((height, weight, 1), dtype=np.uint8)
        class_ids = np.zeros((1,), dtype=np.int32)
    else:
        mask = np.zeros((height, width, count), dtype=np.uint8)
        class_ids = np.zeros((1,), dtype=np.int32)
        mask[:, :, i] = rle2mask(annotations, height, width).T
    return mask.astype(np.bool), class_ids.astype(np.int32)

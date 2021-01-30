import cv2
import imutils
import numpy as np

MODEL_PATH = r"models/enet-model.net"
NET_INPUT = (1024, 512)

RED = [255, 0, 0]
ORANGE = [255, 165, 0]
YELLOW = [255, 255, 0]
GREEN = [0, 255, 0]
BLUE = [0, 0, 255]
INDIGO = [75, 0, 130]
VIOLET = [238, 130, 238]
BLACK = [0, 0, 0]
GRAY = [127, 127, 127]
WHITE = [255, 255, 255]
CYAN = [0, 255, 255]
PURPLE = [153, 50, 204]
PINK = [255, 51, 255]
DARK_RED = [204, 0, 0]

COLOR_MAP = {
    "Unlabeled": BLACK,
    "Road": GREEN,
    "Sidewalk": BLACK,
    "Building": BLACK,
    "Wall": BLACK,
    "Fence": BLACK,
    "Pole": PURPLE,
    "TrafficLight": PURPLE,
    "TrafficSign": PURPLE,
    "Vegetation": BLACK,
    "Terrain": BLACK,
    "Sky": BLACK,
    "Person": RED,
    "Rider": PINK,
    "Car": YELLOW,
    "Truck": BLACK,
    "Bus": BLACK,
    "Train": BLACK,
    "Motorcycle": PINK,
    "Bicycle": PINK
}


def _do_color_change(colors):
    colors = [c[::-1] for c in colors]
    colors = np.array(colors, dtype="uint8")
    return colors


COLORS = _do_color_change(list(COLOR_MAP.values()))


def _do_segmentation(image, net):
    global COLORS, NET_INPUT

    # Logic here
    image = imutils.resize(image, width=500)

    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, NET_INPUT, 0,
                                 swapRB=True, crop=False)

    net.setInput(blob)

    output = net.forward()

    print(output.shape)

    print(output[0].shape)

    class_map = np.argmax(output[0], axis=0)

    print(class_map.shape)

    mask = COLORS[class_map]

    mask = cv2.resize(mask, (image.shape[1], image.shape[0]),
                      interpolation=cv2.INTER_NEAREST)

    segemented = ((image * 0.7) + (mask * 0.3)).astype("uint8")

    cv2.imshow("original", image)
    cv2.imshow("mask", mask)
    cv2.imshow("segmented", segemented)
    cv2.im
    cv2.waitKey()

    return image


if __name__ == "__main__":
    image_path = r"images/example_02.jpg"

    image = cv2.imread(image_path)

    net = cv2.dnn.readNet(MODEL_PATH)

    segmented_image = _do_segmentation(image, net)

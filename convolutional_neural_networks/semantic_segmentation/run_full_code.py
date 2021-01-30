import cv2
import imutils
import numpy as np

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

MODEL_PATH = r"models/enet-model.net"
ENET_INPUT = (1024, 512) # ENet trained using 1024x512

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


def _get_colors(color_values):
    color_rgb = [c[::-1] for c in color_values]
    color_rgb = np.array(color_rgb, dtype="uint8")
    return color_rgb


COLORS = _get_colors(list(COLOR_MAP.values()))


def do_image_segmentation(net, image, image_w=500, show=False):
    global COLORS, ENET_INPUT

    image = imutils.resize(image, width=image_w)

    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, ENET_INPUT, 0,
                                 swapRB=True, crop=False)

    net.setInput(blob)
    output = net.forward()

    # our output class ID map will be num_classes x height x width in
    # size, so we take the argmax to find the class label with the
    # largest probability for each and every (x, y)-coordinate in the
    # image
    classMap = np.argmax(output[0], axis=0)

    # given the class ID map, we can map each of the class IDs to its
    # corresponding color
    mask = COLORS[classMap]

    # Resize mask to the original image size so we can overlap them
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]),
                      interpolation=cv2.INTER_NEAREST)

    output = ((0.7 * image) + (0.3 * mask)).astype("uint8")

    if show:
        cv2.imshow("Input", image)
        cv2.imshow("Output", output)
        cv2.waitKey(0)
        cv2.imwrite("output/image_0.png", output)

    return output


def do_video_segmentation(net, video_path, out_path, image_w=500, show=False):
    vs = cv2.VideoCapture(video_path)
    writer = None

    try:
        prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
            else cv2.CAP_PROP_FRAME_COUNT
        total = int(vs.get(prop))
    except Exception as e:
        total = -1

    processed_frames = 1

    while True:
        (grabbed, frame) = vs.read()

        if frame is None:
            continue

        if processed_frames > 300:
            break

        output = do_image_segmentation(net, frame, image_w=image_w, show=show)

        print(f"Processing {processed_frames}/{total}")
        processed_frames += 1

        # check if the video writer is None
        if writer is None:
            # initialize our video writer
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(out_path, fourcc, 30,
                                     (output.shape[1], output.shape[0]), True)

        writer.write(output)

    writer.release()
    vs.release()


if __name__ == "__main__":

    is_video = False

    net = cv2.dnn.readNet(MODEL_PATH)

    if is_video:
        video_name = "toronto"

        video_path = f"videos/{video_name}.mp4"
        out_path = f"output/{video_name}-{video_name}.avi"

        segmented_video = do_video_segmentation(net, video_path, out_path, image_w=512, show=False)
    else:
        image_path = r"images/example_01.png"
        image = cv2.imread(image_path)
        segmented_image = do_image_segmentation(net, image, image_w=512, show=True)

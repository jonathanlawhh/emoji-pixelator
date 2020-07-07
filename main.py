from keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2
import numpy as np
import pyautogui
from time import sleep
import helpers
import winsound

WIN_SIZE = 200
STEP_SIZE = WIN_SIZE - 50
CAPTURE_FRAMES = 120

print("Loading model...")
MODEL = load_model('emoji.h5')
print("Loaded")


# Pixelation function. Resize down and up
def pixelate(input_img):
    w, h = input_img.shape[0], input_img.shape[1]

    t = cv2.resize(input_img, (12, 12), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(t, (w, h), interpolation=cv2.INTER_NEAREST)


# Send to model for prediction
def emoji_AI(input_img):
    img = np.expand_dims(img_to_array(input_img), axis=0)
    result = MODEL.predict(img)
    return result[0][0]


# Optimization function for input image
def image_optimization(input_img):
    cv2.medianBlur(input_img, 1, input_img)
    return input_img


# Sliding window function
def slide(input_img, output):
    pixelate_count = 0

    for resized in helpers.pyramid(input_img, scale=2):
        # loop over the sliding window for each layer of the pyramid
        for (x, y, window) in helpers.sliding_window(resized, stepSize=STEP_SIZE, windowSize=(WIN_SIZE, WIN_SIZE)):
            # if the window does not meet our desired window size, ignore it
            if window.shape[0] != WIN_SIZE or window.shape[1] != WIN_SIZE:
                continue

            # This code below shows sliding window
            # clone = resized.copy()
            # cv2.rectangle(clone, (x, y), (x + WIN_SIZE, y + WIN_SIZE), (0, 255, 0), 2)

            # Ensure crop is on the original location after pyramid resize
            temp_y_start = y + (input_img.shape[1] - resized.shape[1])
            temp_x_start = x + (input_img.shape[0] - resized.shape[0])

            temp_y_end = temp_y_start + WIN_SIZE
            temp_x_end = temp_x_start + WIN_SIZE

            try:
                crop_img = cv2.resize(resized[temp_y_start:temp_y_end, temp_x_start:temp_x_end], (200, 200))
            except Exception as e:
                continue

            # Emoji confidence is below 0.9, above is nonsense
            if emoji_AI(crop_img) < 0.9:
                # If verified an emoji, pixelate
                pixelate_count += 1
                output[temp_y_start:temp_y_end, temp_x_start:temp_x_end] = \
                    pixelate(output[temp_y_start:temp_y_end, temp_x_start:temp_x_end])

    print("Number of pixelation:", pixelate_count)
    return output


def detect(input_img, i):
    print("Processing", i)
    input_img = cv2.cvtColor(np.array(input_img), cv2.COLOR_RGB2BGR)
    output = np.array(input_img)

    # Passing to image optimizer then slide window
    output = slide(image_optimization(input_img), output)
    cv2.imwrite("./output/%s.png" % format(i, "04d"), output)


if __name__ == '__main__':
    print(" This system captures screenshots from your screen.")
    print(" Image capture in 2 seconds...")
    sleep(2)
    winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS)

    screenshots = []

    # Below steps can be changed to read a video file instead
    for i in range(CAPTURE_FRAMES):
        image = pyautogui.screenshot()
        screenshots.append(image)
        sleep(.1)

    winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS)

    # Pass for detection
    for i, s in enumerate(screenshots):
        detect(s, i)

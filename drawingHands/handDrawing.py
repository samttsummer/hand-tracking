import cv2
import mediapipe as mp
import numpy as np
import time                     # used to calculate the frame rate

from typing import Union        # union type: union[X, Y]


class AsimovDetector():
    def __init__(
        self,
        mode: bool = False,
        number_hands: int = 2,
        model_complexity: int = 1,
        min_detec_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ):
        # parameters required to initialize Hands, MediaPipe's hand detection solution
        # https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
        self.mode = mode
        self.max_num_hands = number_hands
        self.complexity = model_complexity
        self.detection_con = min_detec_confidence
        self.tracking_con = min_tracking_confidence

        # initialize the hands solution
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            self.mode,
            self.max_num_hands,
            self.complexity,
            self.detection_con,
            self.tracking_con
        )

        self.mp_draw = mp.solutions.drawing_utils

    def find_hands(
        self,
        img: np.ndarray,
        draw_hands: bool = True
    ):
        # convert the image from BGR to RGB
        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # process the image and collect the hand detection results
        self.results = self.hands.process(img_RGB)

        if self.results.multi_hand_landmarks:
            for hand in self.results.multi_hand_landmarks:
                if draw_hands:
                    self.mp_draw.draw_landmarks(
                        img,
                        hand,
                        self.mp_hands.HAND_CONNECTIONS
                    )

        return img

    def find_position(
        self,
        img: np.ndarray,
        hand_number: int = 0,
        draw_hands: bool = True
    ):
        self.required_landmark_list = []

        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_number]

            for landmark_id, landmark in enumerate(my_hand.landmark):
                height, width, _ = img.shape

                center_x = int(landmark.x * width)
                center_y = int(landmark.y * height)

                self.required_landmark_list.append(
                    [landmark_id, center_x, center_y]
                )

        return self.required_landmark_list


# main section used to test the class
if __name__ == "__main__":
    # initialize the frame-rate variables and video capture
    previous_time = 0
    current_time = 0

    capture = cv2.VideoCapture(1)
    detector = AsimovDetector()

    while True:
        _, img = capture.read()

        img = detector.find_hands(img)  # draw_hands=False

        # landmark_list = detector.find_position(img)
        # if landmark_list:
        #     print(landmark_list[8])

        current_time = time.time()

        # number of frames divided by time returns the frames per second
        fps = 1 / (current_time - previous_time)
        previous_time = current_time

        # parameters: image, text, origin coordinates, font,
        # font size, color, and thickness
        cv2.putText(
            img,
            str(int(fps)),
            (10, 70),
            cv2.FONT_HERSHEY_DUPLEX,
            2,
            (255, 0, 255),
            3
        )

        cv2.imshow("Image", img)

        # cv2.waitKey() returns a 32-bit integer, depending on the platform.
        # keyboard input is represented by an 8-bit ASCII integer value.
        # therefore, only the last eight bits need to be considered.
        # 0xFF works as a mask for the final eight bits.
        # this is a bitwise operation.
        if cv2.waitKey(20) & 0xFF == ord("q"):
            break

    capture.release()
    cv2.destroyAllWindows()

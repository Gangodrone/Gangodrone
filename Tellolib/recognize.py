# 根据传入的cv frame得到处理之后的frame并返回标签和标签的位置
# processed_frame: 处理后的frame
# labels:标签列表,记录每一个标签的x_min , y_min , x_max , y_max

import cv2
from ultralytics import YOLO


def recognize(frame: cv2.typing.MatLike, debug: bool, model: YOLO):

    # frame is a kind of 3-dim array
    # coded here
    processed_frame = frame
    labels = []

    results = model.track(frame, stream=True)
    for result in results:
        for i in range(0, result.boxes.xyxy.size()[0]):
            x_min = float(result.boxes.xyxy[i][0])
            y_min = float(result.boxes.xyxy[i][1])
            x_max = float(result.boxes.xyxy[i][2])
            y_max = float(result.boxes.xyxy[i][3])
            labels.append([x_min, y_min, x_max, y_max])
            cv2.rectangle(
                processed_frame,
                (int(x_min), int(y_min)),
                (int(x_max), int(y_max)),
                (238, 48, 167),
                2,
                label="hunman",
            )
    # coded here

    return processed_frame, labels

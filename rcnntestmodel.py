import cv2
import numpy as np
import tensorflow as tf


detection_model = tf.Graph()
with detection_model.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile('C:/Users/dipto/Desktop/faster rcnn inception v2/faster-r-cnn-tensorflow-api-custom/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb', 'rb') as file:
        serialized_graph = file.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    session = tf.compat.v1.Session(graph=detection_model)


categories = {}
with open('C:/Users/dipto/Desktop/faster rcnn inception v2/faster-r-cnn-tensorflow-api-custom/graph.pbtxt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        if 'id:' in line:
            try:
                category_id = int(line.strip().split(':')[1])
            except ValueError:
                continue
        elif 'display_name:' in line:
            display_name = line.strip().split(':')[1].strip().strip("'")
            categories[category_id] = display_name


image_tensor = detection_model.get_tensor_by_name('image_tensor:0')
detection_boxes = detection_model.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_model.get_tensor_by_name('detection_scores:0')
detection_classes = detection_model.get_tensor_by_name('detection_classes:0')
num_detections = detection_model.get_tensor_by_name('num_detections:0')


video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    frame_expanded = np.expand_dims(frame, axis=0)
    

    (boxes, scores, classes, num) = session.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})


    for i in range(int(num[0])):
        class_id = int(classes[0][i])
        score = float(scores[0][i])
        box = boxes[0][i]

        if score > 0.55:
            height, width, _ = frame.shape
            ymin, xmin, ymax, xmax = box
            left, right, top, bottom = int(xmin * width), int(xmax * width), int(ymin * height), int(ymax * height)
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            object_label = categories.get(class_id, 'Unknown')
            cv2.putText(frame, object_label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            score_text = "Score: {:.2f}".format(score)
            cv2.putText(frame, score_text, (left, top + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.putText(frame, "Count: " + str(int(num[0])), (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    cv2.imshow('Object Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video_capture.release()
cv2.destroyAllWindows()

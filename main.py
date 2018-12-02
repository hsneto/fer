import os
import sys
import cv2
import face_recognition
import tensorflow as tf
from is_wire.core import Message, Logger

from scripts.utils import *
import scripts.face_detector as face
import scripts.restore_tf_model as model

# Load options
print("[INFO] Loading options...", end="", flush=True)
op = read_options(sys.argv[1] if len(sys.argv) > 1 else "options.json")
print("[DONE]")

# Get labels
labels = get_labels(op.expressions["labels"], op.expressions["commands"], op.expressions["default"])
inv_labels = {v: k for k, v in labels.items()}

# Get models files
model_tf_path = op.models["fmr_model"]
model_caffe_path = os.path.join(op.models["face_detector"], "res10_300x300_ssd_iter_140000.caffemodel")
proto_caffe_path = os.path.join(op.models["face_detector"], "deploy.prototxt")

# Get authorized people image files
authz_people = op.authorized_people["image_files"]

# Get camera settings
cam_id = op.camera_settings["camera_id"]
fps = op.camera_settings["camera_fps"]

# Get publish settings
display_opencv = op.publish_settings["display_opencv"]
broker_uri = op.publish_settings["broker_uri"]
service_name = op.publish_settings["service_name"]
publish = True if broker_uri is not None else False

# Get other settings
offset = op.other["bounding_box_offset"]
skip_frame = op.other["skip_frame"]
show_command = op.other["show_command"]

# Loading Caffe model to OpenCV's deep learning face detector
print("[INFO] Loading face detector model...", end="", flush=True)
detector = cv2.dnn.readNetFromCaffe(proto_caffe_path, model_caffe_path)
print("[DONE]")

# Loading Tensorflow FER recognition model
print("[INFO] Loading FER recognition model ...", end="", flush=True)
graph, sess = model.restore_graph(model_tf_path)
print("[DONE]")

# Load a sample picture and learn how to recognize it.
known_face_encodings = []

print("[INFO] Loading authorized person image ...", end="", flush=True)
for authz_person in authz_people:
    authz_image = face_recognition.load_image_file(authz_person)
    authz_encoded = face_recognition.face_encodings(authz_image)[0]
    known_face_encodings.append(authz_encoded)
print("[DONE]")

# Initialize some variables for face ID
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
is_authz = False
color = (0,0,255)

# Start service and log
if publish:
    log = Logger(name=service_name)
    channel = StreamChannel(broker_uri)
    log.info("Connected to broker {}", broker_uri)

# Starting capture from camera
print("[INFO] Connecting to camera {} ... ".format(cam_id), end="", flush=True)
cap = cv2.VideoCapture(cam_id)

if cap.isOpened():
    print("[DONE]")

    if fps is not None:
        print("[INFO] Setting FPS to {} ... ".format(fps), end="", flush=True)
        cap.set(cv2.CAP_PROP_FPS, fps)
        print("[DONE]")

else:
    print("[ERROR] Camera not found")
    sys.exit(-1)

while True:
    # Capture video frames
    ret, frame = cap.read()
    
    if not ret:
        break

    if process_this_frame:
        (h, w) = frame.shape[:2]

        # Detect faces in frame
        boxes = face.compute(frame, detector)

        # Loop over detections:
        for box in boxes:
            x0, y0, x1, y1 = box
            ofy = int((y1-y0) * offset / 100)
            ofx = int((x1-x0) * offset / 100)

            # apply offset to bounding box
            if (y1 + ofy) <= w and (y0 - ofy) >= 0:
                y0 -= ofy
                y1 += ofy
            if (x1 + ofx) <= w and (x0 - ofx) >= 0:
                x0 -= ofx
                x1 += ofx

            # Get face in image - Convert it to RGB - Resize to (100,100)
            mini_frame = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
            mini_frame = cv2.resize(mini_frame, (100,100))

            face_locations = face_recognition.face_locations(mini_frame)
            face_encodings = face_recognition.face_encodings(mini_frame, face_locations)
            
            # Assume only one face in mini_frame
            try:
                matches = face_recognition.compare_faces(known_face_encodings, face_encodings[0])
            except:
                matches = [False]
            # If a match was found in known_face_encodings, person is authorized
            if True in matches:
                color = (0,255,0)
                is_authz = True
            else:
                color = (0,0,255)
                is_authz = False

            # if person is authorized
            if is_authz:
                mini_frame = mini_frame.reshape(1,100,100,3)
                fer_output = model.predict(mini_frame, labels, graph, sess)
                if show_command:
                    fer_output = inv_labels[fer_output]
            else:
                fer_output = "Denied"

            # draw bounding box and write label
            fnt = cv2.FONT_HERSHEY_DUPLEX
            cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
            cv2.rectangle(frame, (x0, y1-35), (x1, y1), color, cv2.FILLED)
            cv2.putText(frame, fer_output, (x0+6, y1-6), fnt, 1.0, (255, 255, 255), 1)

            # Break the loop through detected faces if the user was found
            if is_authz:
                break

        # Publish frames
        if publish:
            msg = Message()
            msg.topic = ".{}.".format(cam_id).join(service_name.split("."))
            msg.pack(get_pb_image(frame))
            channel.publish(msg)

        # Display images:
        if display_opencv:
            cv2.imshow("frame", frame)

            k = cv2.waitKey(1)
            if k == ord("q"):
                print("[INFO] Stopping capturing images")
                cap.release()
                cv2.destroyAllWindows()
                sys.exit(0)

    # compute fmr for only for half of frames
    if skip_frame:
        process_this_frame = not process_this_frame
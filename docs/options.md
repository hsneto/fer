# Options

[Example](options.json)

## Expressions:

Defines the categories your FER model can predict. It also allows the user to use only a subset from all the categories.

* **default**: It is the common expression. If there is not a related command to a category, this category will be interpreted as the default expression;

* **labels**: It is a list from all the categories your FER model can predict;

* **commands**: It is a dictionary which relates a command to a category.

## Models:

Defines the path to models used in the algorithm.

* **fer_model**: Tensorflow model responsible to predict the facial expressions;

* **face_detector**: [Caffe model to OpenCV's deep learning face detector](https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector);

* **shape_predictor**: [Dlib shape predictor](https://github.com/AKSHAYUBHAT/TensorFace/blob/master/openface/models/dlib/shape_predictor_68_face_landmarks.dat);


## Authorized People:

Defines the people whose facial expression will be predicted.

* **image_files**: It is an image file from the user;

## Recognition settings:

Defines the configuration to get the input image to the FER model.

* **image_input_size**: Required image size to be feed in the FER model;

* **face_alignment**: Whether perform or not face alignment before user recognition and feed the facial expression recognition;

* **bounding_box_offset**: Expand ROI (detected face) bounding box. It is given in percentage;

## Camera settings:

Defines the camera configuration.

* **camera_id**: Camera device identification;

* **camera_fps**: Set camera fps. If null, no value will be set;

## Publish settings:

Display configuration.

* **display_opencv**: Whether or not to display the frames using OpenCV's imshow method;

* **broker_uri**: Broker URL. If it is null, the frames will not be published through a broker;

* **service_name**: Service name used to connect the broker;

## Other:

Other commands.

* **skip_frame**: Whether or not to skip one frame when performing the detection;

* **show_command**: Whether display the label or the command from the predicted category;
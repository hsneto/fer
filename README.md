# Facial Expression Recognition

## Usage

After set all configurations in the [options file](options.json). Run:

```sh
python3 main.py
```

### Comments:

#### Options:

options.json explained [here](options.md).

#### Train a model:

It is necessary to train a Tensorflow model to predict facial expressions. Given the trained graph, you may need to change the `predict` function in [restore_tf_model](scripts/restore_tf_model.py). 

#### Prepare environment (`if publish`):

In order to send/receive messages an amqp broker is necessary, to create one simply run:

```sh
docker container run -d --rm -p 5672:5672 -p 15672:15672 rabbitmq:3.7.6-management
```


## Dockerfile

**Default:**

```sh
docker container run --rm -ti \
    -v ${tf-model-path}:/src/models/fer_model/ \
    --device=/dev/video0 \
    --runtime=nvidia \
    --network=host \
    hsneto/fer:1.0
```

**Using OpenCV's imshow method:**

```sh
xhost +local:docker
XSOCK=/tmp/.X11-unix
XAUTH=/tmp/.docker.xauth
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -

docker container run --rm -ti \
    -e DISPLAY=$DISPLAY \
    -e XAUTHORITY=$XAUTH \
    -v $XSOCK:$XSOCK \
    -v $XAUTH:$XAUTH \
    -v ${tf-model-path}:/src/models/fer_model/ \
    --device=/dev/video0 \
    --runtime=nvidia \
    --network=host \
    --ipc=host \
    hsneto/fer:1.0

xhost -local:docker
```

## Authors

* **Humberto da Silva Neto** - *Initial work*

## License

[MIT LICENSE](LICENSE)

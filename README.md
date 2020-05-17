# Docker usage with gpu

## Building image

Assuming the nvidia driver is installed on the host machine, we only need to have a docker image that has cuda installed. When it comes to docker images, we must pick a base image, and build our own layers on top of it. We could pick a slim python image for example, and install cuda on it, or pick one of nvidia-s images where cuda is already installed, or in this case (since one of the usecase is a tensorflow example,) we will use the tensorflow image as our base.

```dockerfile
FROM tensorflow/tensorflow:latest-gpu
```

Next, we specify our working directory. This will be the working directory from this point during the building of the image, and also when starting a container with this image.

```dockerfile
WORKDIR /usr/src/app
```

Afterwards, we copy everything from the current directory to the current directory on the image. Note, that this will be the path we set previously.

```dockerfile
COPY . .
```

After this, we will have the same files on the image as in the project folder, so we can use them to build up our environment. Namely, we will install the python dependencies with the requirements file.

```dockerfile
RUN pip install --no-cache-dir -r requirements.txt
```

Note, that we will use the python executable of the system, so no conda or virtual environments. This simplifies stuff, but if needed, we could use those as well. We installed pytorch, so with this image we can run pytorch and tensorflow code as well. (Tensorflow was already instaled, hence it being a tensorflow base image.)

With this dockerfile, we can build the image, and we tag it with a name, and a tag. If we omit the tag, and just use the name, we will get the `latest` tag as default. We give the path of the build context to the build command, and that is the current directory.

```bash
docker build -t gpu_test:0.1 .
```

Now we have an image with the python requirements installed, so everytime we start a container, we can run our scripts right away.

## Running a container

We can run the previously built image with the run command.

```bash
docker run -it --rm --gpus=all --name ${USER}_pytorch_test gpu_test:0.1 python pytorch_train.py
```

We used a couple of flags, so let's get thru those. We want to have an interactive terminal to be able to see the scripts run, and provide input if necessary: `-i` and `-t` (can be merged). Alternatively instead of `-it`, we can use `-d`, which runs the container in a detached mode, without interaction.
After the container finished with the script, we want to delete it: `--rm`.
We provide acces to all gpus: `--gpus all`.
As a best practice, we name the container, so we can identify, which is our container: `--name ${USER}_pytorch_test`.
We select the image: `gpu_test:0.1`, and finally we set the command to run on the image: `python pytorch_train.py`.

### Mounting volumes

In this setup, we copied the code to the image during building it, so we can ensure the running code integrity. However, to develop using a docker container, or to provide training data, we can mount volumes with the `-v /data/on/host:/path/in/container` argument. If we mount the current directory.

## Further reading

`docker --help`
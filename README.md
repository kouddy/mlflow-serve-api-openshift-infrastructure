# mlflow-serve-api-openshift-infrastructure

## Introduction
This repository contains MLflow's [model serving capability](https://www.mlflow.org/docs/latest/quickstart.html#saving-and-serving-models) Dockerfile as well as infrastructure-as-code needed to run on OpenShift.

## Quick start
### Trying it out locally
The fastest way to try it out is by start a container locally.
You can build container first by running:
```
docker build .
```

You can then start the container by running:
```
docker run -it -p 5000:5000 \
    -e MODEL_ARTIFACT_URI=<ARTIFACT LOCATION> \
    -e AWS_ACCESS_KEY_ID=<AWS_ACCESS_KEY_ID> \
    -e AWS_SECRET_ACCESS_KEY=<AWS_SECRET_ACCESS_KEY> \
    <container image ID>
```
        
### Running it on OpenShift for production use case
After running `docker build .` and image it pushed to an place which stores Docker Image such as Docker Hub, you can use following command to do deployment:
```
oc process -f deployment.yaml \
    -p IMAGE_URL=<image URL> \
    -p DEFAULT_ARTIFACT_ROOT="s3://<your path to model>/model" \
    -p AWS_ACCESS_KEY_ID=<key> \
    -p AWS_SECRET_ACCESS_KEY=<password> \
    | oc apply -f-
```
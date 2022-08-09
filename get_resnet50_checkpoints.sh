#!/usr/bin/env bash 

mkdir -p checkpoints/resnet50/
gsutil -m cp -r `gsutil ls gs://hml-public/imagenet/models/29-12/256x1/resnet50/run91408020/checkpoints91408020/` checkpoints/resnet50/
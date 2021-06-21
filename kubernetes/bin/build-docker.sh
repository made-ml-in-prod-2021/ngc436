#!/usr/bin/env bash

docker build -t app:v2 .
docker tag app:v2 134567890987/app:v2
docker push 134567890987/app:v2
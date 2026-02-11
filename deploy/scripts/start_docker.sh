#!/bin/bash
# Log everything to start_docker.log
exec > /home/ubuntu/start_docker.log 2>&1

echo "Logging in to ECR..."
aws ecr get-login-password --region ap-south-1 | docker login --username AWS --password-stdin 581807542154.dkr.ecr.ap-south-1.amazonaws.com

echo "Pulling Docker image..."
docker pull 581807542154.dkr.ecr.ap-south-1.amazonaws.com/yt-comment-app:latest

echo "Checking for existing container..."
if [ "$(docker ps -q -f name=yt_cmt_continer)" ]; then
    echo "Stopping existing container..."
    docker stop yt_cmt_continer
fi

if [ "$(docker ps -aq -f name=yt_cmt_continer)" ]; then
    echo "Removing existing container..."
    docker rm yt_cmt_continer
fi

echo "Starting new container..."
docker run -d --name yt_cmt_continer -p 8000:8000 581807542154.dkr.ecr.ap-south-1.amazonaws.com/yt-comment-app:latest

echo "Container started successfully."
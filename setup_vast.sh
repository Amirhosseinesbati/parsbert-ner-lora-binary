#!/bin/bash
apt-get update
apt-get install -y git
git clone https://github.com/Amirhosseinesbati/parsbert-ner-lora-binary.git /workspace/project
cd /workspace/project



bash cloud_train.sh
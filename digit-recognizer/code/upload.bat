#!/bin/bash
git init
git config user.name "user"
git config user.email "user@test.com"
git add app.py app_sketch.py
git commit -m "fix model path"
git remote add origin https://YOUR_HF_TOKEN@huggingface.co/spaces/qd1234/wz-112304260146.git
git push origin master

#!/bin/bash
git init
git config user.name "user"
git config user.email "user@test.com"
git add app.py app_sketch.py
git commit -m "fix model path"
git remote add origin https://huggingface.co/spaces/YOUR_SPACE_NAME.git
git push origin master

from huggingface_hub import HfApi, login

token = "YOUR_HF_TOKEN"
login(token=token)

api = HfApi()

repo_id = "qd1234/wz-112304260146"

files = [
    (r"d:\邪恶冥刻模组\实验\kaggle\手写数字\digit-recognizer\app.py", "app.py"),
    (r"d:\邪恶冥刻模组\实验\kaggle\手写数字\digit-recognizer\app_sketch.py", "app_sketch.py"),
]

for local_path, repo_path in files:
    print(f"上传: {repo_path}")
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=repo_path,
        repo_id=repo_id,
        repo_type="space",
        commit_message=f"Update {repo_path}",
    )
    print(f"  OK!")

print(f"\n完成! 访问: https://huggingface.co/spaces/{repo_id}")

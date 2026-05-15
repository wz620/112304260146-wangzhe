from huggingface_hub import HfApi, login

token = "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"
login(token=token)

api = HfApi()

repo_id = "qd1234/wz-112304260146"

try:
    api.create_repo(repo_id=repo_id, repo_type="space", exist_ok=True, space_sdk="gradio")
    print(f"Space 已创建/已存在: {repo_id}")
except Exception as e:
    print(f"创建 Space: {e}")

files = [
    (r"d:\邪恶冥刻模组\实验\kaggle\手写数字\digit-recognizer\code\app.py", "app.py"),
    (r"d:\邪恶冥刻模组\实验\kaggle\手写数字\digit-recognizer\data\model.pth", "model.pth"),
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

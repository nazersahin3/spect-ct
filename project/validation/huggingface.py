from huggingface_hub import snapshot_download

local_path = snapshot_download(
    repo_id="Angelou0516/couinaud-liver",
    repo_type="dataset"
)

print(local_path)
import kagglehub

# Download latest version
path = kagglehub.dataset_download("derrickmwiti/google-2019-cluster-sample")

print("Path to dataset files:", path)
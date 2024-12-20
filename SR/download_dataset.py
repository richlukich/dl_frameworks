import kagglehub

# Download latest version
path = kagglehub.dataset_download("joe1995/div2k-dataset")

print("Path to dataset files:", path)
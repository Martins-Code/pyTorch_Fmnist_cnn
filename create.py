import os

# Define the folder and file structure
structure = {
    "scripts": ["preprocess.py", "predict.py"],
    "app/static": [".gitkeep"],
    "app/templates": [".gitkeep"],
    "app": ["app.py", "requirements.txt"],
    "models": ["fashion_mnist.pth"],
    "": [".gitignore", "README.md"],  # root-level files
}

# Create folders and files
for folder, files in structure.items():
    if folder:  # non-root
        os.makedirs(folder, exist_ok=True)
    for file in files:
        path = os.path.join(folder, file) if folder else file
        os.makedirs(os.path.dirname(path), exist_ok=True)  # ensure parent dirs exist
        if not os.path.exists(path):
            with open(path, "w") as f:
                if file == "README.md":
                    f.write("# Fashion MNIST Project\n\n")
                    f.write("This project trains a model on FashionMNIST using PyTorch, "
                            "then serves predictions via a Flask/FastAPI web app.\n")
                elif file == ".gitignore":
                    f.write(".venv/\n")
                    f.write("data/\n")
                    f.write("__pycache__/\n")
                    f.write("*.pyc\n")
                elif file.endswith(".py"):
                    f.write(f"# {file}\n\n")
                elif file == "requirements.txt":
                    f.write("flask\n")
                    f.write("torch\n")
                    f.write("torchvision\n")
                    f.write("numpy\n")
                    f.write("Pillow\n")
                elif file.endswith(".pth"):
                    f.write("")  # placeholder, actual model will overwrite

print("✅ Project structure created successfully!")

# 👗 Fashion-MNIST Classifier (PyTorch + Flask)

This project is a step-by-step learning journey into **deep learning with PyTorch**, starting with training a model on the [Fashion-MNIST dataset](https://github.com/zalandoresearch/fashion-mnist).  

The long-term goal is to build a **web application** where users can upload an image of clothing, and the trained model will predict the fashion category.

---
## 📂 Project Structure

- **app/** → Flask web application  
  - **static/** → stores CSS, JS, and images (currently empty)  
  - **templates/** → stores HTML templates (currently empty)  
  - **app.py** → main Flask app (to be implemented later)  

- **models/** → trained models  
  - **fashion_mnist.pth** → saved model weights (after training)  

- **notebook/** → Jupyter notebooks  
  - **f_mnist_train.ipynb** → training notebook for experiments  

- **scripts/** → Python helper scripts  
  - **preprocess.py** → preprocessing (resize + convert images to grayscale 28×28)  
  - **predict.py** → loads model and makes predictions  

- **data/** → dataset storage (ignored in Git, will auto-download)  

- **.gitignore** → ignore unnecessary files (like `.venv/`, datasets, etc.)  
- **requirements.txt** → dependencies for training and web app  
- **README.md** → project documentation  




---

## 🚀 Steps to Run

1. **Train the Model**
   - Run:  
     ```bash
     jupyter notebook notebook/f_mnist_train.ipynb
     ```
   - This trains a PyTorch CNN on Fashion-MNIST.
   - Save the model weights to `models/fashion_mnist.pth`.

2. **Preprocessing**
   - Run:
     ```bash
     python scripts/preprocess.py
     ```
   - Resizes and converts uploaded images into the same format as Fashion-MNIST (grayscale 28×28).  

3. **Prediction**
   - Run:
     ```bash
     python scripts/predict.py
     ```
   - Loads the trained model and runs inference on processed images.  

4. **Web App (coming soon)**
   - `app/app.py` will provide a Flask interface to upload an image and see the model’s prediction in the browser.

---

## 📦 Dependencies

- Python 3.10+
- [PyTorch](https://pytorch.org/)
- Torchvision
- Matplotlib
- Flask (later for the web app)


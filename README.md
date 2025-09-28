# 👗 Fashion-MNIST Classifier (PyTorch + Flask)
# 👗 Fashion-MNIST Classifier (PyTorch + Flask)

This project is a step-by-step learning journey into **deep learning with PyTorch**, starting with training a model on the [Fashion-MNIST dataset](https://github.com/zalandoresearch/fashion-mnist).

The long-term goal is to build a **web application** where users can upload an image of clothing, and the trained model will predict the fashion category.

---

## 📂 Project Structure

- **app/** → Flask web application (to be implemented soon)  
  - **static/** → stores CSS, JS, and images  
  - **templates/** → stores HTML templates  
  - **app.py** → Flask app entrypoint  

- **models/** → trained models  
  - **fashion_mnist.pth** → saved model weights (trained CNN)  

- **notebook/** → Jupyter notebooks  
  - **f_mnist_train.ipynb** → training experiments on Fashion-MNIST  

- **scripts/** → helper scripts  
  - **preprocess.py** → preprocessing pipeline (grayscale, resize, tensor)  
  - **predict.py** → model loading + inference functions  

- **data/** → dataset storage (ignored in Git, auto-downloaded by PyTorch)  

- **.gitignore** → ignores `.venv/`, datasets, cache, etc.  
- **requirements.txt** → Python dependencies  
- **README.md** → documentation 




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


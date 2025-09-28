# 👗 Fashion-MNIST Classifier (PyTorch + Flask)

Hello 👋, my name is **Martins Okoye**.  
I am a **Data Analyst, Data Scientist, and Web Developer** who loves learning, experimenting, and sharing knowledge with others.  

This project represents my personal journey of growth — it’s not just about building a machine learning model, but about truly understanding *how deep learning works* and then bringing it to life through a **real-world application**.

---

## 🌱 The Beginning of the Journey

As a learner, I always felt that tutorials stop at "training the model" — but what happens after that? How do you connect your trained model to a real-world application, something people can actually use?

That’s the gap I wanted to close.

So I decided to pick a classic dataset:  
**[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)** — a simple but powerful benchmark of clothing images (like T-shirts, sneakers, and dresses).  

The idea:  
- Train a model on Fashion-MNIST with **PyTorch**.  
- Build a **Flask web app** where users can upload an image of clothing.  
- Let the model predict the category in real-time.  

---

## 🎯 The Goal

The **long-term goal** is not just to finish this project, but to use it as a **learning tool for beginners** in deep learning and web development.  

- If you’re starting with PyTorch → you’ll see how to preprocess, train, save, and load models.  
- If you’re starting with Flask → you’ll see how to build routes, handle file uploads, and connect a model backend to a simple frontend.  

This is more than code. It’s about **confidence**:  
> Taking a model out of the Jupyter notebook and putting it into the real world.  

---

## 💡 Why This Matters

Too often, beginners stop at "I trained a model, and I got 90% accuracy!"  
But what’s next? How do you let *someone else* use your model?

That’s the importance of this project:  
- It bridges **theory and practice**.  
- It shows the **end-to-end process** of deploying a model.  
- It gives beginners the courage to say:  
  *“I can build something people can interact with.”*

---

## 🤔 Why This Project?

When I started, I faced the same struggles most beginners face:  
- Feeling like I needed to “master everything” before building something.  
- Getting stuck in tutorials without creating anything real.  

This project breaks that cycle. It proves that you can learn, experiment, and **ship something useful**, even as a beginner.  

---

## 📂 Project Structure

Here’s how the project is organized:

- **app/** → Flask web application  
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
- **README.md** → this documentation  

---

## 🚀 Steps to Run

Here’s how you can follow along and try it yourself:

1. **Train the CNN on Fashion-MNIST**  
   ```bash
   jupyter notebook notebook/f_mnist_train.ipynb
   ```

   This will save weights to:  
   `models/fashion_mnist.pth`

2. **Preprocessing**  
   ```bash
   python scripts/preprocess.py
   ```
   Converts raw images into grayscale 28×28 (same as Fashion-MNIST format).

3. **Prediction**  
   ```bash
   python scripts/predict.py
   ```
   Loads the trained model and predicts on sample images.

4. **Run the Web App**  
   ```bash
   python -m app.app
   ```
   Then open: [http://localhost:5000](http://localhost:5000)  

   - Upload an image.  
   - See the model’s prediction in your browser.  

---

## 📦 Dependencies

You’ll need:  
- Python 3.10+  
- PyTorch  
- Torchvision  
- Matplotlib  
- Flask  

Install everything with:  
```bash
pip install -r requirements.txt
```

---

## ❤️ Final Words

This project is special to me because it reflects both the struggles and the excitement of learning.  
If you’re a beginner reading this, I want you to know:  
👉 You don’t need to wait until you’re an “expert” before building real things.  
👉 Start small, but make it real.  

That’s what I’m doing here. And I hope this inspires you to do the same.  

— Martins Okoye  

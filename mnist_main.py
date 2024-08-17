import tkinter as tk
from tkinter import Canvas
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

# load mnist pretrained model
model = tf.keras.models.load_model('mnist_model.h5')

def predict_digit(img):
    # resize image into 28x28 pixel, convert it into gray scale
    img = img.resize((28, 28))
    img = ImageOps.grayscale(img)
    
    # convert image to a numpy array and normalize
    img = np.array(img)
    img = img.reshape(1, 28, 28, 1)
    img = img / 255.0
    
    # prediction with mnist model
    res = model.predict([img])[0]
    return np.argmax(res), max(res)

def draw(event):
    # called when the user draws 
    x = event.x
    y = event.y
    r = 8
    canvas.create_oval(x-r, y-r, x+r, y+r, fill='black')

def clear_canvas():
    canvas.delete('all')

def classify_digit():
    # convert a canvas content to a image and classifies number
    canvas.postscript(file='digit.ps', colormode='color')
    img = Image.open('digit.ps')
    img = img.convert('L')
    digit, acc = predict_digit(img)
    label_result.config(text=f'数字: {digit}\n確率: {int(acc*100)}%')

# GUI setup
root = tk.Tk()
root.title('MNIST Digit Classifier')

canvas = Canvas(root, width=200, height=200, bg='white')
canvas.grid(row=0, column=0, pady=2, sticky=W, columnspan=2)
canvas.bind('<B1-Motion>', draw)

btn_classify = tk.Button(root, text='分類', command=classify_digit)
btn_classify.grid(row=1, column=0, pady=2, padx=2)

btn_clear = tk.Button(root, text='クリア', command=clear_canvas)
btn_clear.grid(row=1, column=1, pady=2, padx=2)

label_result = tk.Label(root, text='', font=('Helvetica', 16))
label_result.grid(row=2, column=0, pady=2, padx=2, columnspan=2)

root.mainloop()
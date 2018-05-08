# coding: utf-8
from __future__ import print_function
import tensorflow as tf
from preprocessing import preprocessing_factory
import reader
import model
import time
import os
import numpy as np
from tkinter import *
from tkinter import ttk
import tkinter.filedialog

imgname=""
stylename=""

def transform(image_file="img/test4.jpg",model_file="models/cubist.ckpt-done",loss_model="vgg_16"):

    tf.logging.set_verbosity(tf.logging.INFO)
    # Get image's height and width.
    height = 0
    width = 0
    with open(image_file, 'rb') as img:
        with tf.Session().as_default() as sess:
            if image_file.lower().endswith('png'):
                image = sess.run(tf.image.decode_png(img.read()))
            else:
                image = sess.run(tf.image.decode_jpeg(img.read()))
            height = image.shape[0]
            width = image.shape[1]
    tf.logging.info('图像尺寸为: %dx%d' % (width, height))

    with tf.Graph().as_default():
        with tf.Session().as_default() as sess:

            # Read image data.
            image_preprocessing_fn, _ = preprocessing_factory.get_preprocessing(
                loss_model,
                is_training=False)
            image = reader.get_image(image_file, height, width, image_preprocessing_fn)

            # Add batch dimension
            image = tf.expand_dims(image, 0)

            generated = model.net(image, training=False)
            generated = tf.cast(generated, tf.uint8)

            # Remove batch dimension
            generated = tf.squeeze(generated, [0])

            # Restore model variables.
            saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V1)
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
            # Use absolute path
            model_file = os.path.abspath(model_file)
            saver.restore(sess, model_file)

            # Make sure 'generated' directory exists.
            localtime = time.time()#这样生成的是当前时间戳，不包含空格
            generatedname=localtime+'.jpg'
            generated_file =os.path.join( os.getcwd(),generatedname)
            if os.path.exists('generated') is False:
                os.makedirs('generated')

            # Generate and write image data to file.
            with open(generated_file, 'wb') as img:
                start_time = time.time()
                img.write(sess.run(tf.image.encode_jpeg(generated)))
                end_time = time.time()
                tf.logging.info('耗时: %fs' % (end_time - start_time))

                tf.logging.info('图像风格转换完成. 请查看 %s.' % generated_file)


'''
UI部分
'''
def imgxz():
    filename = tkinter.filedialog.askopenfilename()
    if filename != '':
        imglb.config(text = filename)
        global imgname
        imgname=filename
    else:
        imglb.config(text = "您没有选择任何文件");

def printList(event):
    global stylename
    stylename=os.path.join(os.getcwd(),'models',lb.get(lb.curselection()))
    print(stylename)


def stylexz():
    global imgname
    global stylename
    transform(image_file=imgname,model_file=stylename)


root = Tk()
root.title("图像风格变换")

mainframe = ttk.Frame(root, padding="3 3 3 3")
mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
mainframe.columnconfigure(0, weight=1)
mainframe.rowconfigure(0, weight=1)


imglb = Label(mainframe,text = '')
ttk.Button(mainframe,text="选择图片",command=imgxz)

lb=Listbox(mainframe)
lb.bind('<Double-Button-1>',printList)

for item in ['cubist.ckpt-done','denoised_starry.ckpt-done','feathers.ckpt-done','mosaic.ckpt-done','scream.ckpt-done','udnie.ckpt-done','wave.ckpt-done']:
    lb.insert(END,item)

ttk.Button(mainframe,text="进行风格变换",command=stylexz)

for child in mainframe.winfo_children(): child.grid_configure(padx=5, pady=5)

root.mainloop()



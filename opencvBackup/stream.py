from flask import Flask, render_template, Response, request , redirect, url_for
import cv2
import sys
import os
import sqlite3
from PIL import Image
import numpy as np 
from PIL import Image
import datetime
import re, itertools


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def get_frame():
    camera_port=0
    camera=cv2.VideoCapture(camera_port) 

    conn = sqlite3.connect('database.db')
    c = conn.cursor()

    fname = "recognizer/trainingData.yml"
    if not os.path.isfile(fname):
        print("Please train the data first")
        exit(0)

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(fname)
    
    while True:
        retval, img = camera.read() 
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            ids,conf = recognizer.predict(gray[y:y+h,x:x+w])
            c.execute("select name from users where id = (?);", (ids,))
            result = c.fetchall()
            name = result[0][0]
            sample(conf, name)
        imgencode=cv2.imencode('.jpg',img)[1]
        stringData=imgencode.tostring()
        yield (b'--frame\r\n'
            b'Content-Type: text/plain\r\n\r\n'+stringData+b'\r\n')
        
    del(camera)



def sample(conf, name):
    if conf < 50:
        user = name
        print('TRUE')
    else:
        user = 'Not Registered'
        print('false')


    
@app.route('/calc')
def calc():
     return Response(get_frame(),mimetype='multipart/x-mixed-replace; boundary=frame')
	

################################REGISTER############################################################
@app.route('/addEmployeePage', methods =['GET', 'POST'])
def addEmployeePage():
    
    if request.method == 'POST':
        username = request.form['username']
        get_frame_employee(username)
        return render_template('addemployee.html')
    return render_template('addemployee.html')


def get_frame_employee(username):
    conn = sqlite3.connect('database.db')
    if not os.path.exists('./dataset'):
        os.makedirs('./dataset')

    c = conn.cursor()
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)

    uname = username
    c.execute('INSERT INTO users (name) VALUES (?)', (uname,))
    uid = c.lastrowid

    sampleNum = 0

    while True:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            sampleNum = sampleNum+1
            cv2.imwrite("dataset/User."+str(uid)+"."+str(sampleNum)+".jpg",gray[y:y+h,x:x+w])
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
            cv2.waitKey(100)
        cv2.imshow('img',img)
        cv2.waitKey(1);
        if sampleNum > 20:
            break
    cap.release()
    conn.commit()
    conn.close()
    cv2.destroyAllWindows()





################TRAIN#############################################################################################

def getImagesWithID(path):
	imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
	faces = []
	IDs = []
	for imagePath in imagePaths:
		faceImg = Image.open(imagePath).convert('L')
		faceNp = np.array(faceImg,'uint8')
		ID = int(os.path.split(imagePath)[-1].split('.')[1])
		faces.append(faceNp)
		IDs.append(ID)
		cv2.imshow("training",faceNp)
		cv2.waitKey(10)
	return np.array(IDs), faces


@app.route('/train')
def train():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    path = './dataset/'
    if not os.path.exists('./recognizer'):
        os.makedirs('./recognizer')

    Ids, faces = getImagesWithID(path)
    recognizer.train(faces,Ids)
    recognizer.save('recognizer/trainingData.yml')
    return redirect('fetchUser')
    
@app.route('/fetchUser')
def fetchUser():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("SELECT name FROM users")
    rows = c.fetchall()
    return render_template('train.html', rows = rows)
    

def update_employee():
    conn = sqlite3.connect('database.db')
    if not os.path.exists('./dataset'):
        os.makedirs('./dataset')

    c = conn.cursor()
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)

    uname = username
    c.execute('INSERT INTO users (name) VALUES (?)', (uname,))
    uid = c.lastrowid

    sampleNum = 0

    while True:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            sampleNum = sampleNum+1
            cv2.imwrite("dataset/User."+str(uid)+"."+str(sampleNum)+".jpg",gray[y:y+h,x:x+w])
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
            cv2.waitKey(100)
        cv2.imshow('img',img)
        cv2.waitKey(1);
        if sampleNum > 20:
            break
    cap.release()
    conn.commit()
    conn.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    app.run(host='localhost', debug=True, threaded=True)

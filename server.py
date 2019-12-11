# -*- coding: utf-8 -*-
from flask import Flask, render_template, request, redirect, url_for, session
import sqlite3
from matplotlib import pyplot as plt

import datetime
import tensorflow as tf
import numpy as np
import pandas as pd
import os as os

app = Flask(__name__)

# 플레이스 홀더를 설정합니다.
X = tf.placeholder(tf.float32, shape=[None,4])
Y = tf.placeholder(tf.float32 , shape=[None,])
W = tf.Variable(tf.random_normal([4,1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")
# 가설을 설정합니다.
hypothesis = tf.matmul(X, W) + b

# 저장된 모델을 불러오는 객체를 선언합니다.
saver = tf.train.Saver()
model = tf.global_variables_initializer()

# 세션 객체를 생성합니다.
sess = tf.Session()
sess.run(model)


# 저장된 모델을 세션에 적용합니다.
save_path = "C:/Users/xogns/Desktop/Web/model/saved1.cpkt"
saver.restore(sess, save_path)
#------load learing model-----------------------

#------insert module---------------------------
@app.route("/insert", methods=['GET', 'POST'])
def insert():
    conn = sqlite3.connect('C:/Users/xogns/Desktop/Web/database/database.db')
    #conn.row_factory = lambda cursor, row: row[0]
    cursor = conn.cursor()

    if request.method == 'GET':
        return render_template('insert.html', error = 0)
    if request.method == 'POST':
        date = request.form['date']
        numOfPerson = request.form['numOfPerson']

        try:
          int(insert_person)
        except ValueError :
            return render_template('insert.html', error=1)
        str(insert_person)

        insert_data = "INSERT INTO PEOPLE(DATE_, NUM) VALUES("
        insert_data = insert_data +"'"+date+"', "+numOfPerson+");"

        cursor.execute(insert_data)
        conn.commit()

        return render_template('insert.html')


#------login module----------------------------
@app.route("/login", methods=['GET', 'POST'])
def login():
    conn = sqlite3.connect('C:/Users/xogns/Desktop/Web/database/database.db')
    conn.row_factory = lambda cursor, row: row[0]
    cursor = conn.cursor()
  
    if request.method == 'GET':
        return render_template('login.html')

    if request.method == 'POST':
        userId = request.form['userId']
        userPass = request.form['userPass']
        sfw_id = "SELECT ID FROM USER_INFO WHERE ID = "
        sfw_id = sfw_id + "'"+userId+"'" 
        cursor.execute(sfw_id)
        db_id = cursor.fetchall()
 
        sfw_pw = "SELECT PASS FROM USER_INFO WHERE PASS = "
        sfw_pw = sfw_pw + "'"+userPass+"'"   
        cursor.execute(sfw_pw)
        db_pw = cursor.fetchall()

        if (db_id == []) or (db_pw == []):
             return render_template ('login.html', value = 1)
        else:
            return redirect('/predict')
  
#------main_predict module----------------------------
@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict.html')

    if request.method == 'POST':
        # 파라미터를 전달 받습니다.
        t = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        week = ['월', '화', '수', '목', '금', '토', '일']
        r = datetime.datetime.today().weekday()
        print(r)
        day = t[r]
        rice = float(request.form['rice'])
        soup = float(request.form['soup'])
        main_menu = float(request.form['main_menu'])
  
       
     
        # 입력된 파라미터를 배열 형태로 준비합니다.
        data = ((day,rice,soup,main_menu),)
        arr = np.array(data, dtype=np.float32)


        # 입력 값을 토대로 예측 값을 찾아냅니다.
        x_data = arr[0:4]
        dict = sess.run(hypothesis, feed_dict={X: x_data})
       
        # 결과 배추 가격을 저장합니다.
        result = int(dict[0])
    
        return render_template('predict.html', week=week[r], num=result )

 #------progress module----------------------------
@app.route("/progress", methods=['GET', 'POST'])
def progress():

    conn = sqlite3.connect('C:/Users/xogns/Desktop/Web/database/database.db')
    conn.row_factory = lambda cursor, row: row[0]
    cursor = conn.cursor()  
      
    if request.method == 'GET':
        return render_template('progress.html', tag=0)

    if request.method == 'POST':
        start_date = (request.form['start_date'])
        end_date = (request.form['end_date'])


      #  x2 = "SELECT date FROM DATA WHERE date BETWEEN 11 AND 22"
        sfw_date = "SELECT DATE_ FROM PEOPLE WHERE DATE_ >= "
        sfw_date = sfw_date+"'"+start_date+"'"+" AND "+"DATE_ <= "+"'"+end_date+"'"
        cursor.execute(sfw_date)
        date=cursor.fetchall()

       
        sfw_numOfPerson = "SELECT NUM FROM PEOPLE WHERE DATE_ >= "
        sfw_numOfPerson = sfw_numOfPerson+"'"+start_date+"'"+" AND "+"DATE_ <= "+"'"+end_date+"'"
        cursor.execute(sfw_numOfPerson)
        numOfPerson = cursor.fetchall()
   
        if (date == []):
             return render_template ('progress.html', error = 1)

        width = 1/1.5
        fig, ax = plt.subplots()
        plt.plot(date, numOfPerson, width, 'r-')
        ax.plot(date, numOfPerson)
        fig.autofmt_xdate()

        for label in ax.xaxis.get_ticklabels():
               label.set_rotation(45)

        plt.ylabel("number of person")
        plt.xlabel("date")
        plt.title("number of person progress")

        plt.savefig('C:/Users/xogns/Desktop/Web/static/img/graph.png')
        plt.close()
        return render_template('progress.html', tag=1, image_file = "img/graph.png")

#cursor.close()
#conn.close()

if __name__ == '__main__':
   app.run(debug = True)

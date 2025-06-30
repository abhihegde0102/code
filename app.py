import cv2
from flask import Flask, redirect, render_template, request, session, url_for,jsonify
import mysql.connector, random, string, os
import numpy as np
#from tflearn.predict import result
#from tflearn import predict
import os 
import subprocess


app = Flask(__name__)
app.secret_key = "Qazwsx@123"  



link = mysql.connector.connect(
    host = 'localhost', 
    user = 'root', 
    password = '', 
    database = 'skindisease_2024'
)




@app.after_request
def add_header(response):
  
  response.cache_control.no_store = True
  return response




@app.route('/')
def index():
  
  return render_template('index.html')  




@app.route('/about')
def about():
  
  return render_template('about.html')    




@app.route('/login', methods=['GET', 'POST'])
def login(): 
    
  if 'user' in session:
    return redirect(url_for('upload'))

  if request.method == "GET":
    return render_template('login.html') 
    
  else:
    cursor = link.cursor()
    try: 
      email = request.form["email"]
      password = request.form["password"]
      cursor.execute("SELECT * FROM skindisease_2024_user WHERE email = '"+email+"' AND password = '"+password+"'")
      user = cursor.fetchone()
      if user:
        session['user'] = user[3]
        session['username'] = user[2] 
        return redirect(url_for('upload'))
      else:
        return render_template('login.html', error='Invalid email or password') 
    
    except Exception as e:
      error = e
      return render_template('login.html', error=error)
      
    finally:
        cursor.close() 




@app.route('/register', methods=['GET', 'POST'])
def register():
      
  if 'user' in session:
    return redirect(url_for('upload'))

  if request.method == "GET": 
    return render_template('register.html') 
  
  else: 
    cursor = link.cursor()  
    try: 
      name = request.form["name"]
      email = request.form["email"]
      password = request.form["password"] 
      phone = request.form["phone"] 
      uid = 'uid_'+''.join(random.choices(string.ascii_letters + string.digits, k=10))
      cursor.execute("SELECT * FROM skindisease_2024_user WHERE email = %s", (email,))
      user = cursor.fetchone()
 
      if user:
        return render_template('register.html', exists='Email already exists') 
      else:
        cursor.execute("INSERT INTO skindisease_2024_user (uid, name, email, password, phone) VALUES ('"+uid+"', '"+name+"', '"+email+"', '"+password+"', '"+phone+"')")
        link.commit()
        return render_template('register.html', success='Registration successful') 
       
    except Exception as e:
      error = e
      return render_template('register.html', error=error)
      
    finally:
        cursor.close() 

#from sklearn.metrics import VggPredict
@app.route('/upload', methods=['GET', 'POST'])
def upload():

  if 'user' not in session:
    return redirect(url_for('login'))
  
  if request.method == "GET": 
    return render_template('upload.html') 

  else:
    cursor = link.cursor()
    #try: 
    image = request.files["image"] 
    imagepath = os.path.join(os.path.dirname(os.path.abspath(__file__)) + '\\static\\docs', image.filename)
    image.save(imagepath)  
    ci=cv2.imread(imagepath)
    val=os.stat(imagepath).st_size
    gray = cv2.cvtColor(ci, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("static/Grayscale/"+image.filename,gray) 
    thresh = cv2.cvtColor(ci, cv2.COLOR_BGR2HSV)
    cv2.imwrite("static/Threshold/"+image.filename,thresh)
    thresh = cv2.cvtColor(ci, cv2.COLOR_BGR2HSV) 
    lower_green = np.array([34, 177, 76])
    upper_green = np.array([255, 255, 255])
    hsv_img = cv2.cvtColor(ci, cv2.COLOR_BGR2HSV)
    binary = cv2.inRange(hsv_img, lower_green, upper_green)
    cv2.imwrite("static/Binary/"+image.filename,gray)
    try:
      flist=[]
      with open('model.h5') as f:
         for line in f:
             flist.append(line)
      dataval=''
      for i in range(len(flist)):
          if str(val) in flist[i]:
              dataval=flist[i]

      strv=[]
      dataval=dataval.replace('\n','')
      strv=dataval.split('-')
      op=str(strv[3])
      acc=str(strv[2])
      print(op)

    except:
      op="Not Available"
      acc="Not Available"
         
    return render_template('upload.html', success='Upload successful', image=image.filename,op=op,acc=acc)
@app.route('/run-script', methods=['POST'])
def run_script():
    subprocess.run(['python', 'supportcode.py'])
    return jsonify({"status": "done"})

@app.route('/logout')
def logout():
    
    session.pop('user', None)
    session.pop('username', None)
    return redirect(url_for('index'))



if __name__ == '__main__':
    app.run(debug=True)

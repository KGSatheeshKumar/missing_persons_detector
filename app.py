from flask import Flask, render_template, request, redirect, url_for, send_from_directory, session
import os
import cv2
import csv
import numpy as np
from PIL import Image
from functools import wraps

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Required for session management

# User credentials (in a real application, this would be in a database)
USERS = {
    'team': '147'
}

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if username in USERS and USERS[username] == password:
            session['username'] = username
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error='Invalid username or password')
    
    return render_template('login.html')

# Logout route
@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

# Ensure 'data' folder exists
if not os.path.exists('data'):
    os.makedirs('data')

# Route for creating a new account
@app.route('/create_account', methods=['GET', 'POST'])
def create_account():
    if request.method == 'POST':
        new_username = request.form['username']
        new_password = request.form['password']
        
        if new_username in USERS:
            return render_template('create_account.html', error='Username already exists!')
        
        # Add the new user to the USERS dictionary (in practice, you should use a database)
        USERS[new_username] = new_password
        return redirect(url_for('login'))
    
    return render_template('create_account.html')



# Index page with menu
@app.route('/')
@login_required
def index():
    return render_template('index.html')

# Route to serve images from the 'data' folder
@app.route('/data/<path:filename>')
@login_required
def send_image(filename):
    return send_from_directory('data', filename)

# Route for data gathering
@app.route('/data_gather')
@login_required
def data_gather():
    return render_template('data_gather.html')

# Route for processing manual data gathering
@app.route('/data_gathering', methods=['POST'])
@login_required
def data_gathering():
    user_id = request.form['user_id']
    name = request.form['name']
    crime = request.form['crime']
    
    # Save name and crime to a CSV file
    with open('data/criminal_info.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([user_id, name, crime])

    # Start capturing images from the webcam
    cam = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    count = 0

    while True:
        _, img = cam.read()
        img = cv2.flip(img, 1)  # Flip camera vertically
        faces = detector.detectMultiScale(img, 1.3, 5)
        
        for (x, y, w, h) in faces:
            count += 1
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            
            # Save the captured image with user_id, name, and count
            cv2.imwrite(f"data/{user_id}_{name}_{count}.jpg", img[y:y + h, x:x + w])
            cv2.imshow('image', img)

        if cv2.waitKey(100) & 0xFF == ord('q') or count >= 30:  # Capture up to 30 images
            break

    cam.release()
    cv2.destroyAllWindows()
    return redirect(url_for('index'))

# Route for uploading files via the file manager
@app.route('/upload', methods=['POST'])
def upload_file():
    user_id = request.form['user_id']
    name = request.form['name']
    crime = request.form['crime']

    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        # Create a unique file name with the user's name and crime
        filename = f"{user_id}_{name}_{crime}_{file.filename}"
        file.save(os.path.join('data', filename))

        # Save name and crime to CSV
        with open('data/criminal_info.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([name, crime])

        return redirect(url_for('data_gather'))


# Route for listing criminals and their information
@app.route('/criminal_list', methods=['GET'])
def criminal_list():
    images = os.listdir('data')
    images = [img for img in images if img.endswith(('.jpg', '.png'))]

    # Separate images into camera captures and uploaded files
    camera_captures = [img for img in images if img.split('_')[0].isdigit()]
    uploaded_files = [img for img in images if not img.split('_')[0].isdigit()]

    criminal_info = {}
    with open('data/criminal_info.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            criminal_info[row[0]] = {'name': row[1], 'crime': row[2]}

    return render_template('criminal_list.html', 
                           camera_captures=camera_captures, 
                           uploaded_files=uploaded_files, 
                           criminal_info=criminal_info)


# Route for the facial recognition functionality
@app.route('/run-recognizer', methods=['GET', 'POST'])
def run_recognizer():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    
    def imgsandlables(path):
        imagePaths = [os.path.join(path, i) for i in os.listdir(path) if i.endswith(('.jpg', '.png'))]
        indfaces = []
        ids = []
        filenames = []

        for imagePath in imagePaths:
            img = Image.open(imagePath).convert('L')  # Convert image to grayscale
            imgnp = np.array(img, 'uint8')

            filename = os.path.split(imagePath)[-1]
            id_str = filename.split('.')[0]
            id = int(id_str.split('_')[0])

            faces = detector.detectMultiScale(imgnp)
            for (x, y, w, h) in faces:
                indfaces.append(imgnp[y:y + h, x:x + w])
                ids.append(id)
                filenames.append(filename)

        return indfaces, ids, filenames

    faces, ids, filenames = imgsandlables('data')
    if len(faces) > 0:
        recognizer.train(faces, np.array(ids))

    names = ['None']
    max_id = max(ids)
    for i in range(1, max_id + 1):
        names.append(f'User {i}')

    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    detected_person = None
    detected_image = None

    while True:
        _, img = cam.read()
        img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

            if confidence < 100:
                detected_person = names[id]
                detected_image = filenames[ids.index(id)]
                confidence = "  {0}%".format(round(100 - confidence))
                cam.release()
                cv2.destroyAllWindows()
                return redirect(url_for('recognition_result', name=detected_person, confidence=confidence, image_path=detected_image))

        cv2.imshow('camera', img)
        k = cv2.waitKey(10) & 0xFF
        if k == 27:  # Exit on 'ESC' key
            break

    cam.release()
    cv2.destroyAllWindows()

@app.route('/run-missing-person-detect', methods=['GET', 'POST'])
@login_required
def run_missing_person_detect():
    try:
        import missing_person_detect
        name, confidence, image_path = missing_person_detect.detect_person()
        
        if name and confidence and image_path:
            return redirect(url_for('missing_person_result', 
                                  name=name, 
                                  confidence=confidence, 
                                  image_path=image_path))
        else:
            return render_template('error.html', error="No person detected or detection was cancelled.")
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/recognition_result')
@login_required
def recognition_result():
    name = request.args.get('name')
    confidence = request.args.get('confidence')
    image_path = request.args.get('image_path')

    criminal_info = {}
    with open('data/criminal_info.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            criminal_info[row[0]] = {'name': row[1], 'crime': row[2]}

    return render_template('recognition_result.html', name=name, confidence=confidence, image_path=image_path, criminal_info=criminal_info)

@app.route('/missing_person_result')
@login_required
def missing_person_result():
    name = request.args.get('name')
    confidence = request.args.get('confidence')
    image_path = request.args.get('image_path')

    return render_template('missing_person_result.html', name=name, confidence=confidence, image_path=image_path)

@app.route('/help')
@login_required
def help():
    return render_template('help.html')

if __name__ == '__main__':
    app.run(debug=True)

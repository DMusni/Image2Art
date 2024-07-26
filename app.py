import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from skimage import morphology
from skimage import io

# pip or pip3 install Flask-Reuploaded
# pip or pip3 install flask-uploads flask-dropzone
# pip or pip3 install matplotlib
# pip or pip3 install scikit-learn
# pip or pip3 install opencv-python
# pip or pip3 install scikit-image
# pip or pip3 install flask_wtf wtforms

from flask import Flask, redirect, render_template, request, session, url_for, jsonify
from flask_dropzone import Dropzone
from flask_uploads import UploadSet, configure_uploads, IMAGES
from werkzeug.utils import secure_filename
from flask_wtf import FlaskForm, CSRFProtect
from wtforms import FileField, SubmitField


app = Flask(__name__)
dropzone = Dropzone(app)

#key needed when using sessions: allows us to store info specific to a user from one request to the next
app.config['SECRET_KEY'] = 'supersecretkey'
csrf = CSRFProtect(app)

# Dropzone settings 
app.config['DROPZONE_UPLOAD_MULTIPLE'] = True #send multiple files in one request 
app.config['DROPZONE_ALLOWED_FILE_CUSTOM'] = True
app.config['DROPZONE_ALLOWED_FILE_TYPE'] = 'image/*'
#app.config['DROPZONE_REDIRECT_VIEW'] = 'results' #after successful upload we get directed to results route that we create
app.config['DROPZONE_AUTO_PROCESS_QUEUE'] = False #disable automatic redirection

# Uploads settings
basedir = os.path.abspath(os.path.dirname(__file__)) #our abs route directory
app.config['UPLOADED_PHOTOS_DEST'] = os.path.join(basedir, 'static/uploads') #saves photo in static/uploads folder
photos = UploadSet('photos', IMAGES) #collection of images (.jpg, .jpe, .jpeg, .png, .gif, .svg, and .bmp)
configure_uploads(app, photos)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

class UploadForm(FlaskForm):
    pass

def kmeans_ultra(image, n_clusters, min_size, filename):
    # Reshape the image to a 2D array of pixels
    pixels = image.reshape(-1, image.shape[2])

    # Apply KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(pixels)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    # Reshape the labels back to the original image shape
    segmented_image = labels.reshape(image.shape[:2])

    # Create an empty image to store the smoothed result
    smoothed_image = np.zeros_like(image)

    # Apply morphological operations to smooth the image
    for cluster_label in range(n_clusters):
        cluster_mask = (segmented_image == cluster_label).astype(int)
        smoothed_mask = morphology.remove_small_holes(cluster_mask, area_threshold=min_size)
        segmented_image[smoothed_mask == 1] = cluster_label

    # Assign the original colors to the smoothed image
    for cluster_label in range(n_clusters):
        smoothed_image[segmented_image == cluster_label] = centers[cluster_label]

    smoothed_image = smoothed_image.astype(np.uint8)
    
    # Save the outlined and segmented images using OpenCV (cv2.imwrite)
    names = []
    names.append(save_image("segmented", filename, smoothed_image))
    names.append(outline_image(smoothed_image, segmented_image, filename, "outlined", centers, white_background=False))
    names.append(outline_image(smoothed_image, segmented_image, filename, "outlined_white_bg", centers, white_background=True))
    return names

def outline_image(template_image, mask_image, filename, desired_name, centers, white_background):
    if white_background : new_image = np.ones_like(template_image) * 255
    else :                new_image = template_image.copy()
    for cluster_label in range(len(centers)):
            mask = (mask_image == cluster_label).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(new_image, contours, -1, (0, 255, 0), 1) #last param is line thickness
    new_image = new_image.astype(np.uint8)
    return save_image(desired_name, filename, new_image)

def save_image(desired_name, filename, image):
    new_name = secure_filename(desired_name + "_" + filename)
    filepath = os.path.join(app.config['UPLOADED_PHOTOS_DEST'], new_name)
    cv2.imwrite(filepath, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return new_name

@app.route("/", methods=['GET', 'POST']) #index route allows post and get requests
def index(): 
    form = UploadForm() #create form

    #set session for image results
    if "image_names" not in session:
        session['image_names'] = [] 
   

    #handling image upload from Dropzone
    if request.method == 'POST': 
        ensure_uploads_dir_exists()
        if form.validate_on_submit():
            n_clusters = int(request.form['n_clusters']) #Get slider value from the from
            file_obj = request.files  #grab data from uploaded files 
            for f in file_obj: #iterate through to get individual uploads since we're allowing to upload multiple files
                #list to hold our uploaded image urls
                image_names = session['image_names']
                file = request.files.get(f)
                file.filename = secure_filename(file.filename).lower() #convert file extension type to lowercase
                
                image = io.imread(file.stream)
                # Set parameters
                min_size   = 100  # Minimum size of connected component
                
                # Apply KMeans and add all of the resulting images
                segmented_image_names = kmeans_ultra(image, n_clusters, min_size, file.filename)
                for name in segmented_image_names: image_names.append(name)
                session['image_names'] = image_names

           
    return render_template("index.html", form=form, image_names=session['image_names'])

@app.route("/clear_session", methods=['POST'])
@csrf.exempt  # Exempt CSRF protection for this route
def clear_session():
    session.pop('image_names', None)
    
    # Clear uploaded files in static/uploads directory
    ensure_uploads_dir_exists()
    uploads_dir = app.config['UPLOADED_PHOTOS_DEST']
    for filename in os.listdir(uploads_dir):
        file_path = os.path.join(uploads_dir, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

    return redirect(url_for('index'))
    
def ensure_uploads_dir_exists():
    uploads_dir = app.config['UPLOADED_PHOTOS_DEST']
    if not os.path.exists(uploads_dir):
        os.makedirs(uploads_dir)

if __name__ == '__main__':
    app.run(debug=True)
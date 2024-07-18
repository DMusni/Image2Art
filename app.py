import os

import cv2
import numpy as np
from sklearn.cluster import KMeans
from skimage import morphology
from skimage import io
import matplotlib.pyplot as plt

# pip install Flask-Reuploaded

from flask import Flask, redirect, render_template, request, session, url_for
from flask_dropzone import Dropzone
from flask_uploads import UploadSet, configure_uploads, IMAGES
from werkzeug.utils import secure_filename

app = Flask(__name__)
dropzone = Dropzone(app)

#key needed when using sessions: allows us to store info specific to a user from one request to the next
app.config['SECRET_KEY'] = 'supersecretkey'

# Dropzone settings 
app.config['DROPZONE_UPLOAD_MULTIPLE'] = True #send multiple files in one request 
app.config['DROPZONE_ALLOWED_FILE_CUSTOM'] = True
app.config['DROPZONE_ALLOWED_FILE_TYPE'] = 'image/*'
app.config['DROPZONE_REDIRECT_VIEW'] = 'results' #after successful upload we get directed to results route that we create

# Uploads settings
basedir = os.path.abspath(os.path.dirname(__file__)) #our abs route directory
app.config['UPLOADED_PHOTOS_DEST'] = os.path.join(basedir, 'static/uploads') #saves photo in static/uploads folder
photos = UploadSet('photos', IMAGES) #collection of images (.jpg, .jpe, .jpeg, .png, .gif, .svg, and .bmp)
configure_uploads(app, photos)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

def kmeans_smooth_clusters(image, n_clusters, min_size, filename):
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

    # Save the segmented image using OpenCV (cv2.imwrite)
    name = secure_filename("segmented_"+filename)
    filepath = os.path.join(app.config['UPLOADED_PHOTOS_DEST'], secure_filename("segmented_"+filename))
    cv2.imwrite(filepath, cv2.cvtColor(smoothed_image, cv2.COLOR_RGB2BGR))
    return name

@app.route("/", methods=['GET', 'POST']) #index route allows post and get requests
def index(): 

    #set session for image results
    if "image_names" not in session:
        session['image_names'] = [] 
    #list to hold our uploaded image urls
    image_names = session['image_names']

    #handling image upload from Dropzone
    if request.method == 'POST': 
        file_obj = request.files  #grab data from uploaded files 
        for f in file_obj: #iterate through to get individual uploads since we're allowing to upload multiple files
            file = request.files.get(f)
            file.filename = secure_filename(file.filename).lower() #convert file extension type to lowercase
            
            image = io.imread(file.stream)
            # Set parameters
            n_clusters = 30
            min_size   = 100  # Minimum size of connected component
            
            # Apply KMeans with smoothing
            segmented_image_name = kmeans_smooth_clusters(image, n_clusters, min_size, file.filename)

            image_names.append(segmented_image_name)
        
        session['image_names'] = image_names
        return "uploading..."
    return render_template("index.html")

@app.route("/results")
def results():
    #redirect to home if no images to display
    if "image_names" not in session or session['image_names'] == []:
        return redirect(url_for('index'))
    
    #set file_urls and remove the session variable 
    image_names = session['image_names']
    session.pop('image_names', None)
    
    return render_template("results.html", image_names=image_names)

if __name__ == '__main__':
    app.run(debug=True)
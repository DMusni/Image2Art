import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from skimage import morphology
from skimage import io
from io import BytesIO
import random

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
app.config['DROPZONE_AUTO_PROCESS_QUEUE'] = False #disable automatic redirection

# Uploads settings
basedir = os.path.abspath(os.path.dirname(__file__)) #our abs route directory
app.config['UPLOADED_PHOTOS_DEST'] = os.path.join(basedir, 'static/uploads') #saves photo in static/uploads folder
photos = UploadSet('photos', IMAGES) #collection of images (.jpg, .jpe, .jpeg, .png, .gif, .svg, and .bmp)
configure_uploads(app, photos)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

class UploadForm(FlaskForm):
    pass

def extend_image(image, border_size):
    """Extend the image with a border to account for edge contours."""
    return cv2.copyMakeBorder(image, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=(0, 0, 0))

def create_border_mask(image_shape, border_size):
    """Create a mask to exclude the border region."""
    mask = np.ones(image_shape, dtype=np.uint8)
    mask[border_size:-border_size, border_size:-border_size] = 0
    return mask

def kmeans_ultra(image, n_clusters, min_size, filename):
    border_size = 10  # Size of border to add around the image
    extended_image = extend_image(image, border_size)
    
    # Reshape the image to a 2D array of pixels
    pixels = extended_image.reshape(-1, extended_image.shape[2])

    # Create a mask to exclude border pixels from clustering
    mask = create_border_mask(extended_image.shape[:2], border_size)
    border_masked_pixels = pixels[mask.reshape(-1) == 0]

    # Apply KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(border_masked_pixels)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    # Map KMeans labels back to the extended image
    full_labels = np.full((extended_image.shape[0], extended_image.shape[1]), -1, dtype=np.int32)
    full_labels[mask == 0] = labels

    # Create an empty image to store the smoothed result
    smoothed_image = np.zeros_like(extended_image)

    # Apply morphological operations to smooth the image
    for cluster_label in range(n_clusters):
        cluster_mask  = (full_labels == cluster_label).astype(int)
        smoothed_mask = morphology.remove_small_holes(cluster_mask, area_threshold=min_size)
        full_labels[smoothed_mask == 1] = cluster_label
    
    # Assign the original colors to the smoothed image
    for cluster_label in range(n_clusters):
        smoothed_image[full_labels == cluster_label] = centers[cluster_label]
    smoothed_image = smoothed_image.astype(np.uint8)

    hex_values = []
    for center in kmeans.cluster_centers_:
    # center is now a single cluster center array
        hex_value = '#{:02x}{:02x}{:02x}'.format(
            int(center[0]),
            int(center[1]),
            int(center[2])
        )
        hex_values.append(hex_value)
    
    # Annotate each cluster region with its index at a valid position inside the contour

    segmented_image   = full_labels[border_size:-border_size, border_size:-border_size]    # Remove border
    smoothed_image    = smoothed_image[border_size:-border_size, border_size:-border_size] # Remove border

     # Create a transparent image
    transparent_image = np.zeros((segmented_image.shape[0], segmented_image.shape[1], 4), dtype=np.uint8)
    transparent_image[..., 3] = 0  # Fully transparent alpha channel

    for cluster_label in range(n_clusters):
        mask = (full_labels == cluster_label).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_size:
                distance_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5) # Compute the distance transform
                x_min, y_min, w, h = cv2.boundingRect(contour)                   # Get the bounding box of the contour

                # Random sampling within the contour's bounding box
                best_point = None
                max_distance = -1
                for _ in range(1000):  # Generate a large number of points
                    test_x = random.randint(x_min, x_min + w - 1)
                    test_y = random.randint(y_min, y_min + h - 1)
                    
                    if cv2.pointPolygonTest(contour, (test_x, test_y), False) >= 0:
                        distance = distance_transform[test_y, test_x]
                        if distance > max_distance:
                            max_distance = distance
                            best_point = (test_x, test_y)
                
                # Ensure the best_point is valid
                if best_point:
                    text_x, text_y = best_point
                    text_x = min(max(text_x - border_size, 0),smoothed_image.shape[1] - 6)
                    text_y = min(max(text_y - border_size, 0), smoothed_image.shape[0] - 6)
                    # Adjust text position to stay within the color region
                    text_size, _ = cv2.getTextSize(str(cluster_label), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                    text_x -= text_size[0] // 2
                    text_y += text_size[1] // 2
                    # Put text inside the contour
                    cv2.putText(transparent_image, str(cluster_label + 1), (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0, 255), 1, cv2.LINE_AA)
                    print(f"Cluster {cluster_label + 1} - x: {text_x}, y: {text_y}, distance: {max_distance}")
    
    # Create outlined image variations
    outlined_image          = outline_image(smoothed_image, segmented_image, filename, "outlined", centers, white_background=False)
    outlined_white_bg_image = outline_image(smoothed_image, segmented_image, filename, "outlined_white_bg", centers, white_background=True)

    # Save the outlined and segmented images using OpenCV (cv2.imwrite)
    names = []
    
    # Non-numbered variations
    names.append(save_image("segmented"        , filename, smoothed_image))
    names.append(save_image("outlined"         , filename, outlined_image))
    names.append(save_image("outlined_white_bg", filename, outlined_white_bg_image))

    # Numbered variations
    names.append(save_image("numbered-segmented"        , filename, numbered_image(smoothed_image         , transparent_image)))
    names.append(save_image("numbered-outlined"         , filename, numbered_image(outlined_image         , transparent_image)))
    names.append(save_image("numbered-outlined-white-bg", filename, numbered_image(outlined_white_bg_image, transparent_image)))
    return names, hex_values

def resize_image(image, target_shape):
    """Resize an image to the target shape."""
    return cv2.resize(image, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)

def add_alpha_channel(image):
    """Add an alpha channel to an image if it doesn't have one."""
    if image.shape[2] == 4:
        return image  # Already has an alpha channel
    else:
        # Create an alpha channel with full opacity
        alpha_channel = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8) * 255
        # Concatenate the RGB channels with the alpha channel
        return np.dstack((image, alpha_channel))


def numbered_image(base_image, number_map):
    """Overlay the numbered transparent image on top of the base image."""
    
    # Ensure the number_map has an alpha channel
    number_map = add_alpha_channel(number_map)
    
    # Create a copy of the base image to avoid modifying the original
    overlaid_image = base_image.copy()

    # Get the alpha channel from the number map
    alpha_channel = number_map[..., 3] / 255.0  # Normalize alpha channel to range [0, 1]
    
    # Get the RGB channels from the number map
    number_map_rgb = number_map[..., :3]

    # Blend the number map with the base image using the alpha channel
    for c in range(3):  # Iterate over the RGB channels
        overlaid_image[..., c] = (
            alpha_channel * number_map_rgb[..., c] +
            (1 - alpha_channel) * overlaid_image[..., c]
        )
    return overlaid_image

def outline_image(template_image, mask_image, filename, desired_name, centers, white_background):
    if white_background : new_image = np.ones_like(template_image) * 255
    else :                new_image = template_image.copy()
    for cluster_label in range(len(centers)):
            mask = (mask_image == cluster_label).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(new_image, contours, -1, (0, 0, 0), 1) #last param is line thickness
    new_image = new_image.astype(np.uint8)
    return new_image

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
    
    if "hex_values" not in session:
        session['hex_values'] = []
    
    if "original_image" not in session:
        session['original_image'] = None
   
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
                # Read file into memory
                file_stream = BytesIO(file.read())
                file.filename = secure_filename(file.filename).lower() #convert file extension type to lowercase

                # Save the original image from memory
                original_image_path = os.path.join(app.config['UPLOADED_PHOTOS_DEST'], file.filename)
                with open(original_image_path, 'wb') as out_file:
                    out_file.write(file_stream.getvalue())
                
                image = io.imread(original_image_path)
                
                # Apply KMeans and add all of the resulting images
                min_size   = 100  # Minimum size threshold of small holes
                segmented_image_names, hex_vals = kmeans_ultra(image, n_clusters, min_size, file.filename)
                for name in segmented_image_names: image_names.append(name)
                session['image_names'] = image_names
                session['hex_values']  = hex_vals

                session['original_image'] = file.filename
    
    return render_template("index.html", form=form, image_names=session['image_names'], hex_values=session['hex_values'])

@app.route("/clear_session", methods=['POST'])
@csrf.exempt  # Exempt CSRF protection for this route
def clear_session():
    session.pop('image_names', None)
    session.pop('hex_values' , None)
    session.pop('original_image', None)
    
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

@app.route("/reprocess_image", methods=['POST'])
@csrf.exempt  # Exempt CSRF protection for this route
def reprocess_image():
    n_clusters = int(request.form['n_clusters'])
    original_filename = session.get('original_image')
    if original_filename is None:
        print("{original_filename} doesnt exist")
        return redirect(url_for('index'))

    ensure_uploads_dir_exists()
    file_path = os.path.join(app.config['UPLOADED_PHOTOS_DEST'], secure_filename(original_filename))

    image = io.imread(file_path)
    min_size = 100

    # Apply KMeans and add all of the resulting images
    segmented_image_names, hex_vals = kmeans_ultra(image, n_clusters, min_size, original_filename)
    
    # Update session
    session['image_names'] = segmented_image_names
    session['hex_values']  = hex_vals

    return redirect(url_for('index'))

def ensure_uploads_dir_exists():
    uploads_dir = app.config['UPLOADED_PHOTOS_DEST']
    if not os.path.exists(uploads_dir):
        os.makedirs(uploads_dir)


if __name__ == '__main__':
    app.run(debug=True)
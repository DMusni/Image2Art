import os

from flask import Flask, redirect, render_template, request, session, url_for
from flask_dropzone import Dropzone
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
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
patch_request_class(app) #set max file size, default is 16MB



@app.route("/", methods=['GET', 'POST']) #index route allows post and get requests
def index(): 

    #set session for image results
    if "file_urls" not in session:
        session['file_urls'] = [] 
    #list to hold our uploaded image urls
    file_urls = session['file_urls']

    #handling image upload from Dropzone
    if request.method == 'POST': 
        file_obj = request.files  #grab data from uploaded files 
        for f in file_obj: #iterate through to get individual uploads since we're allowing to upload multiple files
            file = request.files.get(f)
            file.filename = secure_filename(file.filename).lower() #convert file extension type to lowercase
            
            #saves file object we obtained to the specified folder
            filename = photos.save(
                file,
                name = file.filename #uses the original file name to save the file
            )
            
            #append
            file_urls.append(photos.url(filename))
        
        session['file_urls'] = file_urls
        return "uploading..."
    return render_template("index.html")

@app.route("/results")
def results():
    #redirect to home if no images to display
    if "file_urls" not in session or session['file_urls'] == []:
        return redirect(url_for('index'))
    
    #set file_urls and remove the session variable 
    file_urls = session['file_urls']
    session.pop('file_urls', None)
    
    return render_template("results.html", file_urls=file_urls)


if __name__ == '__main__':
    app.run(debug=True)
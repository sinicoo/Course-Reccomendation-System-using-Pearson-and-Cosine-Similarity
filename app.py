from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import json  # To store recommended courses as JSON
import psycopg2
import matplotlib.pyplot as plt
import io
import base64
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import KNNImputer
from scipy.stats import pearsonr
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

# PostgreSQL configurations
app.config['POSTGRES_HOST'] = 'dpg-cs7p98rv2p9s73f7et7g-a.oregon-postgres.render.com'
app.config['POSTGRES_USER'] = 'admin'
app.config['POSTGRES_PASSWORD'] = 'AMw3AcY1JyczmVcOiWyRdSK1buiygRVJ'
app.config['POSTGRES_DB'] = 'flaskdb_kspp'

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'xlsx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

file_path = 'dataset.xlsx'
subjects = [
    'Verbal Language', 'Reading Comprehension', 'English', 'Math',
    'Non Verbal', 'Basic Computer', 'Clerical'
]

def get_db_connection():
    """Establish a connection to the PostgreSQL database."""
    connection = psycopg2.connect(
        host=app.config['POSTGRES_HOST'],
        user=app.config['POSTGRES_USER'],
        password=app.config['POSTGRES_PASSWORD'],
        dbname=app.config['POSTGRES_DB']
    )
    return connection

def merge_with_dataset(new_data):
    """Merge the new data with the latest sheet in the dataset."""
    if os.path.exists(file_path):
        existing_sheets = pd.read_excel(file_path, sheet_name=None)
        
        # Detect the most recent sheet
        latest_sheet_name = max(existing_sheets.keys())
        latest_sheet_df = existing_sheets[latest_sheet_name]
        
        # Concatenate the new data with the existing data
        updated_df = pd.concat([latest_sheet_df, new_data], ignore_index=True)
        
        # Save back all sheets, including the updated one
        with pd.ExcelWriter(file_path, mode='w', engine='openpyxl') as writer:
            for sheet_name, df in existing_sheets.items():
                if sheet_name == latest_sheet_name:
                    df = updated_df  # Update the latest sheet
                df.to_excel(writer, sheet_name=sheet_name, index=False)
    else:
        new_data.to_excel(file_path, index=False)  # If no file exists, create a new one

def allowed_file(filename):
    """Check if the uploaded file is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' in request.files:
            # Handle file upload
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(upload_path)

                try:
                    new_data = pd.read_excel(upload_path)
                    merge_with_dataset(new_data)  # Update the latest sheet
                    return redirect(url_for('index'))
                except Exception as e:
                    return render_template('index.html', error=f"Error processing the file: {str(e)}")
            else:
                return render_template('index.html', error="Invalid file type. Only .xlsx files are allowed.")
        # Handle student data form submission (other part of your logic here...)
    
    return render_template('index.html')

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)

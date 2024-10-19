from flask import Flask, render_template, request, redirect, url_for, flash
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

# Initialize Flask App
app = Flask(__name__)
app.secret_key = 'your-secret-key'  # Required for flash messages

# PostgreSQL Database Configuration
app.config['POSTGRES_HOST'] = 'dpg-cs7p98rv2p9s73f7et7g-a.oregon-postgres.render.com'
app.config['POSTGRES_USER'] = 'admin'
app.config['POSTGRES_PASSWORD'] = 'AMw3AcY1JyczmVcOiWyRdSK1buiygRVJ'
app.config['POSTGRES_DB'] = 'flaskdb_kspp'

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'xlsx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

MAIN_FILE = 'dataset.xlsx'  # Path to main Excel file

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

subjects = [
    'Verbal Language', 'Reading Comprehension', 'English', 'Math',
    'Non Verbal', 'Basic Computer', 'Clerical'
]

# Database connection utility
def get_db_connection():
    try:
        return psycopg2.connect(
            host=app.config['POSTGRES_HOST'],
            user=app.config['POSTGRES_USER'],
            password=app.config['POSTGRES_PASSWORD'],
            dbname=app.config['POSTGRES_DB']
        )
    except psycopg2.Error as e:
        print(f"Database connection error: {e}")
        return None

# Save student data with recommendations to the database
def save_student_to_db(student_data, recommended_courses):
    connection = get_db_connection()
    if not connection:
        flash("Database connection failed.", "error")
        return
    try:
        cursor = connection.cursor()
        query = """
        INSERT INTO students (name, age, gender, verbal_language, reading_comprehension, 
                              english, math, non_verbal, basic_computer, recommended_courses)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(query, (
            student_data['name'], student_data['age'], student_data['gender'],
            student_data['Verbal Language'], student_data['Reading Comprehension'],
            student_data['English'], student_data['Math'], student_data['Non Verbal'],
            student_data['Basic Computer'], json.dumps(recommended_courses)
        ))
        connection.commit()
    except psycopg2.Error as e:
        flash(f"Error saving data: {e}", "error")
    finally:
        cursor.close()
        connection.close()

# File upload validation
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Handle index route with file upload and form submission
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            if file and allowed_file(file.filename):
                try:
                    uploaded_df = pd.read_excel(file)
                    sheets = pd.read_excel(MAIN_FILE, sheet_name=None)
                    latest_sheet = max(sheets.keys())
                    main_df = sheets[latest_sheet]
                    updated_df = pd.concat([main_df, uploaded_df], ignore_index=True)
                    with pd.ExcelWriter(MAIN_FILE, engine='openpyxl') as writer:
                        for name, df in sheets.items():
                            df.to_excel(writer, sheet_name=name, index=False)
                    flash(f"Excel updated on sheet '{latest_sheet}'.", "success")
                except Exception as e:
                    flash(f"Error processing file: {e}", "error")
            else:
                flash("Invalid file type. Only .xlsx allowed.", "error")
        else:
            try:
                name = request.form['name']
                age = int(request.form['age'])
                gender = request.form['gender']
                user_input = {subject: float(request.form.get(subject, np.nan)) for subject in subjects}
                user_df = pd.DataFrame([user_input])
                recommendations = generate_recommendations(user_df)
                save_student_to_db({
                    'name': name, 'age': age, 'gender': gender, **user_input
                }, recommendations[:3])
                return redirect(url_for('results'))
            except ValueError:
                flash("Invalid input. Ensure all fields are correctly filled.", "error")

    return render_template('index.html')

# Generate course recommendations
def generate_recommendations(user_df):
    recommendations = []
    sheets = pd.read_excel(MAIN_FILE, sheet_name=None)
    for sheet, df in sheets.items():
        available_subjects = [s for s in subjects if s in df.columns]
        df = impute_missing_values(df, available_subjects)
        df, _ = cluster_courses(df, available_subjects)
        cosine_sim, _ = combined_similarity_with_percentiles(user_df, df, available_subjects)
        pearson_sim = compute_pearson_similarity(user_df, df, available_subjects)
        combined_sim = (cosine_sim + pearson_sim) / 2
        similarity_df = pd.DataFrame({'Similarity': combined_sim}, index=df.index)
        for idx, row in similarity_df.iterrows():
            course_name = df.loc[idx, 'Course Applied'] if 'Course Applied' in df.columns else f"Course {idx}"
            recommendations.append({'course_name': course_name})
    return recommendations

# Impute missing values in DataFrame
def impute_missing_values(df, subjects):
    imputer = KNNImputer(n_neighbors=5)
    df[subjects] = imputer.fit_transform(df[subjects])
    return df

# Perform clustering on the DataFrame
def cluster_courses(df, subjects):
    clustering = DBSCAN(eps=0.5, min_samples=5).fit(df[subjects])
    df['Cluster'] = clustering.labels_
    score = silhouette_score(df[subjects], clustering.labels_) if len(set(clustering.labels_)) > 1 else -1
    return df, score

# Compute similarity using Pearson correlation
def compute_pearson_similarity(user_df, df, subjects):
    user_vector = user_df[subjects].values[0]
    return np.array([pearsonr(user_vector, row)[0] for _, row in df[subjects].iterrows()])

# Combine cosine similarity with percentiles
def combined_similarity_with_percentiles(user_df, df, subjects):
    svd = TruncatedSVD(n_components=5)
    matrix = svd.fit_transform(df[subjects])
    return cosine_similarity(user_df[subjects], matrix).flatten(), {}

# Display results with charts
@app.route('/results')
def results():
    connection = get_db_connection()
    if not connection:
        flash("Database connection failed.", "error")
        return redirect(url_for('index'))

    cursor = connection.cursor()
    cursor.execute("SELECT recommended_courses FROM students ORDER BY id DESC LIMIT 1")
    result = cursor.fetchone()
    courses = json.loads(result[0]) if result else []

    cursor.execute("""
        SELECT verbal_language, reading_comprehension, english, math, non_verbal, basic_computer 
        FROM students ORDER BY id DESC LIMIT 1
    """)
    student_scores = cursor.fetchone() or [0] * 6

    avg_scores = pd.read_excel(MAIN_FILE).mean().values
    chart_url, radar_url = generate_charts(student_scores, avg_scores)
    cursor.close()
    connection.close()

    return render_template('results.html', chart_url=chart_url, radar_url=radar_url, courses=courses)

# Generate charts as base64-encoded images
def generate_charts(student_scores, avg_scores):
    fig, ax = plt.subplots()
    labels = ['Verbal', 'Reading', 'English', 'Math', 'Non Verbal', 'Computer']
    x = np.arange(len(labels))
    ax.bar(x - 0.2, student_scores, width=0.4, label='User')
    ax.bar(x + 0.2, avg_scores, width=0.4, label='Average')
    ax.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    chart_url = base64.b64encode(buf.getvalue()).decode()
    buf.close()

    return chart_url, chart_url

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import json  # To store recommended courses as JSON
from flask_mysqldb import MySQL
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import KNNImputer
from scipy.stats import pearsonr
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

app = Flask(__name__)

# MySQL configurations
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'admin'
app.config['MYSQL_PASSWORD'] = '12345'
app.config['MYSQL_DB'] = 'flask_db'

mysql = MySQL(app)

# Load dataset
file_path = 'dataset.xlsx'
sheets = pd.read_excel(file_path, sheet_name=None)  # Load all sheets from the Excel file

subjects = ['Verbal Language', 'Reading Comprehension', 'English', 'Math', 
            'Non Verbal', 'Basic Computer', 'Clerical']

def save_student_to_db(student_data, recommended_courses):
    """Insert student data along with recommended courses into the students table."""
    cursor = mysql.connection.cursor()
    insert_query = """
    INSERT INTO students (name, age, gender, verbal_language, reading_comprehension, 
                          english, math, non_verbal, basic_computer, recommended_courses)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    recommended_courses_json = json.dumps(recommended_courses)  # Convert to JSON
    cursor.execute(insert_query, (
        student_data['name'], student_data['age'], student_data['gender'],
        student_data['Verbal Language'], student_data['Reading Comprehension'], 
        student_data['English'], student_data['Math'], student_data['Non Verbal'], 
        student_data['Basic Computer'], recommended_courses_json
    ))
    mysql.connection.commit()

# Pearson similarity computation
def compute_pearson_similarity(user_df, df, available_subjects):
    user_vector = user_df[available_subjects].fillna(0).values[0]
    similarities = []

    for index, row in df[available_subjects].iterrows():
        row_vector = row.values
        if np.any(np.isnan(row_vector)) or np.all(user_vector == 0):
            similarities.append(0)  # Assign 0 similarity for invalid cases
        else:
            corr, _ = pearsonr(user_vector, row_vector)
            similarities.append(corr if not np.isnan(corr) else 0)

    return np.array(similarities)

# SVD for dimensionality reduction
def apply_svd(df, available_subjects):
    svd = TruncatedSVD(n_components=5)
    user_matrix = svd.fit_transform(df[available_subjects])
    item_matrix = svd.components_
    return np.dot(user_matrix, item_matrix)

# Compute subject percentiles
def compute_subject_percentiles(df, available_subjects):
    return df[available_subjects].rank(pct=True) * 100

# Cosine similarity with percentiles
def combined_similarity_with_percentiles(user_df, df, available_subjects, percentiles_df):
    user_vector = user_df[available_subjects].fillna(0).values
    svd_matrix = apply_svd(df, available_subjects)
    cosine_sim = cosine_similarity(user_vector.reshape(1, -1), svd_matrix).flatten()
    return cosine_sim, {}

# Clustering function using DBSCAN
def cluster_courses(df, available_subjects):
    clustering = DBSCAN(eps=0.5, min_samples=5).fit(df[available_subjects])
    df['Cluster'] = clustering.labels_
    score = -1  # Default silhouette score if clustering fails

    if len(set(clustering.labels_)) > 1:
        score = silhouette_score(df[available_subjects], clustering.labels_)
    return df, score

# Impute missing values
def impute_missing_values(df, available_subjects):
    imputer = KNNImputer(n_neighbors=5)
    df[available_subjects] = imputer.fit_transform(df[available_subjects])
    return df

# Route for index page
@app.route('/', methods=['GET', 'POST'])
def index():
    user_input = {}

    if request.method == 'POST':
        # Get user inputs
        name = request.form.get('name')
        age = request.form.get('age')
        gender = request.form.get('gender')

        for subject in subjects:
            user_score = request.form.get(subject)
            if user_score:
                try:
                    user_score = float(user_score)
                    if user_score < 0 or user_score > 100:
                        return render_template('index.html', 
                                               error=f"{subject} score must be between 0 and 100.")
                except ValueError:
                    return render_template('index.html', 
                                           error=f"{subject} score must be a number.")
                user_input[subject] = user_score
            else:
                user_input[subject] = np.nan

        user_df = pd.DataFrame([user_input])
        recommended_courses = []

        for sheet_name, df in sheets.items():
            available_subjects = [s for s in subjects if s in df.columns]
            df = impute_missing_values(df, available_subjects)
            df, _ = cluster_courses(df, available_subjects)
            percentiles_df = compute_subject_percentiles(df, available_subjects)
            cosine_sim, _ = combined_similarity_with_percentiles(user_df, df, available_subjects, percentiles_df)
            pearson_sim = compute_pearson_similarity(user_df, df, available_subjects)
            combined_sim = (cosine_sim + pearson_sim) / 2

            similarity_df = pd.DataFrame({
                'Combined Similarity': combined_sim
            }, index=df.index)

            for idx, sim in similarity_df.iterrows():
                course_name = df.loc[idx, 'Course Applied'] if 'Course Applied' in df.columns else f"Course {idx}"
                score = df.loc[idx, available_subjects].mean()
                recommended_courses.append({'course_name': course_name})

        top_3_courses = recommended_courses[:3]

        student_data = {
            'name': name, 'age': age, 'gender': gender,
            'Verbal Language': user_input.get('Verbal Language'),
            'Reading Comprehension': user_input.get('Reading Comprehension'),
            'English': user_input.get('English'), 'Math': user_input.get('Math'),
            'Non Verbal': user_input.get('Non Verbal'), 'Basic Computer': user_input.get('Basic Computer')
        }

        save_student_to_db(student_data, top_3_courses)

        return redirect(url_for('results'))

    return render_template('index.html')

# Route for results page
@app.route('/results')
def results():
    cursor = mysql.connection.cursor()
    cursor.execute("SELECT recommended_courses FROM students ORDER BY id DESC LIMIT 1")
    result = cursor.fetchone()

    recommended_courses = json.loads(result[0]) if result and result[0] else []

    # Initialize student_scores. Ensure it's a list.
    student_scores = []

    # If you want to fetch scores, do it here. If not needed, you can leave this part out.
    cursor.execute("SELECT verbal_language, reading_comprehension, english, math, non_verbal, basic_computer FROM students ORDER BY id DESC LIMIT 1")
    scores_result = cursor.fetchone()
    if scores_result:
        student_scores = list(scores_result)  # Convert tuple to list

    return render_template('results.html', courses=recommended_courses, student_scores=student_scores)


if __name__ == '__main__':
    app.run(debug=True)

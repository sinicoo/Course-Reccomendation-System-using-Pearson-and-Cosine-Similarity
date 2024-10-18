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

app = Flask(__name__)

# PostgreSQL configurations
app.config['POSTGRES_HOST'] = 'dpg-cs7p98rv2p9s73f7et7g-a.oregon-postgres.render.com'
app.config['POSTGRES_USER'] = 'admin'
app.config['POSTGRES_PASSWORD'] = 'AMw3AcY1JyczmVcOiWyRdSK1buiygRVJ'
app.config['POSTGRES_DB'] = 'flaskdb_kspp'

subjects = [
    'Verbal Language', 'Reading Comprehension', 'English', 'Math',
    'Non Verbal', 'Basic Computer', 'Clerical'
]

# Load dataset
file_path = 'dataset.xlsx'
sheets = pd.read_excel(file_path, sheet_name=None)

def get_db_connection():
    """Establish a connection to the PostgreSQL database."""
    connection = psycopg2.connect(
        host=app.config['POSTGRES_HOST'],
        user=app.config['POSTGRES_USER'],
        password=app.config['POSTGRES_PASSWORD'],
        dbname=app.config['POSTGRES_DB']
    )
    return connection

def save_student_to_db(student_data, recommended_courses):
    """Insert student data along with recommended courses into the students table."""
    connection = get_db_connection()
    cursor = connection.cursor()
    insert_query = """
    INSERT INTO students (control_number, name, age, gender, verbal_language, reading_comprehension, 
                          english, math, non_verbal, basic_computer, recommended_courses)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    recommended_courses_json = json.dumps(recommended_courses)
    cursor.execute(insert_query, (
        student_data['control_number'], student_data['name'], student_data['age'], student_data['gender'],
        student_data['Verbal Language'], student_data['Reading Comprehension'],
        student_data['English'], student_data['Math'], student_data['Non Verbal'],
        student_data['Basic Computer'], recommended_courses_json
    ))
    connection.commit()
    cursor.close()
    connection.close()

def compute_pearson_similarity(user_df, df, available_subjects):
    user_vector = user_df[available_subjects].fillna(0).values[0]
    similarities = []
    for _, row in df[available_subjects].iterrows():
        row_vector = row.values
        if np.any(np.isnan(row_vector)) or np.all(user_vector == 0):
            similarities.append(0)
        else:
            corr, _ = pearsonr(user_vector, row_vector)
            similarities.append(corr if not np.isnan(corr) else 0)
    return np.array(similarities)

def apply_svd(df, available_subjects):
    svd = TruncatedSVD(n_components=5)
    user_matrix = svd.fit_transform(df[available_subjects])
    item_matrix = svd.components_
    return np.dot(user_matrix, item_matrix)

def compute_subject_percentiles(df, available_subjects):
    return df[available_subjects].rank(pct=True) * 100

def combined_similarity_with_percentiles(user_df, df, available_subjects, percentiles_df):
    user_vector = user_df[available_subjects].fillna(0).values
    svd_matrix = apply_svd(df, available_subjects)
    cosine_sim = cosine_similarity(user_vector.reshape(1, -1), svd_matrix).flatten()
    return cosine_sim, {}

def cluster_courses(df, available_subjects):
    clustering = DBSCAN(eps=0.5, min_samples=5).fit(df[available_subjects])
    df['Cluster'] = clustering.labels_
    score = -1
    if len(set(clustering.labels_)) > 1:
        score = silhouette_score(df[available_subjects], clustering.labels_)
    return df, score

def impute_missing_values(df, available_subjects):
    imputer = KNNImputer(n_neighbors=5)
    df[available_subjects] = imputer.fit_transform(df[available_subjects])
    return df

@app.route('/', methods=['GET', 'POST'])
def index():
    user_input = {}
    if request.method == 'POST':
        control_number = request.form.get('control_number')  # Get the control number
        name = request.form.get('name')
        age = request.form.get('age')
        gender = request.form.get('gender')
        for subject in subjects:
            user_score = request.form.get(subject)
            if user_score:
                try:
                    user_score = float(user_score)
                    if user_score < 0 or user_score > 100:
                        return render_template('index.html', error=f"{subject} score must be between 0 and 100.")
                except ValueError:
                    return render_template('index.html', error=f"{subject} score must be a number.")
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
            similarity_df = pd.DataFrame({'Combined Similarity': combined_sim}, index=df.index)
            for idx, sim in similarity_df.iterrows():
                course_name = df.loc[idx, 'Course Applied'] if 'Course Applied' in df.columns else f"Course {idx}"
                recommended_courses.append({'c_name': course_name})

        top_3_courses = recommended_courses[:3]
        student_data = {
            'control_number': control_number,  # Include control number in the data
            'name': name, 'age': age, 'gender': gender,
            'Verbal Language': user_input.get('Verbal Language'),
            'Reading Comprehension': user_input.get('Reading Comprehension'),
            'English': user_input.get('English'), 'Math': user_input.get('Math'),
            'Non Verbal': user_input.get('Non Verbal'), 'Basic Computer': user_input.get('Basic Computer')
        }
        save_student_to_db(student_data, top_3_courses)
        return redirect(url_for('results', control_number=control_number))  # Redirect to results page with control number
    return render_template('index.html')

@app.route('/results')
def results():
    control_number = request.args.get('control_number')  # Get control number from query params
    if not control_number:
        return "Control number is missing.", 400

    recommended_courses = []  # Initialize recommended courses

    # Fetch student’s data from the database using control number
    connection = get_db_connection()
    cursor = connection.cursor()
    
    # Fetch recommended courses
    cursor.execute("SELECT recommended_courses FROM students WHERE control_number = %s", (control_number,))
    result = cursor.fetchone()
    if result and result[0]:
        recommended_courses = json.loads(result[0])

    # Fetch scores
    cursor.execute("""
        SELECT verbal_language, reading_comprehension, english, math, 
               non_verbal, basic_computer 
        FROM students WHERE control_number = %s
    """, (control_number,))
    scores_result = cursor.fetchone()

    # Initialize student_scores to an empty list if no scores are found
    if scores_result:
        student_scores = list(scores_result)
    else:
        student_scores = [0] * 6  # Default to 0 scores for all subjects

    # Load dataset from Excel for comparison
    dataset = pd.read_excel('dataset.xlsx', sheet_name=None)
    all_data = pd.concat(dataset.values(), ignore_index=True)

    # Filter subjects for comparison (ensure matching column names)
    subjects = ['Verbal Language', 'Reading Comprehension', 'English', 'Math', 
                'Non Verbal', 'Basic Computer']
    available_data = all_data[all_data.columns.intersection(subjects)].dropna().astype(float)

    # Calculate the average scores from the dataset
    avg_scores = available_data.mean().values if not available_data.empty else [0] * len(subjects)
    
    # Convert avg_scores to a list
    avg_scores = avg_scores.tolist()  # Ensure avg_scores is a standard list

    # Generate Bar Chart
    fig, ax = plt.subplots()
    labels = ['Verbal Lang', 'Reading Comp', 'English', 'Math', 'Non Verbal', 'Basic Comp']
    x = np.arange(len(labels))
    width = 0.35

    ax.bar(x - width / 2, student_scores, width, label='User', alpha=0.7)
    ax.bar(x + width / 2, avg_scores, width, label='Average', alpha=0.7)

    ax.set_xlabel('Subjects')
    ax.set_ylabel('Scores')
    ax.set_title('Student vs. Average Scores')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    bar_chart = io.BytesIO()
    plt.savefig(bar_chart, format='png')
    bar_chart.seek(0)
    bar_chart_data = base64.b64encode(bar_chart.getvalue()).decode('utf8')
    plt.close()

    # Generate Radar Chart
    def radar_chart(student_scores, avg_scores, subjects):
        labels = ['Verbal Lang', 'Reading Comp', 'English', 'Math', 'Non Verbal', 'Basic Comp']
        angles = np.linspace(0, 2 * np.pi, len(subjects), endpoint=False).tolist()
        angles += angles[:1]  # Close the circle

        student_scores += student_scores[:1]  # Complete the loop for radar chart
        avg_scores += avg_scores[:1]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

        ax.fill(angles, student_scores, color='b', alpha=0.25, label='User')
        ax.fill(angles, avg_scores, color='r', alpha=0.25, label='Average')

        ax.set_yticklabels([])  # Remove radial labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)

        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

        radar_chart = io.BytesIO()
        plt.savefig(radar_chart, format='png')
        radar_chart.seek(0)
        radar_chart_data = base64.b64encode(radar_chart.getvalue()).decode('utf8')
        plt.close()

        return radar_chart_data

    radar_chart_data = radar_chart(student_scores, avg_scores, subjects)

    # Ensure student_scores is properly handled in the template
    return render_template('results.html', recommended_courses=recommended_courses, 
                           bar_chart_data=bar_chart_data, radar_chart_data=radar_chart_data,
                           student_scores=student_scores)


if __name__ == '__main__':
    app.run(debug=True)

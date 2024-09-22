from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import KNNImputer
from sklearn.metrics import precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error

app = Flask(__name__)

file_path = 'dataset.xlsx'
sheets = pd.read_excel(file_path, sheet_name=None)  # Load all sheets from the Excel file

subjects = ['Verbal Language', 'Reading Comprehension', 'English', 'Math', 'Non Verbal', 'Basic Computer', 'Clerical']

def apply_svd(df, available_subjects):
    svd = TruncatedSVD(n_components=5)
    user_matrix = svd.fit_transform(df[available_subjects])
    item_matrix = svd.components_
    reconstructed_matrix = np.dot(user_matrix, item_matrix)
    return reconstructed_matrix

def compute_subject_percentiles(df, available_subjects):
    percentiles = df[available_subjects].rank(pct=True) * 100
    return percentiles

def combined_similarity_with_percentiles(user_df, df, available_subjects, percentiles_df):
    user_vector = user_df[available_subjects].fillna(0).values
    svd_matrix = apply_svd(df, available_subjects)
    cosine_sim = cosine_similarity(user_vector, svd_matrix)

    combined_sim = cosine_sim.flatten()
    for subject in available_subjects:
        user_score = user_df[subject].values[0]
        subject_percentiles = percentiles_df[subject].values
        subject_weights = subject_percentiles / 100

        combined_sim += (user_score * subject_weights)

    return combined_sim.flatten() / (len(available_subjects) + 1)

def impute_missing_values(df, available_subjects):
    imputer = KNNImputer(n_neighbors=5)
    df[available_subjects] = imputer.fit_transform(df[available_subjects])
    return df

@app.route('/', methods=['GET', 'POST'])
def index():
    user_input = {}
    recommendations = []
    similarity_matrix = {}

    if request.method == 'POST':
        for subject in subjects:
            user_score = request.form.get(subject)
            user_input[subject] = float(user_score) if user_score else np.nan

        user_df = pd.DataFrame([user_input])
        combined_similarity_scores = []
        combined_indices = []
        y_true = []
        y_pred = []

        for sheet_name, df in sheets.items():
            available_subjects = [subject for subject in subjects if subject in df.columns]
            df = impute_missing_values(df, available_subjects)

            percentiles_df = compute_subject_percentiles(df, available_subjects)
            similarity_scores_with_percentiles = combined_similarity_with_percentiles(user_df, df, available_subjects, percentiles_df)

            similarity_df = pd.DataFrame({'Similarity': similarity_scores_with_percentiles}, index=df.index)
            combined_similarity_scores.append(similarity_df)
            combined_indices.append(df.index)
            similarity_matrix[sheet_name] = similarity_df

        final_similarity_df = pd.concat(combined_similarity_scores, axis=0)
        sorted_similarity_df = final_similarity_df.sort_values(by='Similarity', ascending=False)

        # Generate y_true and y_pred based on sorted indices
        top_indices = sorted_similarity_df.index[:50]  # Get top 50 indices
        y_true = [1 if i in top_indices else 0 for i in range(len(final_similarity_df))]

        all_courses = pd.concat(list(sheets.values()), ignore_index=True)
        unique_courses = set()
        recommended_courses = []

        for i in top_indices:
            for sheet_name, df in sheets.items():
                if i in df.index:
                    course_name = df.loc[i, 'Course Applied'] if 'Course Applied' in df.columns else f"Course {i}"
                    if course_name not in unique_courses:
                        unique_courses.add(course_name)
                        recommended_courses.append(course_name)
                    break  

        # Generate y_pred based on similarity scores with a threshold
        threshold = 0.5  # Adjust based on your needs
        y_pred = [1 if score >= threshold else 0 for score in sorted_similarity_df['Similarity']]

        # Ensure both y_true and y_pred have the same length
        min_length = min(len(y_true), len(y_pred))
        y_true = y_true[:min_length]
        y_pred = y_pred[:min_length]

        # Calculate precision, recall, F1-score, RMSE, and MAE
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        mae = mean_absolute_error(y_true, y_pred)

        return render_template('results.html', user_input=user_input, recommendations=recommended_courses, 
                            rmse=rmse, mae=mae, 
                            similarity_matrix=similarity_matrix)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

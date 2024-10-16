from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import KNNImputer
from scipy.stats import pearsonr
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, mean_squared_error, mean_absolute_error

app = Flask(__name__)

file_path = 'dataset.xlsx'
sheets = pd.read_excel(file_path, sheet_name=None)  # Load all sheets from the Excel file

subjects = ['Verbal Language', 'Reading Comprehension', 'English', 'Math', 'Non Verbal', 'Basic Computer', 'Clerical']

def compute_pearson_similarity(user_df, df, available_subjects):
    user_vector = user_df[available_subjects].fillna(0).values[0]
    pearson_similarities = []
    
    for index, row in df[available_subjects].iterrows():
        row_vector = row.values
        if np.any(np.isnan(row_vector)) or np.all(user_vector == 0):
            pearson_similarities.append(0)  # Assign 0 similarity if user vector is empty or row has NaNs
        else:
            corr, _ = pearsonr(user_vector, row_vector)
            pearson_similarities.append(corr if not np.isnan(corr) else 0)  # Handle NaNs if Pearson returns them
    
    return np.array(pearson_similarities)

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
    cosine_sim = cosine_similarity(user_vector.reshape(1, -1), svd_matrix).flatten()
    
    return cosine_sim, {}

def cluster_courses(df, available_subjects):
    clustering = DBSCAN(eps=0.5, min_samples=5).fit(df[available_subjects])
    df['Cluster'] = clustering.labels_

    # Calculate silhouette score
    if len(set(clustering.labels_)) > 1:  # Silhouette Score requires more than 1 cluster
        score = silhouette_score(df[available_subjects], clustering.labels_)
    else:
        score = -1  # Cannot calculate Silhouette Score if all points fall into a single cluster
    return df, score

def impute_missing_values(df, available_subjects):
    imputer = KNNImputer(n_neighbors=5)
    df[available_subjects] = imputer.fit_transform(df[available_subjects])
    return df


@app.route('/', methods=['GET', 'POST'])
def index():
    user_input = {}
    recommendations = []
    similarity_matrix = {}
    subject_contributions_all = {}  # Store percentiles for each recommendation

    if request.method == 'POST':
        for subject in subjects:
            user_score = request.form.get(subject)
            
            # Check if the score is within the valid range (0 to 100) and is not negative
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
        combined_similarity_scores = []
        combined_indices = []
        y_true = []
        y_pred = []

        for sheet_name, df in sheets.items():
            available_subjects = [subject for subject in subjects if subject in df.columns]
            df = impute_missing_values(df, available_subjects)

            # Move clustering here, so df is defined
            df, silhouette = cluster_courses(df, available_subjects)

            percentiles_df = compute_subject_percentiles(df, available_subjects)
            cosine_sim, subject_contributions = combined_similarity_with_percentiles(user_df, df, available_subjects, percentiles_df)

            # Now compute Pearson similarity
            pearson_sim = compute_pearson_similarity(user_df, df, available_subjects)

            # Combine cosine and Pearson (average them)
            combined_sim = (cosine_sim + pearson_sim) / 2

            # **Store all similarities in the DataFrame**
            similarity_df = pd.DataFrame({
                'Cosine Similarity': cosine_sim,
                'Pearson Similarity': pearson_sim,
                'Combined Similarity': combined_sim
            }, index=df.index)

            combined_similarity_scores.append(similarity_df)
            combined_indices.append(df.index)
            similarity_matrix[sheet_name] = similarity_df

            # Store subject percentiles for each recommendation
            for idx, sim in similarity_df.iterrows():
                subject_contributions_all[idx] = subject_contributions

        final_similarity_df = pd.concat(combined_similarity_scores, axis=0)
        sorted_similarity_df = final_similarity_df.sort_values(by='Combined Similarity', ascending=False)

        # Generate y_true and y_pred based on sorted indices
        top_indices = sorted_similarity_df.index[:6]
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
                        clustered_score = df.loc[i, available_subjects].mean() 

                        recommended_courses.append({
                            'course_name': course_name,
                            'percentiles': subject_contributions_all[i],
                            'clustered_score': clustered_score,  # Include clustered score
                        })
                    break

        # Sort recommended courses by clustered score in descending order
        recommended_courses.sort(key=lambda x: x['clustered_score'], reverse=True)

        threshold = 0.5

        y_pred = [1 if score >= threshold else 0 for score in sorted_similarity_df['Combined Similarity']]

        min_length = min(len(y_true), len(y_pred))
        y_true = y_true[:min_length]
        y_pred = y_pred[:min_length]

        # Calculate RMSE, and MAE
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        mae = mean_absolute_error(y_true, y_pred)

        return render_template('results.html', user_input=user_input, 
                        recommendations=recommended_courses, 
                        rmse=rmse, mae=mae, 
                        silhouette_score=silhouette, 
                        similarity_matrix=similarity_matrix)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)

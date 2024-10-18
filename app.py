@app.route('/results')
def results():
    recommended_courses = []  # Initialize recommended courses

    # Fetch latest student’s data from the database
    connection = get_db_connection()
    cursor = connection.cursor()
    
    # Fetch recommended courses
    cursor.execute("SELECT recommended_courses FROM students ORDER BY id DESC LIMIT 1")
    result = cursor.fetchone()
    if result and result[0]:
        recommended_courses = json.loads(result[0])

    # Fetch scores
    cursor.execute("""
        SELECT verbal_language, reading_comprehension, english, math, 
               non_verbal, basic_computer 
        FROM students ORDER BY id DESC LIMIT 1
    """)
    scores_result = cursor.fetchone()
    student_scores = list(scores_result) if scores_result else [0] * 6

    # Ensure student_scores is a standard list
    student_scores = np.array(student_scores).tolist()  # Convert to list

    # Load dataset from Excel for comparison
    dataset = pd.read_excel('dataset.xlsx', sheet_name=None)
    all_data = pd.concat(dataset.values(), ignore_index=True)

    # Filter subjects for comparison (ensure matching column names)
    subjects = ['Verbal Language', 'Reading Comprehension', 'English', 'Math', 
                'Non Verbal', 'Basic Computer']
    available_data = all_data[all_data.columns.intersection(subjects)].dropna().astype(float)

    # Calculate the average scores from the dataset
    avg_scores = available_data.mean().values if not available_data.empty else [0] * len(subjects)
    avg_scores = avg_scores.tolist()  # Ensure avg_scores is a standard list

    # Generate Bar Chart
    fig, ax = plt.subplots()
    labels = ['Verbal Lang', 'Reading Comp', 'English', 'Math', 'Non Verbal', 'Basic Comp']
    x = np.arange(len(labels))
    width = 0.35

    ax.bar(x - width / 2, student_scores, width, label='User', alpha=0.7)
    ax.bar(x + width / 2, avg_scores, width, label='Dataset Avg', alpha=0.7)

    ax.set_xlabel('Subjects')
    ax.set_ylabel('Scores')
    ax.set_title('Comparison of User Scores with Dataset Average')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # Save Bar Chart to Buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    chart_url = base64.b64encode(buf.getvalue()).decode('utf8')
    buf.close()

    # Generate Radar Chart
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    num_vars = len(subjects)

    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    student_scores = np.concatenate((student_scores, [student_scores[0]]))
    avg_scores = np.concatenate((avg_scores, [avg_scores[0]]))
    angles += angles[:1]

    # Draw the radar chart
    ax.fill(angles, student_scores, color='red', alpha=0.25, label='User Scores')
    ax.fill(angles, avg_scores, color='blue', alpha=0.25, label='Average Scores')

    # Draw one axe per variable and add labels
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

    # Save Radar Chart to Buffer
    radar_buf = io.BytesIO()
    plt.savefig(radar_buf, format='png')
    radar_buf.seek(0)
    radar_chart_url = base64.b64encode(radar_buf.getvalue()).decode('utf8')
    radar_buf.close()

    return render_template('results.html', 
                           recommended_courses=recommended_courses, 
                           student_scores=student_scores,
                           avg_scores=avg_scores,
                           chart_url=chart_url, 
                           radar_chart_url=radar_chart_url)

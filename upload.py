from flask import Flask, request, render_template, redirect, flash
import pandas as pd
import os

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Required for flash messages

MAIN_FILE = 'data/dataset.xlsx'

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            flash("No file part", "error")
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash("No selected file", "error")
            return redirect(request.url)

        try:
            # Load the uploaded Excel file into a DataFrame
            uploaded_df = pd.read_excel(file)

            # Read all existing sheets from the main Excel file
            existing_sheets = pd.read_excel(MAIN_FILE, sheet_name=None)

            # Find the most recent sheet (assumes the sheet names are years or sortable)
            latest_sheet_name = max(existing_sheets.keys())

            # Load the data from the latest sheet
            main_df = existing_sheets[latest_sheet_name]

            # Append the new data to the main DataFrame
            updated_df = pd.concat([main_df, uploaded_df], ignore_index=True)

            # Update the latest sheet and preserve all other sheets
            existing_sheets[latest_sheet_name] = updated_df

            # Save all sheets back to the main Excel file
            with pd.ExcelWriter(MAIN_FILE, mode='w', engine='openpyxl') as writer:
                for sheet_name, df in existing_sheets.items():
                    df.to_excel(writer, sheet_name=sheet_name, index=False)

            flash(f"Excel file updated successfully on sheet '{latest_sheet_name}'!", "success")
        except Exception as e:
            flash(f"Error processing Excel files: {e}", "error")
            return redirect(request.url)

        return redirect(request.url)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)


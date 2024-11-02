from flask import Flask, render_template, request, redirect, url_for, send_file, flash
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import os

app = Flask(__name__)
app.secret_key = 'some_secret_key'

# Global variables to store data and model parameters
data = None
preprocessed_data = None
evaluation = {}

@app.route('/')
def home():
    return render_template("base.html", title="AdaBoost Workflow")

@app.route('/upload', methods=['GET', 'POST'])
def upload_data():
    global data
    if request.method == 'POST':
        file = request.files['file']
        if file and (file.filename.endswith('.csv') or file.filename.endswith('.xlsx')):
            data = pd.read_csv(file) if file.filename.endswith('.csv') else pd.read_excel(file)
            return redirect(url_for('show_uploaded_data'))
        else:
            flash("Please upload a valid CSV or Excel file.", "danger")
    return render_template("upload.html", title="Upload Data")

@app.route('/show_data', methods=['GET'])
def show_uploaded_data():
    global data
    if data is None:
        flash("No data uploaded. Please upload data first.", "warning")
        return redirect(url_for('upload_data'))

    # Show the first 5 rows of the uploaded data
    first_five_rows = data.head()
    return render_template("show_data.html", title="Uploaded Data", first_five_rows=first_five_rows)

@app.route('/preprocess', methods=['GET', 'POST'])
def preprocess_data():
    global data, preprocessed_data
    if data is None:
        flash("No data uploaded. Please upload data first.", "warning")
        return redirect(url_for('upload_data'))

    if request.method == 'POST':
        columns_to_remove = request.form.getlist('columns_to_remove')
        missing_strategy = request.form.get('missing_strategy')
        scaling_method = request.form.get('scaling_method')

        if columns_to_remove:
            data.drop(columns=columns_to_remove, inplace=True)

        data.replace("?", pd.NA, inplace=True)
        data[:] = data.apply(pd.to_numeric, errors='coerce')

        if missing_strategy in ["mean", "median"]:
            imputer = SimpleImputer(strategy=missing_strategy)
            data[:] = imputer.fit_transform(data)

        if scaling_method == "standard":
            scaler = StandardScaler()
        elif scaling_method == "minmax":
            scaler = MinMaxScaler()
        else:
            scaler = None
        
        if scaler:
            data[:] = scaler.fit_transform(data)

        preprocessed_data = data
        return redirect(url_for('visualize_data'))

    return render_template("preprocess.html", title="Preprocess Data", columns=data.columns)

@app.route('/visualize', methods=['GET', 'POST'])
def visualize_data():
    global preprocessed_data
    if preprocessed_data is None:
        flash("Please complete preprocessing before visualization.", "warning")
        return redirect(url_for('preprocess_data'))
    
    plot_url = None
    columns = preprocessed_data.columns.tolist()
    
    if request.method == 'POST':
        plot_type = request.form.get('plot_type')
        x_column = request.form.get('x_column')
        y_column = request.form.get('y_column')

        plt.figure(figsize=(10, 6))

        if plot_type == 'scatter' and x_column and y_column:
            sns.scatterplot(data=preprocessed_data, x=x_column, y=y_column)
        elif plot_type == 'line' and x_column and y_column:
            sns.lineplot(data=preprocessed_data, x=x_column, y=y_column)
        elif plot_type == 'bar' and x_column and y_column:
            sns.barplot(data=preprocessed_data, x=x_column, y=y_column)
        elif plot_type == 'histogram' and x_column:
            sns.histplot(preprocessed_data[x_column], kde=True)
        else:
            flash("Please select valid options for the plot.", "danger")
            return redirect(url_for('visualize_data'))

        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()

        plt.close()  # Close the plot to free memory

        # Show the generated plot on the same page
        return render_template("visualize.html", title="Visualize Data", columns=columns, plot_url=plot_url)

    return render_template("visualize.html", title="Visualize Data", columns=columns, plot_url=plot_url)

@app.route('/train', methods=['GET', 'POST'])
def train_model():
    global preprocessed_data, evaluation
    if preprocessed_data is None:
        flash("Please complete preprocessing and visualization before training.", "warning")
        return redirect(url_for('visualize_data'))

    columns = preprocessed_data.columns.tolist()

    if request.method == 'POST':
        target_column = request.form.get('target_column')
        learning_rate = request.form.get('learning_rate')  # Get learning rate from the form
        n_estimators = request.form.get('n_estimators')  # Get number of estimators from the form

        # Ensure that learning_rate and n_estimators are not None
        if learning_rate is None or n_estimators is None:
            flash("Learning rate and number of estimators must be provided.", "danger")
            return redirect(url_for('train_model'))

        try:
            learning_rate = float(learning_rate)  # Convert to float
            n_estimators = int(n_estimators)  # Convert to int
        except ValueError:
            flash("Please provide valid numbers for learning rate and number of estimators.", "danger")
            return redirect(url_for('train_model'))

        # Check if target_column is selected
        if target_column is None:
            flash("Please select a target column.", "danger")
            return redirect(url_for('train_model'))

        X = preprocessed_data.drop(target_column, axis=1)
        y = preprocessed_data[target_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = AdaBoostClassifier(learning_rate=learning_rate, n_estimators=n_estimators)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        evaluation['accuracy'] = accuracy_score(y_test, predictions)
        evaluation['precision'] = precision_score(y_test, predictions, average='weighted')
        evaluation['recall'] = recall_score(y_test, predictions, average='weighted')

        return redirect(url_for('download_results'))

    return render_template("train.html", title="Train Model", columns=columns)

@app.route('/download', methods=['GET', 'POST'])
def download_results():
    global preprocessed_data, evaluation
    if request.method == 'POST':
        output_format = request.form.get('output_format')
        file_path = "processed_data." + ("csv" if output_format == "csv" else "xlsx")

        if output_format == "csv":
            preprocessed_data.to_csv(file_path, index=False)
        else:
            preprocessed_data.to_excel(file_path, index=False)

        return send_file(file_path, as_attachment=True)

    return render_template("download.html", title="Download Results", evaluation=evaluation)

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, render_template, request, redirect, url_for, send_file, flash
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay,roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import os
import numpy as np

app = Flask(__name__)
app.secret_key = 'some_secret_key'

# Global variables to store data and model parameters
data = None
preprocessed_data = None
evaluation = {}
model = None
X_test = None
y_test = None

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

        return render_template("visualize.html", title="Visualize Data", columns=columns, plot_url=plot_url)

    return render_template("visualize.html", title="Visualize Data", columns=columns, plot_url=plot_url)

@app.route('/train', methods=['GET', 'POST'])
def train_model():
    global preprocessed_data, evaluation, model, X_test, y_test , predictions
    if preprocessed_data is None:
        flash("Please complete preprocessing and visualization before training.", "warning")
        return redirect(url_for('visualize_data'))

    columns = preprocessed_data.columns.tolist()

    if request.method == 'POST':
        target_column = request.form.get('target_column')
        learning_rate = request.form.get('learning_rate')
        n_estimators = request.form.get('n_estimators')

        if learning_rate is None or n_estimators is None:
            flash("Learning rate and number of estimators must be provided.", "danger")
            return redirect(url_for('train_model'))

        try:
            learning_rate = float(learning_rate)
            n_estimators = int(n_estimators)
        except ValueError:
            flash("Please provide valid numbers for learning rate and number of estimators.", "danger")
            return redirect(url_for('train_model'))

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

        return redirect(url_for('post_training'))

    return render_template("train.html", title="Train Model", columns=columns)

@app.route('/post_training', methods=['GET', 'POST'])
def post_training():
    global y_test, predictions, model, X_test

    if y_test is None or predictions is None:
        flash("No evaluation data available. Train the model first.", "warning")
        return redirect(url_for('train_model'))

    if request.method == 'POST':
        pos_label = request.form.get('pos_label')  # Get the selected positive label
        if pos_label is None or pos_label == "":
            flash("Please select a positive label.", "danger")
            return redirect(url_for('post_training'))

        pos_label = int(pos_label)

        try:
            # For binary classification or specific class in multiclass
            y_pred_prob = model.predict_proba(X_test)
            if len(y_test.unique()) == 2:  # Binary classification
                fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob[:, 1], pos_label=pos_label)
                auc_score = roc_auc_score(y_test, y_pred_prob[:, 1])
            else:  # Multiclass case
                fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob[:, pos_label], pos_label=pos_label)
                auc_score = roc_auc_score(y_test, y_pred_prob, multi_class="ovr")

            # Plot ROC Curve
            plt.figure(figsize=(10, 6))
            plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.2f})")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve")
            plt.legend()
            roc_img = io.BytesIO()
            plt.savefig(roc_img, format='png')
            roc_img.seek(0)
            roc_url = base64.b64encode(roc_img.getvalue()).decode()
            plt.close()

            # Confusion Matrix
            cm_display = ConfusionMatrixDisplay.from_predictions(y_test, predictions)
            cm_img = io.BytesIO()
            plt.savefig(cm_img, format='png')
            cm_img.seek(0)
            cm_url = base64.b64encode(cm_img.getvalue()).decode()
            plt.close()

            return render_template(
                "post_training.html",
                title="Post-Training Analysis",
                cm_url=cm_url,
                roc_url=roc_url,
                unique_values=y_test.unique(),
                selected_pos_label=pos_label
            )

        except Exception as e:
            flash(f"Error generating graphs: {str(e)}", "danger")
            return redirect(url_for('post_training'))

    # For GET request, just show the form
    return render_template(
        "post_training.html",
        title="Post-Training Analysis",
        unique_values=y_test.unique(),
        selected_pos_label=None
    )


if __name__ == '__main__':
    app.run(debug=True)

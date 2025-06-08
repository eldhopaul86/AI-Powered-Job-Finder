from flask import Flask, render_template, request, url_for
import json
import pandas as pd
import fitz  # PyMuPDF for reading PDFs
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Ensure stopwords are available
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

# Function to extract text from a PDF resume
def extract_resume_text(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = "\n".join([page.get_text() for page in doc])
        return text.strip()
    except Exception as e:
        return f"Error reading PDF: {e}"

# Load job data
def load_jobs(json_file):
    try:
        with open(json_file, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return pd.DataFrame(data)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

# Preprocess job descriptions
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    return text

# Check if text is a valid resume
def is_valid_resume(text):
    resume_keywords = ["experience", "skills", "education", "job", "developer", "engineer", "manager"]
    return any(word in text.lower() for word in resume_keywords)

# Recommend jobs based on resume
def recommend_jobs(resume_text, job_df, top_n=10, min_sim_threshold=0.15):
    if job_df is None or job_df.empty or 'job_description' not in job_df.columns:
        return [], "Error: No job data available."

    job_df['processed_description'] = job_df['job_description'].apply(preprocess_text)
    
    processed_resume = preprocess_text(resume_text)
    if len(processed_resume.split()) < 10:
        return [], "Uploaded document is too short to be a resume."

    if not is_valid_resume(processed_resume):
        return [], "Uploaded document is not a valid resume."

    corpus = job_df['processed_description'].tolist()
    corpus.append(processed_resume)

    try:
        stop_words = stopwords.words('english')
    except LookupError:
        nltk.download('stopwords')
        stop_words = stopwords.words('english')

    vectorizer = TfidfVectorizer(stop_words=stop_words)
    tfidf_matrix = vectorizer.fit_transform(corpus)

    similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    job_df['similarity_score'] = similarity_scores[0]

    recommended_jobs = job_df[job_df['similarity_score'] >= min_sim_threshold].sort_values(by='similarity_score', ascending=False).head(top_n)

    if recommended_jobs.empty:
        return [], "No relevant jobs found."

    title_col = next((col for col in ['job_title', 'Job.Title', 'title', 'Title'] if col in recommended_jobs.columns), None)
    company_col = next((col for col in ['company_name', 'Company.Name', 'company', 'Company'] if col in recommended_jobs.columns), None)

    if title_col and company_col:
        return recommended_jobs[[title_col, company_col, 'similarity_score', 'job_description']].to_dict(orient='records'), None
    elif title_col:
        return recommended_jobs[[title_col, 'similarity_score', 'job_description']].to_dict(orient='records'), None
    else:
        return recommended_jobs[['similarity_score', 'job_description']].to_dict(orient='records'), None

# Flask Routes
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/upload", methods=["GET", "POST"])
def upload_resume():
    recommendations = []
    message = None

    if request.method == "POST":
        if "resume" not in request.files:
            message = "No file uploaded."
        else:
            file = request.files["resume"]
            if file.filename.endswith(".pdf"):
                pdf_path = "./uploaded_resume.pdf"
                file.save(pdf_path)

                job_df = load_jobs("filtered_jobs.json")
                resume_text = extract_resume_text(pdf_path)

                if resume_text:
                    recommendations, message = recommend_jobs(resume_text, job_df)
                else:
                    message = "Error reading resume file."

            else:
                message = "Only PDF files are allowed."

    return render_template("upload.html", recommendations=recommendations, message=message)

@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    app.run(debug=True)
from flask import Flask, request, render_template
import os
import docx2txt
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Flask App Setup ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')

app = Flask(__name__, template_folder=TEMPLATE_DIR)
app.config['UPLOAD_FOLDER'] = UPLOAD_DIR

print("Template folder:", app.template_folder)
print("Upload folder:", app.config['UPLOAD_FOLDER'])

# --- Ensure upload folder exists ---
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


# --- Helper Functions ---
def extract_text_from_pdf(file_path):
    text = ""
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                if page.extract_text():
                    text += page.extract_text()
    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")
    return text


def extract_text_from_docx(file_path):
    try:
        return docx2txt.process(file_path)
    except Exception as e:
        print(f"Error reading DOCX {file_path}: {e}")
        return ""


def extract_text_from_txt(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Error reading TXT {file_path}: {e}")
        return ""


def extract_text(file_path):
    if file_path.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith('.docx'):
        return extract_text_from_docx(file_path)
    elif file_path.endswith('.txt'):
        return extract_text_from_txt(file_path)
    else:
        print(f"Unsupported file format: {file_path}")
        return ""


# --- Routes ---
@app.route('/')
def home():
    return render_template('recruiter.html')


@app.route('/candidate')
def candidate_page():
    return render_template('candidate.html')


@app.route('/upload', methods=['POST'])
def upload_resumes():
    resume_files = request.files.getlist('resumes')
    if not resume_files:
        return render_template('candidate.html', message="⚠️ Please upload at least one resume.")

    for resume_file in resume_files:
        if resume_file and resume_file.filename:
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], resume_file.filename)
            resume_file.save(save_path)
            print(f"✅ Saved: {resume_file.filename}")

    return render_template('candidate.html', message="✅ Resumes uploaded successfully!")


@app.route('/recruiter')
def recruiter_page():
    return render_template('recruiter.html')


@app.route('/matcher', methods=['POST'])
def matcher():
    job_description = request.form.get('job_description', '').strip()

    # --- If no job description entered ---
    if not job_description:
        return render_template('recruiter.html', message="⚠️ Please enter a job description.")

    # --- Fetch all uploaded resumes from uploads folder ---
    resume_files = [
        f for f in os.listdir(app.config['UPLOAD_FOLDER'])
        if f.lower().endswith(('.pdf', '.docx', '.txt'))
    ]

    if not resume_files:
        return render_template('recruiter.html', message="⚠️ No resumes uploaded yet.")

    resumes = []
    filenames = []

    # --- Extract text from uploaded resumes ---
    for filename in resume_files:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        text = extract_text(file_path)
        if text.strip():
            resumes.append(text)
            filenames.append(filename)

    if not resumes:
        return render_template('recruiter.html', message="❌ No readable resumes found.")

    # --- Matching Logic (exact same as your version) ---
    vectorizer = TfidfVectorizer().fit_transform([job_description] + resumes)
    vectors = vectorizer.toarray()
    job_vector = vectors[0]
    resume_vectors = vectors[1:]

    similarities = cosine_similarity([job_vector], resume_vectors)[0]
    # --- Top 5 results ---
    top_indices = similarities.argsort()[-5:][::-1]
    top_resumes = [filenames[i] for i in top_indices]
    similarity_scores = [round(similarities[i] * 100, 2) for i in top_indices]

    # Combine results for template
    results = [
        {"filename": top_resumes[i], "score": similarity_scores[i]}
        for i in range(len(top_resumes))
    ]

    # Render template
    return render_template(
        'recruiter.html',
        message="✅ Top matching resumes:",
        results=results
    )


# --- Run App ---
if __name__ == '__main__':
    app.run(debug=True)

# main.py
# Main application file for Apex Recruiter.
# This file initializes the Eel application and exposes Python functions to the JavaScript frontend.

import eel
import os
import base64
from io import BytesIO
import json

# --- Python Standard Libraries ---
import re
from datetime import datetime
from collections import defaultdict, Counter

# --- Third-party libraries ---
# To install required libraries:
# pip install eel PyPDF2 python-docx
try:
    import PyPDF2
    import docx
except ImportError:
    print("Please install required libraries: pip install eel PyPDF2 python-docx")
    exit()

# --- Custom Algorithm Modules ---
# In a larger application, these would be separate files. For this self-contained example, they are included here.

# =======================================================================================
# ALGORITHM MODULE 1: RESUME PARSING & TEXT EXTRACTION
# =======================================================================================

def extract_text_from_pdf(file_content):
    """
    Extracts text from the content of an uploaded PDF file.
    Handles potential extraction errors gracefully.
    """
    try:
        pdf_reader = PyPDF2.PdfReader(BytesIO(file_content))
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return "Error: Could not extract text from PDF file."

def extract_text_from_docx(file_content):
    """
    Extracts text from the content of an uploaded DOCX file.
    """
    try:
        document = docx.Document(BytesIO(file_content))
        text = "\n".join([para.text for para in document.paragraphs])
        return text
    except Exception as e:
        print(f"Error reading DOCX: {e}")
        return "Error: Could not extract text from DOCX file."

def extract_text_from_txt(file_content):
    """
    Extracts text from the content of an uploaded TXT file.
    """
    try:
        return file_content.decode('utf-8')
    except Exception as e:
        print(f"Error reading TXT: {e}")
        return "Error: Could not read text file."

def parse_resume(filename, file_content):
    """
    Orchestrator function to parse resume based on file extension.
    """
    ext = filename.split('.')[-1].lower()
    if ext == 'pdf':
        return extract_text_from_pdf(file_content)
    elif ext == 'docx':
        return extract_text_from_docx(file_content)
    elif ext == 'txt':
        return extract_text_from_txt(file_content)
    else:
        return "Error: Unsupported file format. Please use PDF, DOCX, or TXT."

# =======================================================================================
# ALGORITHM MODULE 2: KEYWORD MATCH SCORING
# =======================================================================================

def get_synonyms():
    """
    Defines a simple, hardcoded dictionary of synonyms for technical terms.
    In a real-world app, this could come from a file or a more extensive database.
    """
    return {
        "javascript": ["js", "es6", "react", "vue", "angular", "node.js"],
        "python": ["py", "django", "flask", "pandas", "numpy"],
        "aws": ["amazon web services", "ec2", "s3", "lambda"],
        "project manager": ["pm", "product manager", "scrum master"],
        "lead": ["leader", "senior", "manager", "principal"],
        "develop": ["engineer", "code", "program", "build"],
    }

def analyze_keyword_context(resume_text, keywords):
    """
    Custom Algorithm: Context-Aware Keyword Scoring

    Design:
    1.  **Section Identification**: The algorithm first splits the resume into sections using common headers
        (e.g., "Experience", "Skills", "Education"). This is done with a regular expression that looks for
        short lines (potential headers) followed by a newline.
    2.  **Contextual Weighting**: It defines weights for different sections. Keywords found under "Skills" or
        "Experience" are considered more valuable than those in a general "Summary" or "Education" section.
    3.  **Keyword & Synonym Matching**: For each section, it iterates through the user-provided keywords.
        It checks for the primary keyword and its predefined synonyms.
    4.  **Scoring**:
        - A base score is awarded for each match.
        - This score is then multiplied by the section's weight.
        - The algorithm keeps a detailed log of which keyword was found, where (section), and what score it received.
    5.  **Normalization**: The total raw score is calculated. The maximum possible score is also calculated (assuming
        every keyword is found in the highest-weighted section). The final score is `(raw_score / max_score) * 100`.

    Edge Cases Handled:
    - Resumes with no clear section headers are treated as a single "Body" section with a neutral weight.
    - Keywords are case-insensitive to ensure matches aren't missed.
    - Overlapping synonyms (e.g., "React" for "JavaScript") are counted towards the main keyword's score.
    """
    resume_text_lower = resume_text.lower()
    keywords = [k.lower().strip() for k in keywords.split(',') if k.strip()]
    synonyms = get_synonyms()

    # Create a reverse map from synonym to main keyword
    synonym_map = {}
    for main_word, syn_list in synonyms.items():
        for syn in syn_list:
            synonym_map[syn] = main_word

    # 1. Section Identification
    # A simple regex to find lines that look like headers (short, often all caps, followed by newline)
    sections = re.split(r'\n([A-Z][A-Z\s]{5,25}|Experience|Skills|Education|Projects|Certifications)\n', resume_text, flags=re.IGNORECASE)
    
    structured_text = {"header": sections[0]} # Text before the first header
    if len(sections) > 1:
        for i in range(1, len(sections), 2):
            header = sections[i].strip().lower()
            content = sections[i+1]
            structured_text[header] = content

    # 2. Contextual Weighting
    context_weights = {
        "skills": 1.5,
        "experience": 1.4,
        "work experience": 1.4,
        "professional experience": 1.4,
        "projects": 1.3,
        "summary": 1.0,
        "objective": 1.0,
        "education": 0.8,
        "certifications": 1.2,
        "header": 1.0, # Default for text before first header
        "body": 1.0 # Default for resumes with no structure
    }

    # 3 & 4. Scoring Logic
    score_details = defaultdict(lambda: {"count": 0, "score": 0, "locations": []})
    total_score = 0
    
    # If no sections found, treat the whole resume as one block
    if len(structured_text) == 1:
        structured_text = {"body": resume_text}

    for section_header, section_content in structured_text.items():
        section_content_lower = section_content.lower()
        # Find the most relevant weight key
        weight = context_weights["body"]
        for key in context_weights:
            if key in section_header:
                weight = context_weights[key]
                break

        for keyword in keywords:
            # Check for main keyword and its synonyms
            search_terms = [keyword] + synonyms.get(keyword, [])
            
            for term in search_terms:
                if f" {term} " in f" {section_content_lower} " or f"\n{term}\n" in f" {section_content_lower} ":
                    match_count = section_content_lower.count(term)
                    current_score = match_count * weight
                    total_score += current_score
                    
                    # Attribute score to the main keyword
                    main_keyword = synonym_map.get(term, keyword)
                    score_details[main_keyword]["count"] += match_count
                    score_details[main_keyword]["score"] += current_score
                    score_details[main_keyword]["locations"].append(section_header.title())

    # 5. Normalization
    # Max possible score assumes each keyword is found once in the highest-weighted section
    max_possible_score = len(keywords) * max(context_weights.values())
    normalized_score = min(100, (total_score / max_possible_score) * 100) if max_possible_score > 0 else 0

    # Format details for display
    detailed_breakdown = []
    for kw, data in score_details.items():
        detailed_breakdown.append({
            "keyword": kw.title(),
            "count": data['count'],
            "score": round(data['score'], 2),
            "locations": list(set(data['locations'])) # Unique locations
        })

    return {
        "total_score": round(normalized_score),
        "breakdown": detailed_breakdown
    }


@eel.expose
def process_resume_for_scoring(file_info, keywords):
    """
    Eel-exposed function to handle single resume scoring.
    """
    filename = file_info['name']
    file_b64 = file_info['data']
    file_content = base64.b64decode(file_b64)

    resume_text = parse_resume(filename, file_content)
    if resume_text.startswith("Error:"):
        return {"error": resume_text}

    if not keywords:
        return {"error": "Please enter at least one keyword."}

    scoring_result = analyze_keyword_context(resume_text, keywords)
    return scoring_result

# =======================================================================================
# ALGORITHM MODULE 3: RESUME COMPARISON & RANKING
# =======================================================================================

def analyze_additional_factors(resume_text):
    """
    Custom Algorithm: Multi-Criteria Analysis for Ranking

    Design:
    This algorithm extracts quantitative and qualitative metrics beyond simple keyword matching
    to provide a more holistic view of a candidate's profile.

    1.  **Experience Length Estimation**:
        - It searches for year patterns (e.g., "2018 - 2022", "Jan 2020 - Present").
        - It extracts all 4-digit numbers that look like years (between 1980 and current year).
        - The total experience is estimated by `(max_year - min_year)`. This is a heuristic and
          works well for chronological resumes but is less accurate for others. It's a trade-off
          for not having complex date parsing.
        - "Present" is converted to the current year to calculate ongoing roles.
    2.  **Skill Diversity**:
        - It uses a predefined list of common technical and soft skills.
        - It counts how many unique skills from this list are present in the resume. This rewards
          candidates with a broader skillset.
    3.  **Certification Detection**:
        - It looks for a "Certifications" section header or keywords like "certified", "certification".
        - This is a binary check (yes/no) that adds a bonus to the ranking score.

    Edge Cases Handled:
    - Resumes with no dates will return 0 years of experience.
    - The skill list is not exhaustive but covers a wide range of common areas.
    """
    text = resume_text.lower()
    
    # 1. Experience Length
    years = re.findall(r'\b(19[8-9]\d|20[0-2]\d)\b', text)
    # Handle "Present"
    if "present" in text or "current" in text:
        years.append(str(datetime.now().year))
    
    numeric_years = [int(y) for y in years]
    experience_years = 0
    if len(numeric_years) > 1:
        experience_years = max(numeric_years) - min(numeric_years)
        if experience_years > 50: experience_years = 0 # Sanity check for weird numbers

    # 2. Skill Diversity
    skill_list = [
        'python', 'java', 'c++', 'javascript', 'sql', 'nosql', 'react', 'angular', 'vue', 'django', 'flask',
        'spring', 'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform', 'git', 'jira', 'agile',
        'scrum', 'leadership', 'management', 'communication', 'teamwork', 'problem solving'
    ]
    found_skills = {skill for skill in skill_list if skill in text}
    skill_diversity_score = len(found_skills)

    # 3. Certification Detection
    has_certifications = False
    if re.search(r'certifications?|certified', text, re.IGNORECASE):
        has_certifications = True

    return {
        "experience_years": experience_years,
        "skill_diversity": skill_diversity_score,
        "has_certifications": has_certifications
    }


@eel.expose
def rank_resumes(file_list, keywords):
    """
    Eel-exposed function to handle comparison and ranking of multiple resumes.

    Custom Algorithm: Weighted Multi-Factor Ranking

    Design:
    This is the core ranking engine. It combines the outputs of the Keyword Scorer and the
    Additional Factors analyzer into a single, final rank.

    1.  **Data Collection**: For each resume, it performs:
        - Text extraction.
        - Keyword match scoring (using `analyze_keyword_context`).
        - Additional factors analysis (using `analyze_additional_factors`).
    2.  **Score Combination & Weighting**:
        - It defines weights for each of the four core metrics:
            - `keyword_score`: The most important factor (weight: 0.5).
            - `experience_years`: Important, but less so than direct skill match (weight: 0.25).
            - `skill_diversity`: Shows breadth of knowledge (weight: 0.15).
            - `certification_bonus`: A small bonus for formal qualifications (weight: 0.10).
        - The final score is a weighted sum:
          `Final = (kw * w1) + (exp * w2) + (skills * w3) + (cert * w4)`
    3.  **Normalization & Ranking**:
        - To make scores comparable, experience and skill diversity are normalized on a 0-100 scale
          based on the max value found across all candidates in the current batch.
        - Candidates are then sorted in descending order based on their final weighted score.
    4.  **Rationale Generation**: For transparency, the algorithm generates a plain-English rationale
        for each candidate's rank, explaining how their keyword match, experience, and skills
        contributed to their position.
    5.  **Shortlist Suggestion**: A simple heuristic is applied: the top 25% of candidates (or at least
        the top candidate) are suggested for a shortlist.
    """
    if not file_list:
        return {"error": "Please upload at least one resume."}
    if not keywords:
        return {"error": "Please enter keywords to rank against."}

    results = []
    for file_info in file_list:
        filename = file_info['name']
        file_b64 = file_info['data']
        file_content = base64.b64decode(file_b64)
        
        resume_text = parse_resume(filename, file_content)
        if resume_text.startswith("Error:"):
            results.append({"filename": filename, "error": resume_text})
            continue

        keyword_score_data = analyze_keyword_context(resume_text, keywords)
        additional_factors = analyze_additional_factors(resume_text)

        results.append({
            "filename": filename,
            "keyword_score": keyword_score_data['total_score'],
            **additional_factors
        })

    # Normalize experience and skill diversity scores (0-100)
    max_exp = max([r.get('experience_years', 0) for r in results] or [1])
    max_skills = max([r.get('skill_diversity', 0) for r in results] or [1])

    ranked_candidates = []
    for r in results:
        if "error" in r:
            continue
            
        norm_exp = (r['experience_years'] / max_exp) * 100 if max_exp > 0 else 0
        norm_skills = (r['skill_diversity'] / max_skills) * 100 if max_skills > 0 else 0
        cert_bonus = 20 if r['has_certifications'] else 0 # Give a flat 20 point bonus for certs

        # Weighted final score calculation
        # Weights: Keyword Match (50%), Experience (25%), Skills (15%), Certs (10%)
        final_score = (
            (r['keyword_score'] * 0.50) +
            (norm_exp * 0.25) +
            (norm_skills * 0.15) +
            (cert_bonus * 0.10)
        )
        
        rationale = f"Strong keyword match ({r['keyword_score']}%). "
        rationale += f"{r['experience_years']} years estimated experience. "
        rationale += f"Found {r['skill_diversity']} relevant skills. "
        if r['has_certifications']:
            rationale += "Certifications detected."

        ranked_candidates.append({
            "filename": r['filename'],
            "final_score": round(final_score),
            "keyword_match": r['keyword_score'],
            "experience": r['experience_years'],
            "skill_count": r['skill_diversity'],
            "certifications": "Yes" if r['has_certifications'] else "No",
            "rationale": rationale
        })

    # Sort by final score
    ranked_candidates.sort(key=lambda x: x['final_score'], reverse=True)

    # Suggest a shortlist
    num_to_shortlist = max(1, len(ranked_candidates) // 4)
    for i, candidate in enumerate(ranked_candidates):
        candidate['shortlist'] = "Yes" if i < num_to_shortlist else "No"


    return {"ranking": ranked_candidates}

# =======================================================================================
# ALGORITHM MODULE 4: JOB FIT Q&A
# =======================================================================================

@eel.expose
def process_qna(job_desc, resume_text, question):
    """
    Eel-exposed function for the rule-based Q&A chat.

    Custom Algorithm: Rule-Based Requirement Matching Engine

    Design:
    This algorithm avoids complex NLP and uses a system of targeted regular expressions and
    keyword matching to answer specific, predefined questions about a job fit.

    1.  **Requirement Extraction**:
        - It uses regex to find sentences or bullet points in the Job Description containing
          keywords like "required", "must have", "minimum", "desired", "plus".
        - It categorizes these into "hard_requirements" and "soft_requirements".
    2.  **Question Parsing**:
        - The user's question is normalized (lowercase, punctuation removed).
        - It checks for keywords in the question to map it to a specific "intent".
          - "meet requirements" -> `handle_met_requirements`
          - "miss requirements" -> `handle_missing_requirements`
          - "summarize experience" -> `handle_summarize_experience`
    3.  **Intent Handling**:
        - `handle_met_requirements`: Iterates through extracted requirements and checks if keywords
          from the requirement phrase exist in the resume.
        - `handle_missing_requirements`: The inverse of the above.
        - `handle_summarize_experience`: Searches for the specific skill mentioned in the question
          (e.g., "Summarize experience with Python"). It then tries to find paragraphs in the resume
          containing that skill and returns them as a summary.
    4.  **Default Response**: If the question doesn't match any known intent, it provides a helpful
        message listing the supported questions.

    Edge Cases Handled:
    - Poorly formatted job descriptions might not yield good requirements. The algorithm does its
      best with line-by-line analysis.
    - Vague questions will trigger the default response.
    """
    jd_lower = job_desc.lower()
    resume_lower = resume_text.lower()
    question_lower = question.lower()

    # 1. Requirement Extraction
    hard_reqs = re.findall(r'.*(?:required|must have|minimum|need|core)\s*[:\s].*', jd_lower)
    soft_reqs = re.findall(r'.*(?:desired|plus|nice to have|preferred)\s*[:\s].*', jd_lower)
    all_reqs = hard_reqs + soft_reqs
    
    # Simple keyword extraction from requirements
    def get_keywords_from_req(req):
        # A simple heuristic: find nouns and adjectives, avoid verbs/articles
        # This is basic and could be improved with a proper NLP library if allowed
        words = re.findall(r'\b[a-z][a-z-]+\b', req)
        # Filter out common noise words
        noise = {'a', 'an', 'the', 'in', 'of', 'for', 'with', 'and', 'or', 'to', 'is', 'are', 'be'}
        return [w for w in words if len(w) > 2 and w not in noise]


    # 2. & 3. Intent Handling
    if "meet" in question_lower and "requirement" in question_lower:
        met = []
        for req in all_reqs:
            keywords = get_keywords_from_req(req)
            if any(kw in resume_lower for kw in keywords):
                met.append(req.strip().replace('\n', ' '))
        if not met: return "No specific requirements from the job description appear to be met in the resume."
        return "The following requirements appear to be met:\n- " + "\n- ".join(met)

    elif "miss" in question_lower and "requirement" in question_lower:
        missing = []
        for req in all_reqs:
            keywords = get_keywords_from_req(req)
            if not any(kw in resume_lower for kw in keywords):
                missing.append(req.strip().replace('\n', ' '))
        if not missing: return "All identified requirements appear to be met in the resume."
        return "The following requirements appear to be missing or not explicitly mentioned:\n- " + "\n- ".join(missing)

    elif "summarize" in question_lower:
        # Find what to summarize, e.g., "Summarize experience with python"
        topic_match = re.search(r'with\s+([a-z0-9\s]+)', question_lower)
        if not topic_match:
            return "I can summarize experience for a specific skill. Please ask like: 'Summarize experience with Python'."
        
        topic = topic_match.group(1).strip()
        # Find paragraphs mentioning the topic
        sentences = re.split(r'[.\n]', resume_text)
        relevant_sentences = [s.strip() for s in sentences if topic in s.lower()]
        if not relevant_sentences:
            return f"No specific experience with '{topic}' was found in the resume."
        return f"Found the following sentences related to '{topic}':\n- " + "\n- ".join(relevant_sentences)

    # 4. Default Response
    else:
        return "I can answer questions like:\n- 'Which requirements are met?'\n- 'What requirements are missing?'\n- 'Summarize experience with [skill]'"

# =======================================================================================
# ALGORITHM MODULE 5: INTERVIEW SCHEDULER
# =======================================================================================

@eel.expose
def schedule_interviews(recruiter_slots_str, candidates_str):
    """
    Eel-exposed function to find optimal interview slots.

    Custom Algorithm: Greedy Best-Fit Scheduler

    Design:
    This algorithm aims to find a conflict-free schedule that accommodates the maximum number
    of candidates. It's a greedy algorithm, meaning it makes the locally optimal choice at each step.

    1.  **Input Parsing**:
        - It parses the input strings into structured data. Recruiter slots are a list of datetimes.
        - Candidate data is parsed into a list of objects, each with a name and a list of their
          preferred datetime slots.
    2.  **Candidate Prioritization**:
        - The core of the greedy strategy: candidates are sorted by the number of available slots
          they have, in *ascending* order. The candidate with the *fewest* options goes first.
        - This heuristic is effective because it tackles the most constrained candidates first,
          leaving more options open for those who are more flexible.
    3.  **Slot Assignment**:
        - The algorithm iterates through the prioritized list of candidates.
        - For each candidate, it checks their preferred slots one by one.
        - It looks for the first preferred slot that is also in the recruiter's availability list
          and has *not yet been assigned* to another candidate.
        - Once a match is found, that slot is assigned to the candidate, and the slot is removed
          from the pool of available times.
    4.  **Handling Unscheduled Candidates**:
        - If a candidate cannot be placed (none of their preferred slots are available), they are
          added to an "unscheduled" list.
    5.  **Output Generation**:
        - The final output is a dictionary containing the final schedule table and a list of any
          candidates who could not be scheduled, along with a reason.

    Edge Cases Handled:
    - Empty inputs for recruiter or candidates.
    - Time slots in mixed or invalid formats (try-except blocks handle parsing errors).
    - All candidates wanting the same slot. The prioritization ensures one gets it, and the rest
      are handled based on their other options or marked as unscheduled.
    """
    try:
        # 1. Input Parsing
        recruiter_slots = sorted([s.strip() for s in recruiter_slots_str.split('\n') if s.strip()])
        
        candidates_data = []
        for block in candidates_str.strip().split('\n\n'):
            lines = block.split('\n')
            name = lines[0].replace(':', '').strip()
            slots = sorted([s.strip() for s in lines[1:] if s.strip()])
            if name and slots:
                candidates_data.append({"name": name, "slots": slots})

        if not recruiter_slots or not candidates_data:
            return {"error": "Please provide both recruiter and candidate availability."}
            
    except Exception as e:
        return {"error": f"Error parsing input. Please check the format. Details: {e}"}

    # 2. Candidate Prioritization (most constrained first)
    candidates_data.sort(key=lambda c: len(c['slots']))

    # 3. Slot Assignment
    final_schedule = []
    unscheduled = []
    available_recruiter_slots = set(recruiter_slots)

    for candidate in candidates_data:
        assigned = False
        # Find a common, available slot
        for slot in candidate['slots']:
            if slot in available_recruiter_slots:
                final_schedule.append({"candidate": candidate['name'], "slot": slot})
                available_recruiter_slots.remove(slot) # Mark slot as taken
                assigned = True
                break
        
        if not assigned:
            unscheduled.append({
                "candidate": candidate['name'],
                "reason": "No overlap with available recruiter slots."
            })

    return {
        "schedule": sorted(final_schedule, key=lambda x: x['slot']),
        "unscheduled": unscheduled
    }


# --- EEL APPLICATION SETUP ---

def start_eel_app():
    """Initializes and starts the Eel application."""
    try:
        # Set the path to the 'web' directory
        web_dir = os.path.join(os.path.dirname(__file__), 'web')
        if not os.path.exists(web_dir):
            # Create a dummy web folder and index.html if it doesn't exist
            os.makedirs(web_dir)
            with open(os.path.join(web_dir, 'index.html'), 'w') as f:
                f.write('<h1>Error: index.html not found.</h1> <p>Please make sure the web folder with all required files is in the same directory as main.py</p>')
            print(f"Created dummy web directory at {web_dir}")

        eel.init(web_dir)
        
        # Start the app. This will open a browser window.
        # The size is chosen to be a common resolution.
        eel.start('index.html', size=(1280, 800), block=True)

    except (IOError, SystemExit):
        print("Eel app has closed.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    start_eel_app()

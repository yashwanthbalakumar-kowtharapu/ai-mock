import openai
import pyttsx3
import speech_recognition as sr
from flask import Flask, jsonify, request, render_template
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
from dotenv import load_dotenv
load_dotenv()
def analyze_technical_performance(candidate_answer, ideal_answer):
    """Analyze technical performance and provide feedback"""
    # Initialize feedback lists
    technical_strengths = []
    technical_weaknesses = []
    
    # Calculate answer relevance
    relevance_score = calculate_answer_relevance(candidate_answer, ideal_answer)
    
    # Analyze technical aspects
    if relevance_score > 0.8:
        technical_strengths.append("Strong understanding of core concepts")
        technical_strengths.append("Clear and accurate explanation")
    elif relevance_score > 0.6:
        technical_strengths.append("Good grasp of fundamentals")
        technical_weaknesses.append("Could provide more detailed explanations")
    else:
        technical_weaknesses.append("Need to improve technical knowledge")
        technical_weaknesses.append("Answer lacks key technical details")
    # Check for company and role specific knowledge
    if 'company' in request.args and 'role' in request.args:
        company = request.args.get('company')
        role = request.args.get('role')
        
        # Add company-specific analysis
        if company.lower() in candidate_answer.lower():
            technical_strengths.append(f"Shows knowledge of {company}")
        else:
            technical_weaknesses.append(f"Could demonstrate more {company}-specific knowledge")
            
        # Add role-specific analysis
        role_keywords = {
            'Frontend': ['html', 'css', 'javascript', 'react', 'angular', 'vue'],
            'Backend': ['api', 'database', 'server', 'nodejs', 'python', 'java'],
            'Full Stack': ['frontend', 'backend', 'database', 'api', 'fullstack'],
            'DevOps': ['ci/cd', 'docker', 'kubernetes', 'aws', 'cloud'],
            'Machine Learning': ['algorithms', 'neural networks', 'data science', 'tensorflow'],
            'Data Analytics': ['sql', 'python', 'data visualization', 'statistics'],
        }
        
        if role in role_keywords:
            role_specific_terms = set(role_keywords[role])
            candidate_terms_lower = {word.lower() for word in candidate_terms}
            matched_terms = role_specific_terms.intersection(candidate_terms_lower)
            
            if matched_terms:
                technical_strengths.append(f"Demonstrates {role}-specific knowledge")
            else:
                technical_weaknesses.append(f"Could include more {role}-specific terminology")

    # Check for specific keywords in ideal answer
    key_terms = set(ideal_answer.lower().split()) - {'a', 'an', 'the', 'in', 'on', 'at', 'to', 'for'}
    candidate_terms = set(candidate_answer.lower().split())
    
    # Analyze keyword coverage
    covered_terms = key_terms.intersection(candidate_terms)
    if len(covered_terms) / len(key_terms) > 0.7:
        technical_strengths.append("Good use of technical terminology")
    else:
        technical_weaknesses.append("Could use more precise technical terms")

    # Check answer length and structure
    if len(candidate_answer.split()) < len(ideal_answer.split()) * 0.5:
        technical_weaknesses.append("Answer could be more comprehensive")
    elif len(candidate_answer.split()) > len(ideal_answer.split()) * 1.5:
        technical_weaknesses.append("Try to be more concise")

    return {
        'technical_strengths': technical_strengths,
        'technical_weaknesses': technical_weaknesses,
        'relevance_score': relevance_score
    }


app = Flask(__name__)

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Initialize speech recognizer
recognizer = sr.Recognizer()

# Configure OpenAI
openai.api_key = 'AIzaSyDeIcREOvJzJcu6xuNFT6gmvweTminjAOc'
@app.route('/interview_window', methods=['GET'])
def interview_window():
    """API endpoint to generate and display interview questions based on company and role"""
    company = request.args.get('company')
    role = request.args.get('role')
    
    if not company or not role:
        return jsonify({"error": "Company and role must be provided"}), 400

    # Generate interview questions based on company and role
    questions = []
    for _ in range(5):  # Generate 5 questions
        question = generate_interview_question()
        questions.append(question)

    return jsonify({
        'company': company,
        'role': role,
        'questions': questions
    })

def generate_interview_question():
    """Generate interview question using OpenAI API"""
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt="Generate a professional interview question",
        max_tokens=100
    )
    return response.choices[0].text.strip()

def generate_ideal_answer(question):
    """Generate ideal answer using OpenAI API"""
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"Generate an ideal answer for the interview question: {question}",
        max_tokens=200
    )
    return response.choices[0].text.strip()

def speak_text(text):
    """Convert text to speech and play it"""
    engine.say(text)
    engine.runAndWait()

def record_audio():
    """Record audio from microphone and convert to text"""
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
        try:
            return recognizer.recognize_google(audio)
        except:
            return ""

def calculate_answer_relevance(candidate_answer, ideal_answer):
    """Calculate similarity between candidate's answer and ideal answer"""
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([candidate_answer, ideal_answer])
    similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    return float(similarity)

@app.route('/generate_question', methods=['GET'])
def get_interview_question():
    """API endpoint to get interview question"""
    question = generate_interview_question()
    ideal_answer = generate_ideal_answer(question)
    
    # Store ideal answer in session/cache for later comparison
    response = {
        'question': question,
        'ideal_answer': ideal_answer
    }
    return jsonify(response)

@app.route('/submit_answer', methods=['POST'])
def evaluate_answer():
    """API endpoint to evaluate candidate's answer"""
    data = request.json
    candidate_answer = data.get('answer', '')
    ideal_answer = data.get('ideal_answer', '')
    
    relevance_score = calculate_answer_relevance(candidate_answer, ideal_answer)
    
    return jsonify({
        'relevance_score': relevance_score,
        'feedback': get_feedback(relevance_score)
    })

def get_feedback(score):
    """Generate feedback based on relevance score"""
    if score >= 0.8:
        return "Excellent answer! Very relevant and comprehensive."
    elif score >= 0.6:
        return "Good answer, but there's room for improvement."
    else:
        return "Your answer could be more relevant to the question."

if __name__ == '__main__':
    app.run(debug=True)


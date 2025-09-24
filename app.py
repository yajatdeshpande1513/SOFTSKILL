from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

app = Flask(__name__)

# Define the key skills we'll be assessing
skill_list = [
    'Problem Solving', 'Teamwork', 'Technical Aptitude', 
    'Communication', 'Leadership', 'Creativity', 'Adaptability'
]

# Skill descriptions for the results page
skill_descriptions = {
    'Problem Solving': 'Your ability to analyze complex situations and find effective solutions to challenges.',
    'Teamwork': 'Your capacity to collaborate effectively with others and contribute to group efforts.',
    'Technical Aptitude': 'Your proficiency with technology and ability to learn and apply technical skills.',
    'Communication': 'Your skill in conveying ideas clearly and effectively to various audiences.',
    'Leadership': 'Your capability to guide, motivate, and influence others toward a common goal.',
    'Creativity': 'Your ability to generate innovative ideas and think outside conventional frameworks.',
    'Adaptability': 'Your flexibility in adjusting to new conditions and handling change effectively.'
}

# Enhanced scenarios with more variety (keeping your original scenarios)
scenarios = {
    "start": {
        "title": "Welcome to Professional Skills Assessment",
        "text": "This assessment will present you with 10 work scenarios. Your choices will help determine your strongest professional skills and aptitudes.",
        "options": [{"text": "Begin Assessment", "tooltip": "Start the skills assessment"}]
    },
    "scenario1": {
        "title": "Critical Production Bug",
        "text": "A major bug is reported in production that's affecting many users. The system is throwing errors and some features are completely broken. How do you respond?",
        "options": [
            {"text": "Quickly implement a hotfix without full testing to minimize downtime", "tooltip": "Focus on speed of resolution"},
            {"text": "Organize a team to thoroughly investigate root cause before fixing", "tooltip": "Collaborative problem-solving approach"},
            {"text": "Document the issue thoroughly and plan a fix for the next sprint", "tooltip": "Methodical, process-oriented approach"},
            {"text": "Reassign to a junior team member with oversight to help them learn", "tooltip": "Teaching and leadership approach"}
        ],
        "scores": {
            "problem_solving": [2, 4, 1, 3],
            "teamwork": [1, 4, 2, 3],
            "technical_aptitude": [3, 2, 1, 2],
            "leadership": [0, 1, 0, 4]
        }
    },
    "scenario2": {
        "title": "Tight Deadline Pressure",
        "text": "Your team is behind schedule on an important project with a hard deadline approaching in 3 days. The client is expecting delivery as promised. What's your approach?",
        "options": [
            {"text": "Work overtime yourself to meet the deadline", "tooltip": "Individual heroic effort"},
            {"text": "Re-negotiate timeline with stakeholders explaining the situation", "tooltip": "Communication and expectation management"},
            {"text": "Cut non-essential features to deliver core functionality on time", "tooltip": "Strategic prioritization"},
            {"text": "Organize the team to focus on critical paths and work efficiently", "tooltip": "Team coordination and leadership"}
        ],
        "scores": {
            "communication": [1, 4, 3, 2],
            "teamwork": [0, 3, 2, 4],
            "problem_solving": [1, 2, 3, 4],
            "leadership": [0, 2, 1, 3]
        }
    },
    "scenario3": {
        "title": "New Technology Proposal",
        "text": "You've identified a new technology that could significantly improve your team's productivity, but it would require time to learn and integrate. How do you proceed?",
        "options": [
            {"text": "Start using it immediately in a small project to demonstrate its value", "tooltip": "Action-oriented, learn by doing"},
            {"text": "Prepare a detailed presentation for management about costs and benefits", "tooltip": "Analytical and structured approach"},
            {"text": "Organize a lunch-and-learn session to introduce it to the whole team", "tooltip": "Collaborative learning approach"},
            {"text": "Research it thoroughly first and create documentation before proposing", "tooltip": "Methodical and careful approach"}
        ],
        "scores": {
            "technical_aptitude": [3, 2, 2, 4],
            "communication": [1, 4, 3, 2],
            "creativity": [4, 2, 3, 1],
            "adaptability": [4, 1, 3, 2]
        }
    },
    "scenario4": {
        "title": "Conflict in the Team",
        "text": "Two team members are in a heated disagreement about technical approach, causing tension in the team. How do you handle this?",
        "options": [
            {"text": "Let them work it out themselves - technical debates are healthy", "tooltip": "Hands-off approach"},
            {"text": "Facilitate a discussion to find common ground and compromise", "tooltip": "Mediation approach"},
            {"text": "Make an executive decision based on technical merits to resolve it", "tooltip": "Leadership decision"},
            {"text": "Suggest prototyping both approaches to compare results", "tooltip": "Data-driven approach"}
        ],
        "scores": {
            "teamwork": [1, 4, 2, 3],
            "leadership": [0, 3, 4, 2],
            "communication": [1, 4, 2, 3],
            "problem_solving": [0, 2, 1, 4]
        }
    },
    "scenario5": {
        "title": "Unclear Requirements",
        "text": "You're assigned to a project where the requirements are vague and keep changing. The business stakeholders seem unsure what they really want. What do you do?",
        "options": [
            {"text": "Build a flexible prototype to help visualize possibilities", "tooltip": "Interactive and creative approach"},
            {"text": "Schedule detailed meetings to pin down exact requirements", "tooltip": "Structured analytical approach"},
            {"text": "Make reasonable assumptions and document them clearly", "tooltip": "Practical, get-things-done approach"},
            {"text": "Suggest breaking the project into smaller, iterative phases", "tooltip": "Agile and adaptive approach"}
        ],
        "scores": {
            "communication": [3, 4, 2, 3],
            "creativity": [4, 1, 2, 3],
            "adaptability": [3, 1, 2, 4],
            "problem_solving": [3, 2, 4, 3]
        }
    },
    "scenario6": {
        "title": "Technical Debt Decision",
        "text": "You discover significant technical debt in a critical system. Fixing it would take 2 weeks but prevent future issues. Current workload is already heavy. What do you do?",
        "options": [
            {"text": "Fix it immediately - long-term stability is most important", "tooltip": "Quality-focused approach"},
            {"text": "Document it and plan to address it in the next sprint", "tooltip": "Process-oriented approach"},
            {"text": "Fix the worst parts now and schedule the rest for later", "tooltip": "Balanced approach"},
            {"text": "Present the business case to management for dedicated time", "tooltip": "Strategic communication approach"}
        ],
        "scores": {
            "problem_solving": [3, 1, 4, 2],
            "communication": [1, 2, 3, 4],
            "technical_aptitude": [4, 2, 3, 1],
            "leadership": [0, 1, 2, 4]
        }
    },
    "scenario7": {
        "title": "Learning New Skills",
        "text": "Your manager suggests you should expand your skill set to advance your career. How do you approach this?",
        "options": [
            {"text": "Dive deep into a new technical area that interests you", "tooltip": "Technical specialization"},
            {"text": "Take courses in leadership and project management", "tooltip": "Management track preparation"},
            {"text": "Learn a broad range of complementary skills", "tooltip": "Generalist approach"},
            {"text": "Focus on improving your existing core skills", "tooltip": "Mastery approach"}
        ],
        "scores": {
            "technical_aptitude": [4, 1, 3, 2],
            "leadership": [0, 4, 2, 1],
            "adaptability": [3, 2, 4, 1],
            "creativity": [2, 1, 3, 1]
        }
    },
    "scenario8": {
        "title": "Cross-Team Collaboration",
        "text": "You need to work with another team that has very different processes and tools. Their priorities don't align well with yours. How do you proceed?",
        "options": [
            {"text": "Propose a unified process that works for both teams", "tooltip": "Problem-solving approach"},
            {"text": "Focus on clear communication of requirements and expectations", "tooltip": "Communication-focused approach"},
            {"text": "Adapt to their processes to maintain harmony", "tooltip": "Adaptive approach"},
            {"text": "Escalate to management to align priorities", "tooltip": "Structural approach"}
        ],
        "scores": {
            "teamwork": [3, 4, 2, 1],
            "communication": [2, 4, 3, 1],
            "adaptability": [1, 2, 4, 0],
            "leadership": [4, 3, 1, 2]
        }
    },
    "scenario9": {
        "title": "Innovation Opportunity",
        "text": "You have an idea for an innovative feature that could significantly improve your product, but it would require substantial development effort. What do you do?",
        "options": [
            {"text": "Build a quick prototype to demonstrate the concept", "tooltip": "Action-oriented innovation"},
            {"text": "Prepare a detailed business case with ROI analysis", "tooltip": "Analytical approach"},
            {"text": "Discuss with colleagues to refine the idea first", "tooltip": "Collaborative approach"},
            {"text": "Add it to the product roadmap for future consideration", "tooltip": "Strategic planning approach"}
        ],
        "scores": {
            "creativity": [4, 2, 3, 1],
            "problem_solving": [3, 4, 2, 1],
            "communication": [1, 3, 4, 2],
            "technical_aptitude": [4, 1, 2, 0]
        }
    },
    "scenario10": {
        "title": "Work-Life Balance",
        "text": "You've been working long hours consistently to meet deadlines. Your personal life is suffering. How do you address this?",
        "options": [
            {"text": "Push through - career advancement requires sacrifice", "tooltip": "Career-focused approach"},
            {"text": "Have an open discussion with your manager about workload", "tooltip": "Communication approach"},
            {"text": "Improve personal efficiency to reduce hours needed", "tooltip": "Productivity focus"},
            {"text": "Set clear boundaries and prioritize ruthlessly", "tooltip": "Work-life balance approach"}
        ],
        "scores": {
            "communication": [1, 4, 2, 3],
            "adaptability": [0, 2, 3, 4],
            "leadership": [0, 3, 1, 4],
            "problem_solving": [0, 2, 4, 3]
        }
    }
}

@app.route('/')
def index():
    return render_template('scenarios.html')

@app.route('/get_scenario', methods=['POST'])
def get_scenario():
    scenario_id = request.json.get('current_scenario', 'start')
    return jsonify(scenarios.get(scenario_id, {}))

@app.route('/predict_career', methods=['POST'])
def predict_skills():
    user_scores = request.json['scores']
    
    # Replace tech_skills with technical_aptitude if found
    if 'tech_skills' in user_scores:
        user_scores['technical_aptitude'] = user_scores.pop('tech_skills')
    
    # Prepare all skill scores, defaulting to 0 if not provided
    skills_data = {
        'problem_solving': user_scores.get('problem_solving', 0),
        'teamwork': user_scores.get('teamwork', 0),
        'technical_aptitude': user_scores.get('technical_aptitude', 0),
        'communication': user_scores.get('communication', 0),
        'leadership': user_scores.get('leadership', 0),
        'creativity': user_scores.get('creativity', 0),
        'adaptability': user_scores.get('adaptability', 0)
    }
    
    # Normalize scores to a scale of 1-100
    max_possible_score = 40  # 10 scenarios with max 4 points each
    normalized_skills = {k: round((v / max_possible_score) * 100) for k, v in skills_data.items()}
    
    # Find the top 3 skills
    sorted_skills = sorted(normalized_skills.items(), key=lambda x: x[1], reverse=True)
    top_skills = sorted_skills[:3]
    
    # Add descriptions for the top skills
    top_skills_with_descriptions = [
        {
            'name': skill[0].replace('_', ' ').title(),
            'score': skill[1],
            'description': skill_descriptions.get(skill[0].replace('_', ' ').title(), 
                          f"Your proficiency in {skill[0].replace('_', ' ')}.")
        }
        for skill in top_skills
    ]
    
    # Calculate development areas (lowest scoring skills)
    development_areas = sorted_skills[-2:]
    development_areas_with_descriptions = [
        {
            'name': skill[0].replace('_', ' ').title(),
            'score': skill[1],
            'description': f"You may benefit from developing your {skill[0].replace('_', ' ')} abilities."
        }
        for skill in development_areas
    ]
    
    return jsonify({
        'primary_skill': top_skills_with_descriptions[0]['name'],
        'primary_skill_score': top_skills_with_descriptions[0]['score'],
        'primary_skill_description': top_skills_with_descriptions[0]['description'],
        'secondary_skills': [
            top_skills_with_descriptions[1],
            top_skills_with_descriptions[2]
        ],
        'all_skills': {k.replace('_', ' ').title(): v for k, v in normalized_skills.items()},
        'development_areas': development_areas_with_descriptions
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=os.getenv('PORT', 5000))

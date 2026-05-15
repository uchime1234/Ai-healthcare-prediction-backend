import pandas as pd
import joblib
from django.conf import settings
import os
import requests
import json
from django.contrib import auth
from django.contrib.auth.models import User
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.authtoken.models import Token
from .models import PredictionHistory, ChatMessage
import re
import time
import logging
import sklearn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load models
try:
    BODY_MODEL = joblib.load(os.path.join(settings.STATIC_ROOT, 'body_performance.joblib'))
    body_model = BODY_MODEL["model"]

    SLEEP_MODEL = joblib.load(os.path.join(settings.STATIC_ROOT, 'sleep_disorder.joblib'))
    sleep_model = SLEEP_MODEL["model"]

    HEART_MODEL = joblib.load(os.path.join(settings.STATIC_ROOT, 'heart_disease.joblib'))
    heart_model = HEART_MODEL["model"]

    ASTHMA_MODEL = joblib.load(os.path.join(settings.STATIC_ROOT, 'asthma_model.joblib'))
    asthma_model = ASTHMA_MODEL["model"]

    ALZHEIMER_MODEL = joblib.load(os.path.join(settings.STATIC_ROOT, 'alpzeimer_model.joblib'))
    alzheimer_model = ALZHEIMER_MODEL["model"]

    PARKINSON_MODEL = joblib.load(os.path.join(settings.STATIC_ROOT, 'parksion_model.joblib'))
    parkinson_model = PARKINSON_MODEL["model"]

    HYPERTENSION_MODEL = joblib.load(os.path.join(settings.STATIC_ROOT, 'hypertension_model.joblib'))
    hypertension_model = HYPERTENSION_MODEL["model"]

    DIABETES_MODEL = joblib.load(os.path.join(settings.STATIC_ROOT, 'diabetes_model.joblib'))
    diabetes_model = DIABETES_MODEL["model"]
except Exception as e:
    logger.error(f"Failed to load models: {str(e)}")
    raise

# Prediction label mappings
PREDICTION_LABELS = {
    'body': {1: "good shape", 2: "not bad or okay shape", 3: "be worried", 4: "u need to start taking care"},
    'sleep': {0: "none", 1: "insomnia", 2: "sleep apnea"},
    'heart': {0: "no presence", 1: "there is a presence"},
    'asthma': {0: "no asthma", 1: "asthma"},
    'alzheimer': {0: "no alzheimer", 1: "alzheimer"},
    'parkinson': {0: "no parkinson", 1: "high risk of parkinson"},
    'hypertension': {0: "no hypertension", 1: "high risk of hypertension"},
    'diabetes': {0: "no diabetes", 1: "diabetes"}
}

# Define required features for each prediction type
FEATURES_REQUIRED = {
    'body': "age, gender (male/female), height_cm, weight_kg, body fat_%, diastolic, systolic, gripForce, sit and bend forward_cm, sit-ups counts, broad jump_cm",
    'sleep': "gender (male/female), age, occupation, sleep duration, sleep quality, physical activity, stress level, bmi category (normal/overweight/obese), heart rate, daily steps, systolic bp, diastolic bp",
    'heart': "age, gender (male/female), height, weight, systolic bp, diastolic bp, cholesterol (normal/above normal/well above normal), glucose (normal/above normal/well above normal), smoke (yes/no), active (yes/no), alcohol (yes/no)",
    'asthma': "age, smoking_status (yes/no), air_pollution_exposure, exercise_frequency_per_week, family_history (yes/no), obesity (yes/no), symptom_wheezing (yes/no), symptom_shortness_breath (yes/no)",
    'alzheimer': "age, sex (male/female), education_years, memory_loss_score, language_problems_score, family_history (yes/no), depression_score, mri_atrophy_score",
    'parkinson': "age, sex (male/female), voice_pitch_variation, tremor_intensity, muscle_rigidity, slowness_of_movement, family_history (yes/no), speech_clarity_score",
    'hypertension': "age, bmi, systolic_blood_pressure, diastolic_blood_pressure, family_history (yes/no), physical_activity, smoking_status (yes/no), cholesterol_level, stress_level",
    'diabetes': "age, bmi, blood_glucose_level, hba1c, family_history (yes/no), physical_activity, smoking_status (yes/no), systolic_blood_pressure, cholesterol_level"
}

def call_grok_api(prompt, api_key):
    """Call Grok (xAI) API for generating health advice."""
    try:
        # Grok API endpoint (using xAI API)
        url = "https://api.groq.com/openai/v1/chat/completions"
        
        headers = {
              "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "llama-3.3-70b-versatile",  # or "grok-vision-beta" if needed
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful medical assistant providing concise, actionable health advice. Always include a disclaimer to consult healthcare professionals."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.7,
            "max_tokens": 250,
            "top_p": 0.9
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        response.raise_for_status()
        
        result = response.json()
        advice = result['choices'][0]['message']['content'].strip()
        
        return advice
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Grok API request failed: {str(e)}")
        if hasattr(e, 'response') and e.response:
            logger.error(f"Response status: {e.response.status_code}")
            logger.error(f"Response body: {e.response.text}")
        return None
    except KeyError as e:
        logger.error(f"Unexpected API response format: {str(e)}")
        return None

def generate_advice(input_data, prediction, prediction_type):
    """Generate tailored advice using Grok API."""
    default_advice = "Please consult a healthcare professional for personalized advice."
    
    # Get the API key from settings
    api_key = getattr(settings, 'XAI_API_KEY', None)
    
    # If no API key is configured, use rule-based advice
    if not api_key:
        logger.warning("XAI_API_KEY not found in settings. Using rule-based advice.")
        return generate_rule_based_advice(input_data, prediction, prediction_type)
    
    # Prepare prompt for Grok
    prediction_label = PREDICTION_LABELS[prediction_type].get(int(prediction), "Unknown")
    
    prompt = f"""Based on the following health assessment:

Condition: {prediction_type.upper()} Assessment
Result: {prediction_label} (Risk level: {prediction})
Patient Data: {json.dumps(input_data, indent=2)}

Please provide:
1. A brief explanation of what this result means
2. 2-3 actionable lifestyle recommendations
3. When to consult a healthcare professional

Keep response concise (under 150 words) and include a disclaimer."""
    
    # Try to get advice from Grok API
    try:
        advice = call_grok_api(prompt, api_key)
        if advice:
            return advice
        else:
            logger.warning("Grok API returned no advice, falling back to rule-based")
            return generate_rule_based_advice(input_data, prediction, prediction_type)
    except Exception as e:
        logger.error(f"Error in generate_advice: {str(e)}")
        return generate_rule_based_advice(input_data, prediction, prediction_type)

def generate_rule_based_advice(input_data, prediction, prediction_type):
    """Fallback rule-based advice when API is unavailable."""
    
    if prediction_type == 'body':
        if prediction == 1:  # good shape
            return "✅ Your body is in good shape! Continue your current exercise and diet routine. Remember to maintain regular check-ups."
        elif prediction == 2:  # not bad or okay shape
            return "⚠️ Your body is in okay shape. Consider adding 30 minutes of daily exercise and a balanced diet with more vegetables and lean proteins."
        elif prediction == 3:  # be worried
            return "🔴 You should be concerned about your body health. Please consult a doctor and start a structured fitness plan immediately."
        elif prediction == 4:  # u need to start taking care
            return "🚨 Urgent action needed! Seek medical advice within the next week and adopt a healthier lifestyle immediately. Consider working with a nutritionist and personal trainer."
    
    elif prediction_type == 'sleep':
        if prediction == 0:  # none
            return "✅ You have no sleep disorder. Maintain good sleep hygiene: stick to a schedule, avoid screens before bed, and keep your bedroom dark and cool."
        elif prediction == 1:  # insomnia
            return "⚠️ You may have insomnia. Try: maintaining a consistent sleep schedule, avoiding caffeine after 2 PM, and practicing relaxation techniques. Consider consulting a sleep specialist if symptoms persist."
        elif prediction == 2:  # sleep apnea
            return "🔴 You may have sleep apnea. This condition requires medical attention. Please consult a doctor for a sleep study. In the meantime, try sleeping on your side and avoiding alcohol before bed."

    elif prediction_type == 'heart':
        if prediction == 0:  # no presence
            return "✅ No heart disease detected. Maintain heart health with: regular exercise (150 mins/week), a Mediterranean diet, stress management, and avoiding smoking."
        else:
            return "🔴 Heart disease may be present. Schedule an appointment with a cardiologist immediately. In the meantime: reduce salt intake, avoid saturated fats, and start gentle walking if approved by your doctor."

    elif prediction_type == 'asthma':
        if prediction == 0:  # no asthma
            return "✅ No asthma detected. Monitor your respiratory health and avoid smoking and air pollution when possible."
        else:
            return "⚠️ Asthma may be present. See a doctor for proper diagnosis and treatment. Avoid known triggers like dust, pollen, and smoke. Keep an inhaler if prescribed."

    elif prediction_type == 'alzheimer':
        if prediction == 0:  # no alzheimer
            return "✅ No Alzheimer's detected. Keep your brain active with puzzles, learning new skills, social engagement, and regular physical exercise."
        else:
            return "🔴 Alzheimer's risk detected. Consult a neurologist for cognitive assessment. Brain-healthy habits include: Mediterranean diet, regular exercise, mental stimulation, and quality sleep."

    elif prediction_type == 'parkinson':
        if prediction == 0:  # no parkinson
            return "✅ No Parkinson's detected. Maintain a healthy lifestyle with regular exercise, especially activities that involve coordination and balance."
        else:
            return "🔴 High risk of Parkinson's detected. Please consult a neurologist for evaluation. Early intervention can help manage symptoms effectively."

    elif prediction_type == 'hypertension':
        if prediction == 0:  # no hypertension
            return "✅ No hypertension detected. Maintain healthy blood pressure with: low-salt diet, regular exercise, stress management, and limiting alcohol."
        else:
            return "⚠️ High risk of hypertension. Reduce salt intake to <2,300mg daily, exercise 30 minutes daily, maintain healthy weight, and limit alcohol. Monitor blood pressure regularly and consult a doctor."

    elif prediction_type == 'diabetes':
        if prediction == 0:  # no diabetes
            return "✅ No diabetes detected. Continue healthy eating, maintain healthy weight, exercise regularly, and limit sugary foods and drinks."
        else:
            return "🔴 Diabetes may be present. Monitor blood glucose levels, reduce carbohydrate intake, increase physical activity, and consult a doctor for proper diagnosis and management plan."
    
    return "Please consult a healthcare professional for personalized advice based on these results."

def extract_features(transcript, prediction_type):
    """Extract structured data from text input based on prediction type."""
    input_data = {}

    if prediction_type == 'body':
        patterns = {
            'age': r'age\s*(\d+)',
            'gender': r'gender\s*(male|female)',
            'height_cm': r'height_cm\s*(\d+\.?\d*)',
            'weight_kg': r'weight_kg\s*(\d+\.?\d*)',
            'body_fat_%': r'body_fat_%\s*(\d+\.?\d*)',
            'diastolic': r'diastolic\s*(\d+)',
            'systolic': r'systolic\s*(\d+)',
            'grip_force': r'grip_force\s*(\d+\.?\d*)',
            'sit_and_bend_forward_cm': r'sit_and_bend_forward_cm\s*(\d+\.?\d*)',
            'sit_ups_counts': r'sit_ups_counts\s*(\d+)',
            'broad_jump_cm': r'broad_jump_cm\s*(\d+\.?\d*)'
        }
        for key, pattern in patterns.items():
            match = re.search(pattern, transcript, re.IGNORECASE)
            if match:
                input_data[key] = match.group(1)
            else:
                return None, f"Missing {key}. Please provide: {FEATURES_REQUIRED['body']}. Example: 'predict body performance age 25 gender male height_cm 175 weight_kg 70 body_fat_% 15 diastolic 80 systolic 120 grip_force 35 sit_and_bend_forward_cm 20 sit_ups_counts 25 broad_jump_cm 180'"
        # Validate numerical inputs
        numerical_keys = ['age', 'height_cm', 'weight_kg', 'body_fat_%', 'diastolic', 'systolic', 'grip_force', 'sit_and_bend_forward_cm', 'sit_ups_counts', 'broad_jump_cm']
        for key in numerical_keys:
            try:
                float(input_data[key])
            except ValueError:
                return None, f"Invalid value for {key}. Please provide a numerical value."
        input_data['gender'] = 1 if input_data['gender'].lower() == 'male' else 0

    elif prediction_type == 'sleep':
        patterns = {
            'age': r'age\s*(\d+)',
            'gender': r'gender\s*(male|female)',
            'sleep_duration': r'sleep_duration\s*(\d+\.?\d*)',
            'quality_of_sleep': r'quality_of_sleep\s*(\d+)',
            'physical_activity_level': r'physical_activity_level\s*(\d+)',
            'stress_level': r'stress_level\s*(\d+)',
            'bmi_category': r'bmi_category\s*(normal|overweight|obese)',
            'blood_pressure': r'blood_pressure\s*(\d+/\d+)',
            'heart_rate': r'heart_rate\s*(\d+)',
            'daily_steps': r'daily_steps\s*(\d+)'
        }
        for key, pattern in patterns.items():
            match = re.search(pattern, transcript, re.IGNORECASE)
            if match:
                input_data[key] = match.group(1)
            else:
                return None, f"Missing {key}. Please provide: {FEATURES_REQUIRED['sleep']}"
        # Validate numerical inputs
        numerical_keys = ['age', 'sleep_duration', 'quality_of_sleep', 'physical_activity_level', 'stress_level', 'heart_rate', 'daily_steps']
        for key in numerical_keys:
            try:
                float(input_data[key])
            except ValueError:
                return None, f"Invalid value for {key}. Please provide a numerical value."
        input_data['gender'] = 1 if input_data['gender'].lower() == 'male' else 0
        try:
            input_data['bmi_category'] = {'normal': 0, 'overweight': 1, 'obese': 2}[input_data['bmi_category'].lower()]
        except KeyError:
            return None, f"Invalid value for bmi_category. Must be one of: normal, overweight, obese."
        try:
            bp = input_data['blood_pressure'].split('/')
            input_data['systolic'] = float(bp[0])
            input_data['diastolic'] = float(bp[1])
        except (ValueError, IndexError):
            return None, f"Invalid value for blood_pressure. Must be in the format 'systolic/diastolic' (e.g., '120/80')."

    elif prediction_type == 'heart':
        patterns = {
            'age': r'age\s*(\d+)',
            'gender': r'(?:gender|sex)\s*(male|female)',
            'height': r'height\s*(\d+\.?\d*)',
            'weight': r'weight\s*(\d+\.?\d*)',
            'systolic_bp': r'systolic\s*bp\s*(\d+)',
            'diastolic_bp': r'diastolic\s*bp\s*(\d+)',
            'cholesterol': r'cholesterol\s*(normal|above\s*normal|well\s*above\s*normal)',
            'glucose': r'glucose\s*(normal|above\s*normal|well\s*above\s*normal)',
            'smoke': r'smoke\s*(yes|no)',
            "active": r"active\s*(yes|no)",
            'alcohol': r'alcohol\s*(yes|no)'
        }
        for key, pattern in patterns.items():
            match = re.search(pattern, transcript, re.IGNORECASE)
            if match:
                input_data[key] = match.group(1)
            else:
                return None, f"Missing {key}. Please provide: {FEATURES_REQUIRED['heart']}. Example: 'predict heart age 45 sex male height 170 weight 80 systolic bp 140 diastolic bp 90 cholesterol above normal glucose normal smoke yes active yes alcohol no'"

        # Validate numerical inputs
        numerical_keys = ['age', 'height', 'weight', 'systolic_bp', 'diastolic_bp']
        for key in numerical_keys:
            try:
                float(input_data[key])
            except ValueError:
                return None, f"Invalid value for {key}. Please provide a numerical value."

        # Convert categorical inputs
        input_data['gender'] = 1 if input_data['gender'].lower() == 'male' else 2
        input_data['smoke'] = 1 if input_data['smoke'].lower() == 'yes' else 0
        input_data['alcohol'] = 1 if input_data['alcohol'].lower() == 'yes' else 0
        input_data['active'] = 1 if input_data['active'].lower() == 'yes' else 0
        cholesterol_map = {'normal': 1, 'abovenormal': 2, 'wellabovenormal': 3}
        glucose_map = {'normal': 1, 'abovenormal': 2, 'wellabovenormal': 3}
        try:
            input_data['cholesterol'] = cholesterol_map[input_data['cholesterol'].lower().replace(' ', '')]
            input_data['glucose'] = glucose_map[input_data['glucose'].lower().replace(' ', '')]
        except KeyError:
            return None, f"Invalid value for cholesterol or glucose. Must be one of: normal, above normal, well above normal."

    elif prediction_type == 'asthma':
        patterns = {
            'age': r'age\s*(\d+)',
            'gender': r'gender\s*(male|female)',
            'smoking': r'smoking\s*(yes|no)',
            'allergen_exposure': r'allergen_exposure\s*(yes|no)',
            'physical_activity_level': r'physical_activity_level\s*(\d+)',
            'air_quality_index': r'air_quality_index\s*(\d+)',
            'respiratory_infections': r'respiratory_infections\s*(yes|no)',
            'family_history': r'family_history\s*(yes|no)',
            'bmi': r'bmi\s*(\d+\.?\d*)',
            'stress_level': r'stress_level\s*(\d+)'
        }
        for key, pattern in patterns.items():
            match = re.search(pattern, transcript, re.IGNORECASE)
            if match:
                input_data[key] = match.group(1)
            else:
                return None, f"Missing {key}. Please provide: {FEATURES_REQUIRED['asthma']}"
        # Validate numerical inputs
        numerical_keys = ['age', 'physical_activity_level', 'air_quality_index', 'bmi', 'stress_level']
        for key in numerical_keys:
            try:
                float(input_data[key])
            except ValueError:
                return None, f"Invalid value for {key}. Please provide a numerical value."
        input_data['gender'] = 1 if input_data['gender'].lower() == 'male' else 0
        input_data['smoking'] = 1 if input_data['smoking'].lower() == 'yes' else 0
        input_data['allergen_exposure'] = 1 if input_data['allergen_exposure'].lower() == 'yes' else 0
        input_data['respiratory_infections'] = 1 if input_data['respiratory_infections'].lower() == 'yes' else 0
        input_data['family_history'] = 1 if input_data['family_history'].lower() == 'yes' else 0

    elif prediction_type == 'alzheimer':
        patterns = {
            'age': r'age\s*(\d+)',
            'gender': r'gender\s*(male|female)',
            'education_level': r'education_level\s*(\d+)',
            'family_history': r'family_history\s*(yes|no)',
            'smoking': r'smoking\s*(yes|no)',
            'physical_activity_level': r'physical_activity_level\s*(\d+)',
            'cognitive_decline': r'cognitive_decline\s*(yes|no)',
            'depression': r'depression\s*(yes|no)',
            'blood_pressure': r'blood_pressure\s*(\d+/\d+)',
            'cholesterol_level': r'cholesterol_level\s*(\d+\.?\d*)'
        }
        for key, pattern in patterns.items():
            match = re.search(pattern, transcript, re.IGNORECASE)
            if match:
                input_data[key] = match.group(1)
            else:
                return None, f"Missing {key}. Please provide: {FEATURES_REQUIRED['alzheimer']}"
        # Validate numerical inputs
        numerical_keys = ['age', 'education_level', 'physical_activity_level', 'cholesterol_level']
        for key in numerical_keys:
            try:
                float(input_data[key])
            except ValueError:
                return None, f"Invalid value for {key}. Please provide a numerical value."
        input_data['gender'] = 1 if input_data['gender'].lower() == 'male' else 0
        input_data['family_history'] = 1 if input_data['family_history'].lower() == 'yes' else 0
        input_data['smoking'] = 1 if input_data['smoking'].lower() == 'yes' else 0
        input_data['cognitive_decline'] = 1 if input_data['cognitive_decline'].lower() == 'yes' else 0
        input_data['depression'] = 1 if input_data['depression'].lower() == 'yes' else 0
        try:
            bp = input_data['blood_pressure'].split('/')
            input_data['systolic'] = float(bp[0])
            input_data['diastolic'] = float(bp[1])
        except (ValueError, IndexError):
            return None, f"Invalid value for blood_pressure. Must be in the format 'systolic/diastolic' (e.g., '120/80')."

    elif prediction_type == 'parkinson':
        patterns = {
            'age': r'age\s*(\d+)',
            'gender': r'gender\s*(male|female)',
            'family_history': r'family_history\s*(yes|no)',
            'tremor': r'tremor\s*(yes|no)',
            'rigidity': r'rigidity\s*(yes|no)',
            'bradykinesia': r'bradykinesia\s*(yes|no)',
            'postural_instability': r'postural_instability\s*(yes|no)',
            'speech_difficulty': r'speech_difficulty\s*(yes|no)',
            'sleep_disturbance': r'sleep_disturbance\s*(yes|no)',
            'cognitive_impairment': r'cognitive_impairment\s*(yes|no)'
        }
        for key, pattern in patterns.items():
            match = re.search(pattern, transcript, re.IGNORECASE)
            if match:
                input_data[key] = match.group(1)
            else:
                return None, f"Missing {key}. Please provide: {FEATURES_REQUIRED['parkinson']}"
        # Validate numerical inputs
        numerical_keys = ['age']
        for key in numerical_keys:
            try:
                float(input_data[key])
            except ValueError:
                return None, f"Invalid value for {key}. Please provide a numerical value."
        input_data['gender'] = 1 if input_data['gender'].lower() == 'male' else 0
        input_data['family_history'] = 1 if input_data['family_history'].lower() == 'yes' else 0
        input_data['tremor'] = 1 if input_data['tremor'].lower() == 'yes' else 0
        input_data['rigidity'] = 1 if input_data['rigidity'].lower() == 'yes' else 0
        input_data['bradykinesia'] = 1 if input_data['bradykinesia'].lower() == 'yes' else 0
        input_data['postural_instability'] = 1 if input_data['postural_instability'].lower() == 'yes' else 0
        input_data['speech_difficulty'] = 1 if input_data['speech_difficulty'].lower() == 'yes' else 0
        input_data['sleep_disturbance'] = 1 if input_data['sleep_disturbance'].lower() == 'yes' else 0
        input_data['cognitive_impairment'] = 1 if input_data['cognitive_impairment'].lower() == 'yes' else 0

    elif prediction_type == 'hypertension':
        patterns = {
            'age': r'age\s*(\d+)',
            'gender': r'gender\s*(male|female)',
            'bmi': r'bmi\s*(\d+\.?\d*)',
            'smoking': r'smoking\s*(yes|no)',
            'alcohol': r'alcohol\s*(yes|no)',
            'physical_activity_level': r'physical_activity_level\s*(\d+)',
            'stress_level': r'stress_level\s*(\d+)',
            'family_history': r'family_history\s*(yes|no)',
            'salt_intake': r'salt_intake\s*(\d+\.?\d*)',
            'blood_pressure': r'blood_pressure\s*(\d+/\d+)'
        }
        for key, pattern in patterns.items():
            match = re.search(pattern, transcript, re.IGNORECASE)
            if match:
                input_data[key] = match.group(1)
            else:
                return None, f"Missing {key}. Please provide: {FEATURES_REQUIRED['hypertension']}"
        # Validate numerical inputs
        numerical_keys = ['age', 'bmi', 'physical_activity_level', 'stress_level', 'salt_intake']
        for key in numerical_keys:
            try:
                float(input_data[key])
            except ValueError:
                return None, f"Invalid value for {key}. Please provide a numerical value."
        input_data['gender'] = 1 if input_data['gender'].lower() == 'male' else 0
        input_data['smoking'] = 1 if input_data['smoking'].lower() == 'yes' else 0
        input_data['alcohol'] = 1 if input_data['alcohol'].lower() == 'yes' else 0
        input_data['family_history'] = 1 if input_data['family_history'].lower() == 'yes' else 0
        try:
            bp = input_data['blood_pressure'].split('/')
            input_data['systolic'] = float(bp[0])
            input_data['diastolic'] = float(bp[1])
        except (ValueError, IndexError):
            return None, f"Invalid value for blood_pressure. Must be in the format 'systolic/diastolic' (e.g., '120/80')."

    elif prediction_type == 'diabetes':
        patterns = {
            'age': r'age\s*(\d+)',
            'gender': r'gender\s*(male|female)',
            'bmi': r'bmi\s*(\d+\.?\d*)',
            'family_history': r'family_history\s*(yes|no)',
            'physical_activity_level': r'physical_activity_level\s*(\d+)',
            'fasting_glucose': r'fasting_glucose\s*(\d+\.?\d*)',
            'hba1c': r'hba1c\s*(\d+\.?\d*)',
            'blood_pressure': r'blood_pressure\s*(\d+/\d+)',
            'cholesterol_level': r'cholesterol_level\s*(\d+\.?\d*)',
            'smoking': r'smoking\s*(yes|no)'
        }
        for key, pattern in patterns.items():
            match = re.search(pattern, transcript, re.IGNORECASE)
            if match:
                input_data[key] = match.group(1)
            else:
                return None, f"Missing {key}. Please provide: {FEATURES_REQUIRED['diabetes']}"
        # Validate numerical inputs
        numerical_keys = ['age', 'bmi', 'physical_activity_level', 'fasting_glucose', 'hba1c', 'cholesterol_level']
        for key in numerical_keys:
            try:
                float(input_data[key])
            except ValueError:
                return None, f"Invalid value for {key}. Please provide a numerical value."
        input_data['gender'] = 1 if input_data['gender'].lower() == 'male' else 0
        input_data['family_history'] = 1 if input_data['family_history'].lower() == 'yes' else 0
        input_data['smoking'] = 1 if input_data['smoking'].lower() == 'yes' else 0
        try:
            bp = input_data['blood_pressure'].split('/')
            input_data['systolic'] = float(bp[0])
            input_data['diastolic'] = float(bp[1])
        except (ValueError, IndexError):
            return None, f"Invalid value for blood_pressure. Must be in the format 'systolic/diastolic' (e.g., '120/80')."

    return input_data, None

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def predict(request):
    try:
        transcript = request.data.get('transcript')
        if not transcript:
            return Response({'error': 'No message provided'}, status=400)

        # Save the user's message
        ChatMessage.objects.create(
            user=request.user,
            message_type='sent',
            message_text=transcript
        )

        # Handle greetings
        if any(greeting in transcript.lower() for greeting in ['hello', 'hi', 'hey']):
            response_text = "Hello! I'm your AI health assistant. I can help predict risks for: Body Performance, Sleep Disorders, Heart Disease, Asthma, Alzheimer's, Parkinson's, Hypertension, and Diabetes. What would you like to check?"
            ChatMessage.objects.create(
                user=request.user,
                message_type='received',
                message_text=response_text
            )
            return Response({'message': response_text}, status=200)

        elif any(phrase in transcript.lower() for phrase in [
            'am good', 'fine', 'very good', 'doing well'
        ]):
            response_text = "That's great to hear! What health prediction would you like to make today? Options: body performance, sleep disorder, heart disease, asthma, Alzheimer, Parkinson, hypertension, or diabetes."
            ChatMessage.objects.create(
                user=request.user,
                message_type='received',
                message_text=response_text
            )
            return Response({'message': response_text}, status=200)

        # Determine prediction type
        prediction_type = None
        prediction_keywords = {
            'body': ['body performance', 'predict body', 'body'],
            'sleep': ['sleep disorder', 'sleep', 'insomnia', 'sleep apnea'],
            'heart': ['heart disease', 'heart', 'cardio'],
            'asthma': ['asthma', 'breathing', 'respiratory'],
            'alzheimer': ['alzheimer', 'dementia', 'memory loss'],
            'parkinson': ['parkinson', 'tremor', 'movement disorder'],
            'hypertension': ['hypertension', 'high blood pressure', 'blood pressure'],
            'diabetes': ['diabetes', 'blood sugar', 'glucose']
        }

        # Check if the user is requesting a prediction type
        for pred_type, keywords in prediction_keywords.items():
            if any(keyword in transcript.lower() for keyword in keywords):
                prediction_type = pred_type
                break

        if not prediction_type:
            response_text = "Please specify which health prediction you'd like: body performance, sleep disorder, heart disease, asthma, Alzheimer's, Parkinson's, hypertension, or diabetes."
            ChatMessage.objects.create(
                user=request.user,
                message_type='received',
                message_text=response_text
            )
            return Response({'message': response_text}, status=200)

        # Try to extract features from the transcript
        input_data, error = extract_features(transcript, prediction_type)

        # If features are missing, prompt for them
        if error:
            # Check if this is just a type selection (short message without numbers)
            has_numbers = bool(re.search(r'\d+', transcript))
            if not has_numbers and len(transcript.split()) <= 10:
                # Prompt for the required features
                response_text = f"Great! For {prediction_type} prediction, please provide these details in your next message:\n\n{FEATURES_REQUIRED[prediction_type]}\n\nExample format: 'age 45 gender male height 170 weight 80...'"
                ChatMessage.objects.create(
                    user=request.user,
                    message_type='received',
                    message_text=response_text
                )
                return Response({'message': response_text}, status=200)
            else:
                # User attempted but missed some features
                response_text = f"⚠️ {error}\n\nPlease provide all required information in one message."
                ChatMessage.objects.create(
                    user=request.user,
                    message_type='received',
                    message_text=response_text
                )
                return Response({'message': response_text}, status=400)

        # Proceed with prediction
        input_df = None
        prediction = None
        predicted_class_prob = None

        if prediction_type == 'body':
            input_df = pd.DataFrame([{
                'age': float(input_data['age']),
                'gender': input_data['gender'],
                'height_cm': float(input_data['height_cm']),
                'weight_kg': float(input_data['weight_kg']),
                'body fat_%': float(input_data['body_fat_%']),
                'diastolic': float(input_data['diastolic']),
                'systolic': float(input_data['systolic']),
                'gripForce': float(input_data['grip_force']),
                'sit and bend forward_cm': float(input_data['sit_and_bend_forward_cm']),
                'sit-ups counts': float(input_data['sit_ups_counts']),
                'broad jump_cm': float(input_data['broad_jump_cm'])
            }])
            try:
                prediction = body_model.predict(input_df)[0]
                probabilities = body_model.predict_proba(input_df)[0]
                predicted_class_prob = probabilities[int(prediction) - 1] * 100
            except Exception as e:
                logger.error(f"Body model prediction failed: {str(e)}")
                return Response({'error': f"Prediction failed: {str(e)}"}, status=400)

        elif prediction_type == 'sleep':
            input_df = pd.DataFrame([{
                'age': float(input_data['age']),
                'gender': input_data['gender'],
                'sleep duration': float(input_data['sleep_duration']),
                'quality of sleep': float(input_data['quality_of_sleep']),
                'physical activity level': float(input_data['physical_activity_level']),
                'stress level': float(input_data['stress_level']),
                'bmi category': input_data['bmi_category'],
                'heart rate': float(input_data['heart_rate']),
                'daily steps': float(input_data['daily_steps']),
                'systolic': input_data['systolic'],
                'diastolic': input_data['diastolic']
            }])
            try:
                prediction = sleep_model.predict(input_df)[0]
                probabilities = sleep_model.predict_proba(input_df)[0]
                predicted_class_prob = probabilities[int(prediction)] * 100
            except Exception as e:
                logger.error(f"Sleep model prediction failed: {str(e)}")
                return Response({'error': f"Prediction failed: {str(e)}"}, status=400)

        elif prediction_type == 'heart':
            input_df = pd.DataFrame([{
                'age': float(input_data['age']),
                'gender': input_data['gender'],
                'height': float(input_data['height']),
                'weight': float(input_data['weight']),
                'ap_hi': float(input_data['systolic_bp']),
                'ap_lo': float(input_data['diastolic_bp']),
                'cholesterol': input_data['cholesterol'],
                'gluc': input_data['glucose'],
                'smoke': input_data['smoke'],
                'alco': input_data['alcohol'],
                'active': input_data['active']
            }])
            try:
                prediction = heart_model.predict(input_df)[0]
                probabilities = heart_model.predict_proba(input_df)[0]
                predicted_class_prob = probabilities[int(prediction)] * 100
            except Exception as e:
                logger.error(f"Heart model prediction failed: {str(e)}")
                return Response({'error': f"Prediction failed: {str(e)}"}, status=400)

        elif prediction_type == 'asthma':
            input_df = pd.DataFrame([{
                'age': float(input_data['age']),
                'gender': input_data['gender'],
                'smoking': input_data['smoking'],
                'allergen exposure': input_data['allergen_exposure'],
                'physical activity level': float(input_data['physical_activity_level']),
                'air quality index': float(input_data['air_quality_index']),
                'respiratory infections': input_data['respiratory_infections'],
                'family history': input_data['family_history'],
                'bmi': float(input_data['bmi']),
                'stress level': float(input_data['stress_level'])
            }])
            try:
                prediction = asthma_model.predict(input_df)[0]
                probabilities = asthma_model.predict_proba(input_df)[0]
                predicted_class_prob = probabilities[int(prediction)] * 100
            except Exception as e:
                logger.error(f"Asthma model prediction failed: {str(e)}")
                return Response({'error': f"Prediction failed: {str(e)}"}, status=400)

        elif prediction_type == 'alzheimer':
            input_df = pd.DataFrame([{
                'age': float(input_data['age']),
                'gender': input_data['gender'],
                'education level': float(input_data['education_level']),
                'family history': input_data['family_history'],
                'smoking': input_data['smoking'],
                'physical activity level': float(input_data['physical_activity_level']),
                'cognitive decline': input_data['cognitive_decline'],
                'depression': input_data['depression'],
                'systolic': input_data['systolic'],
                'diastolic': input_data['diastolic'],
                'cholesterol level': float(input_data['cholesterol_level'])
            }])
            try:
                prediction = alzheimer_model.predict(input_df)[0]
                probabilities = alzheimer_model.predict_proba(input_df)[0]
                predicted_class_prob = probabilities[int(prediction)] * 100
            except Exception as e:
                logger.error(f"Alzheimer model prediction failed: {str(e)}")
                return Response({'error': f"Prediction failed: {str(e)}"}, status=400)

        elif prediction_type == 'parkinson':
            input_df = pd.DataFrame([{
                'age': float(input_data['age']),
                'gender': input_data['gender'],
                'family history': input_data['family_history'],
                'tremor': input_data['tremor'],
                'rigidity': input_data['rigidity'],
                'bradykinesia': input_data['bradykinesia'],
                'postural instability': input_data['postural_instability'],
                'speech difficulty': input_data['speech_difficulty'],
                'sleep disturbance': input_data['sleep_disturbance'],
                'cognitive impairment': input_data['cognitive_impairment']
            }])
            try:
                prediction = parkinson_model.predict(input_df)[0]
                probabilities = parkinson_model.predict_proba(input_df)[0]
                predicted_class_prob = probabilities[int(prediction)] * 100
            except Exception as e:
                logger.error(f"Parkinson model prediction failed: {str(e)}")
                return Response({'error': f"Prediction failed: {str(e)}"}, status=400)

        elif prediction_type == 'hypertension':
            input_df = pd.DataFrame([{
                'age': float(input_data['age']),
                'gender': input_data['gender'],
                'bmi': float(input_data['bmi']),
                'smoking': input_data['smoking'],
                'alcohol': input_data['alcohol'],
                'physical activity level': float(input_data['physical_activity_level']),
                'stress level': float(input_data['stress_level']),
                'family history': input_data['family_history'],
                'salt intake': float(input_data['salt_intake']),
                'systolic': input_data['systolic'],
                'diastolic': input_data['diastolic']
            }])
            try:
                prediction = hypertension_model.predict(input_df)[0]
                probabilities = hypertension_model.predict_proba(input_df)[0]
                predicted_class_prob = probabilities[int(prediction)] * 100
            except Exception as e:
                logger.error(f"Hypertension model prediction failed: {str(e)}")
                return Response({'error': f"Prediction failed: {str(e)}"}, status=400)

        elif prediction_type == 'diabetes':
            input_df = pd.DataFrame([{
                'age': float(input_data['age']),
                'gender': input_data['gender'],
                'bmi': float(input_data['bmi']),
                'family history': input_data['family_history'],
                'physical activity level': float(input_data['physical_activity_level']),
                'fasting glucose': float(input_data['fasting_glucose']),
                'hba1c': float(input_data['hba1c']),
                'systolic': input_data['systolic'],
                'diastolic': input_data['diastolic'],
                'cholesterol level': float(input_data['cholesterol_level']),
                'smoking': input_data['smoking']
            }])
            try:
                prediction = diabetes_model.predict(input_df)[0]
                probabilities = diabetes_model.predict_proba(input_df)[0]
                predicted_class_prob = probabilities[int(prediction)] * 100
            except Exception as e:
                logger.error(f"Diabetes model prediction failed: {str(e)}")
                return Response({'error': f"Prediction failed: {str(e)}"}, status=400)

        # Map prediction to readable label
        prediction_label = PREDICTION_LABELS[prediction_type].get(int(prediction), "Unknown")
        probability = round(predicted_class_prob, 2)

        # Generate advice using Grok API or fallback
        advice = generate_advice(input_data, prediction, prediction_type)

        # Save prediction history
        prediction_history = PredictionHistory.objects.create(
            user=request.user,
            prediction_type=prediction_type,
            input_data=input_data,
            prediction=prediction,
            prediction_label=prediction_label,
            probabilities=[probability],
            advice=advice
        )

        response_text = f"📊 **{prediction_type.upper()} Assessment Results**\n\n"
        response_text += f"**Prediction:** {prediction_label}\n"
        response_text += f"**Confidence:** {probability}%\n\n"
        response_text += f"**Recommendation:** {advice}\n\n"
        response_text += "⚠️ *Disclaimer: This is an AI prediction based on provided data. Please consult a healthcare professional for medical advice.*"

        ChatMessage.objects.create(
            user=request.user,
            message_type='received',
            message_text=response_text,
            prediction_history=prediction_history
        )

        return Response({
            'prediction_type': prediction_type,
            'prediction': prediction_label,
            'confidence': probability,
            'advice': advice,
            'input_data': input_data,
            'response_text': response_text
        }, status=200)

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        error_message = f"An error occurred: {str(e)}"
        ChatMessage.objects.create(
            user=request.user,
            message_type='received',
            message_text=error_message
        )
        return Response({'error': error_message}, status=400)

@api_view(['POST'])
@permission_classes([AllowAny])
def register(request):
    username = request.data.get('username')
    email = request.data.get('email')
    password = request.data.get('password')
    
    if not all([username, email, password]):
        return Response({"error": "All fields (username, email, password) are required"}, status=400)
    
    if User.objects.filter(email=email).exists():
        return Response({"error": "Email already exists"}, status=400)
    elif User.objects.filter(username=username).exists():
        return Response({"error": "Username already exists"}, status=400)
    else:
        try:
            user = User.objects.create_user(username=username, email=email, password=password)
            user.save()
            token, _ = Token.objects.get_or_create(user=user)
            return Response({
                "message": "User registered successfully",
                "token": token.key,
                "username": user.username,
                "email": user.email
            }, status=status.HTTP_201_CREATED)
        except Exception as e:
            return Response({"error": f"An error occurred: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
@permission_classes([AllowAny])
def login(request):
    if request.method == 'POST':
        username = request.data.get('username')
        password = request.data.get('password')

        if not username or not password:
            return Response({'error': 'Username and password required'}, status=400)

        user = auth.authenticate(request, username=username, password=password)
        if user is not None:
            token, _ = Token.objects.get_or_create(user=user)
            return Response({
                'message': 'Login successful',
                'username': user.username,
                'email': user.email,
                'token': token.key
            }, status=200)
        else:
            return Response({'error': 'Invalid credentials'}, status=400)
    return Response({'error': 'Invalid request method'}, status=405)

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def logout(request):
    if request.method == 'POST':
        try:
            request.user.auth_token.delete()
            return Response({'message': 'Logout successful'}, status=200)
        except Exception as e:
            return Response({'error': f'Logout failed: {str(e)}'}, status=400)
    return Response({'error': 'Invalid request method'}, status=405)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_history(request):
    try:
        chat_history = ChatMessage.objects.filter(user=request.user).values(
            'message_type', 'message_text', 'created_at',
            'prediction_history__prediction_type',
            'prediction_history__input_data',
            'prediction_history__prediction_label',
            'prediction_history__probabilities',
            'prediction_history__advice'
        ).order_by('-created_at')
        
        # Convert to list and format properly
        formatted_history = []
        for item in chat_history:
            formatted_item = {
                'message_type': item['message_type'],
                'message_text': item['message_text'],
                'created_at': item['created_at'],
                'prediction_type': item.get('prediction_history__prediction_type'),
                'input_data': item.get('prediction_history__input_data'),
                'prediction': item.get('prediction_history__prediction_label'),
                'confidence': item.get('prediction_history__probabilities')[0] if item.get('prediction_history__probabilities') else None,
                'advice': item.get('prediction_history__advice')
            }
            formatted_history.append(formatted_item)
            
        return Response(formatted_history, status=200)
    except Exception as e:
        logger.error(f"Error fetching history: {str(e)}")
        return Response({'error': str(e)}, status=400)
"""
Backend API for FaunaVision
Accepts video and animal parameters, processes with vision model,
and uses OpenAI to determine health status.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import cv2
import tempfile
import shutil
from datetime import datetime
from typing import Dict, List, Optional
import logging

# Import YOLO behavior classifier
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.yolo_behavior_classifier import YOLOBehaviorClassifier

# OpenAI integration
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI library not available. Install with: pip install openai")

# Gemini integration
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logging.warning("Gemini library not available. Install with: pip install google-generativeai")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Configuration
UPLOAD_FOLDER = "temp"
MAX_VIDEO_SIZE = 100 * 1024 * 1024  # 100MB
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize YOLO behavior classifier
# Set YOLO_MODEL_PATH environment variable to path of trained model
# Example: export YOLO_MODEL_PATH="models/behavior_classifier.pt"
yolo_classifier = None
yolo_model_path = os.getenv("YOLO_MODEL_PATH", None)
try:
    yolo_classifier = YOLOBehaviorClassifier(model_path=yolo_model_path)
    if yolo_classifier.model is not None:
        logger.info("YOLO behavior classifier initialized successfully")
    else:
        logger.warning("YOLO classifier initialized but model not loaded (using placeholder)")
except Exception as e:
    logger.warning(f"YOLO classifier initialization failed: {e}. Will use placeholder.")


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def process_video_with_yolo(video_path: str) -> Dict:
    """
    Process video with YOLO model to get behavior time percentages.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with:
        - behavior_percentages: Dict[str, float] - Percentages for each behavior (sums to 1.0)
        - primary_behavior: str - Most common behavior
        - length_seconds: float - Video duration
        - frame_interval: float - Processing interval used
    """
    try:
        # Get video duration
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        
        # Process with YOLO classifier
        frame_interval = 1.0  # Process every 1 second
        behavior_percentages = yolo_classifier.analyze_video_percentages(
            video_path, 
            frame_interval=frame_interval
        )
        
        # Get primary behavior
        primary_behavior, primary_percentage = yolo_classifier.get_primary_behavior(behavior_percentages)
        
        return {
            "behavior_percentages": behavior_percentages,
            "primary_behavior": primary_behavior,
            "primary_percentage": primary_percentage,
            "length_seconds": duration,
            "frame_interval": frame_interval
        }
        
    except Exception as e:
        logger.error(f"Error processing video with YOLO: {e}", exc_info=True)
        # Return placeholder percentages on error
        placeholder_percentages = {
            "scratching": 0.33,
            "pacing": 0.33,
            "sleeping": 0.34
        }
        return {
            "behavior_percentages": placeholder_percentages,
            "primary_behavior": "unknown",
            "primary_percentage": 0.0,
            "length_seconds": 0.0,
            "frame_interval": 1.0,
            "error": str(e)
        }


def determine_health_with_ai(
    species: str,
    age: Optional[str],
    diet: Optional[str],
    health_conditions: Optional[str],
    behavior_percentages: Dict[str, float],
    length_seconds: float,
    use_gemini: bool = False
) -> Dict:
    """
    Use OpenAI or Gemini API to determine if animal is healthy based on behavior percentages and parameters.
    
    Args:
        species: Animal species
        age: Animal age
        diet: Animal diet
        health_conditions: Existing health conditions
        behavior_percentages: Dictionary with time percentages for each behavior
        length_seconds: Video duration in seconds
        use_gemini: If True, use Gemini instead of OpenAI
        
    Returns:
        Dictionary with health assessment
    """
    # Check API availability
    if use_gemini:
        if not GEMINI_AVAILABLE:
            logger.error("Gemini not available")
            return {
                "is_healthy": None,
                "reasoning": "Gemini API not configured",
                "recommendations": "Please configure Gemini API key"
            }
    else:
        if not OPENAI_AVAILABLE:
            logger.error("OpenAI not available")
            return {
                "is_healthy": None,
                "reasoning": "OpenAI API not configured",
                "recommendations": "Please configure OpenAI API key"
            }
    
    try:
        # Build context for AI
        length_minutes = length_seconds / 60.0
        
        # Format behavior percentages
        behavior_summary = "\n".join([
            f"- {behavior.capitalize()}: {percentage:.1%} of video time"
            for behavior, percentage in behavior_percentages.items()
        ])
        
        system_prompt = "You are an expert zoo veterinarian. Always respond with valid JSON only."
        user_prompt = f"""You are an expert zoo veterinarian analyzing animal behavior and health.

Animal Information:
- Species: {species}
- Age: {age if age else 'Unknown'}
- Diet: {diet if diet else 'Unknown'}
- Existing Health Conditions: {health_conditions if health_conditions else 'None reported'}

Observed Behavior (Time Percentages):
{behavior_summary}
- Video Duration: {length_minutes:.2f} minutes ({length_seconds:.2f} seconds)

Note: The percentages above sum to 100% and represent how much time the animal spent in each behavior during the video.

Based on this information, determine if the animal is healthy or unhealthy.

Consider:
1. Are the behavior percentages normal for this species? (e.g., excessive pacing >50% may indicate stress)
2. Is the distribution of behaviors healthy? (healthy animals typically switch between behaviors)
3. Are there concerning patterns? (e.g., 90% pacing suggests zoochosis/stress)
4. How do the animal's age, diet, and existing conditions affect the assessment?
5. Species-specific norms: What are normal behavior distributions for this species?

Respond in JSON format with:
{{
    "is_healthy": true or false,
    "reasoning": "Detailed explanation of your assessment based on behavior percentages",
    "recommendations": "Specific recommendations for the animal's care"
}}"""
        
        import json
        import re
        
        if use_gemini:
            # Use Gemini API
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            model = genai.GenerativeModel('gemini-pro')
            
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            response = model.generate_content(full_prompt)
            response_text = response.text.strip()
        else:
            # Use OpenAI API
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model="gpt-4",  # or "gpt-3.5-turbo" for faster/cheaper
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            response_text = response.choices[0].message.content.strip()
        
        # Parse response
        # Remove markdown code blocks if present
        response_text = re.sub(r'```json\s*', '', response_text)
        response_text = re.sub(r'```\s*', '', response_text)
        response_text = response_text.strip()
        
        try:
            health_assessment = json.loads(response_text)
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            logger.warning("Failed to parse AI response as JSON, using fallback")
            health_assessment = {
                "is_healthy": None,
                "reasoning": response_text,
                "recommendations": "Please review the reasoning above"
            }
        
        return health_assessment
        
    except Exception as e:
        logger.error(f"Error calling AI API: {e}", exc_info=True)
        return {
            "is_healthy": None,
            "reasoning": f"Error assessing health: {str(e)}",
            "recommendations": "Please check API configuration"
        }


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "yolo_classifier": yolo_classifier is not None and yolo_classifier.model is not None,
        "yolo_model_path": os.getenv("YOLO_MODEL_PATH", "Not set"),
        "openai_available": OPENAI_AVAILABLE,
        "gemini_available": GEMINI_AVAILABLE
    })


@app.route("/analyze", methods=["POST"])
def analyze_animal():
    """
    Main endpoint to analyze animal health from video and parameters.
    
    Expected request:
    - Form data with 'video' file
    - JSON or form data with parameters:
      - species: str (required)
      - age: str (optional)
      - diet: str (optional)
      - health_conditions: str (optional)
    
    Returns:
    {
        "species": str,
        "behavior_observed": str,
        "length_seconds": float,
        "is_healthy": bool,
        "reasoning": str,
        "recommendations": str,
        "confidence": float
    }
    """
    # Check for video file
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    
    video_file = request.files["video"]
    
    if video_file.filename == "":
        return jsonify({"error": "No video file selected"}), 400
    
    if not allowed_file(video_file.filename):
        return jsonify({"error": f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"}), 400
    
    # Get parameters
    species = request.form.get("species") or request.json.get("species") if request.is_json else None
    if not species:
        return jsonify({"error": "Species parameter is required"}), 400
    
    age = request.form.get("age") or (request.json.get("age") if request.is_json else None)
    diet = request.form.get("diet") or (request.json.get("diet") if request.is_json else None)
    health_conditions = request.form.get("health_conditions") or (request.json.get("health_conditions") if request.is_json else None)
    
    # Save video to temporary file
    temp_dir = tempfile.mkdtemp()
    video_path = os.path.join(temp_dir, video_file.filename)
    
    try:
        video_file.save(video_path)
        
        # Check file size
        file_size = os.path.getsize(video_path)
        if file_size > MAX_VIDEO_SIZE:
            return jsonify({"error": f"Video file too large. Max size: {MAX_VIDEO_SIZE / 1024 / 1024}MB"}), 400
        
        logger.info(f"Processing video: {video_file.filename} for species: {species}")
        
        # Step 1: Process video with YOLO model to get behavior percentages
        logger.info("Step 1: Processing video with YOLO behavior classifier...")
        yolo_result = process_video_with_yolo(video_path)
        
        if "error" in yolo_result:
            return jsonify({"error": f"Video processing failed: {yolo_result['error']}"}), 500
        
        behavior_percentages = yolo_result["behavior_percentages"]
        primary_behavior = yolo_result["primary_behavior"]
        primary_percentage = yolo_result["primary_percentage"]
        length_seconds = yolo_result["length_seconds"]
        
        logger.info(f"Behavior percentages: {behavior_percentages}")
        logger.info(f"Primary behavior: {primary_behavior} ({primary_percentage:.1%})")
        logger.info(f"Video duration: {length_seconds:.2f}s")
        
        # Step 2: Determine health status with OpenAI/Gemini using behavior percentages
        use_gemini = os.getenv("USE_GEMINI", "false").lower() == "true"
        ai_provider = "Gemini" if use_gemini else "OpenAI"
        logger.info(f"Step 2: Assessing health with {ai_provider} using behavior percentages...")
        health_assessment = determine_health_with_ai(
            species=species,
            age=age,
            diet=diet,
            health_conditions=health_conditions,
            behavior_percentages=behavior_percentages,
            length_seconds=length_seconds,
            use_gemini=use_gemini
        )
        
        # Step 3: Build response
        response = {
            "species": species,
            "behavior_percentages": {
                k: round(v, 4) for k, v in behavior_percentages.items()
            },
            "primary_behavior": primary_behavior,
            "primary_behavior_percentage": round(primary_percentage, 4),
            "length_seconds": round(length_seconds, 2),
            "length_minutes": round(length_seconds / 60.0, 2),
            "is_healthy": health_assessment.get("is_healthy"),
            "reasoning": health_assessment.get("reasoning", ""),
            "recommendations": health_assessment.get("recommendations", "")
        }
        
        logger.info(f"Analysis complete. Health status: {response['is_healthy']}")
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error in analyze endpoint: {e}", exc_info=True)
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500
        
    finally:
        # Clean up temporary files
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        except Exception as e:
            logger.warning(f"Error cleaning up temp files: {e}")


@app.route("/analyze/batch", methods=["POST"])
def analyze_batch():
    """
    Analyze multiple videos in batch.
    
    Expected request:
    - JSON with array of analysis requests
    - Each request should have video file path and parameters
    
    Note: This is a placeholder for future batch processing.
    """
    return jsonify({"error": "Batch processing not yet implemented"}), 501


if __name__ == "__main__":
    # Check for required environment variables
    if not os.getenv("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY not set. Health assessment will not work.")
        logger.warning("Set it with: export OPENAI_API_KEY='your-key-here'")
    
    # Run Flask app
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "False").lower() == "true"
    
    logger.info(f"Starting FaunaVision backend on port {port}")
    logger.info(f"Debug mode: {debug}")
    
    app.run(host="0.0.0.0", port=port, debug=debug)

import cv2
import mediapipe as mp
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle
import os
import time
import google.generativeai as genai
from gtts import gTTS
import tempfile
import threading
import os.path
from os import environ
import sys
import io
import re
try:
    from playsound import playsound
    PLAYSOUND_AVAILABLE = True
except ImportError:
    PLAYSOUND_AVAILABLE = False
    print("Warning: playsound library not found. Using fallback audio playback.")
    print("Install using: pip install playsound")

# Suppress MediaPipe initialization warnings
original_stderr = sys.stderr
sys.stderr = io.StringIO()  # Redirect stderr to a string buffer

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,       # Process video stream
    max_num_hands=2,             # Detect up to two hands
    min_detection_confidence=0.5, # Minimum confidence for initial detection
    min_tracking_confidence=0.5  # Minimum confidence for tracking
)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles # For nicer landmark drawing

# Restore stderr after MediaPipe initialization
sys.stderr = original_stderr

# Initialize Gemini API for language processing
# Get your API key from https://ai.google.dev/ and set it as an environment variable
GEMINI_API_KEY = environ.get('GEMINI_API_KEY', '')

# If not found in environment, try reading from a local file
if not GEMINI_API_KEY:
    api_key_file = 'gemini_api_key.txt'
    if os.path.exists(api_key_file):
        try:
            with open(api_key_file, 'r') as f:
                GEMINI_API_KEY = f.read().strip()
            print(f"API key loaded from {api_key_file}")
        except Exception as e:
            print(f"Error reading API key from file: {e}")

try:
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-pro')
        GEMINI_ENABLED = True
        print("Gemini AI initialized successfully")
    else:
        print("Warning: GEMINI_API_KEY not found in environment variables or local file.")
        print("Set it using: export GEMINI_API_KEY='your_api_key'")
        print("Or create a file 'gemini_api_key.txt' with your API key")
        print("Get your API key from https://ai.google.dev/")
        GEMINI_ENABLED = False
except Exception as e:
    print(f"Error initializing Gemini AI: {e}")
    GEMINI_ENABLED = False

# Constants
NUM_LANDMARKS = 21  # Number of hand landmarks in MediaPipe
LANDMARK_DIMS = 3   # x, y, z coordinates for each landmark
MAX_HANDS = 2       # Maximum number of hands the model expects features for
# --- Corrected: Defined SINGLE_HAND_FEATURE_SIZE globally ---
SINGLE_HAND_FEATURE_SIZE = NUM_LANDMARKS * LANDMARK_DIMS # Size for one hand's landmarks
FEATURE_VECTOR_SIZE = SINGLE_HAND_FEATURE_SIZE * MAX_HANDS # Total size of the input vector for the model

# --- Utility Functions ---

def get_screen_resolution():
    """Tries to get screen resolution. Falls back to Full HD."""
    try:
        import screeninfo
        # Get the primary monitor
        primary_monitor = None
        for m in screeninfo.get_monitors():
            if m.is_primary:
                primary_monitor = m
                break
        # If no primary, just take the first one
        screen = primary_monitor if primary_monitor else screeninfo.get_monitors()[0]
        return screen.width, screen.height
    except ImportError:
        print("Warning: 'screeninfo' library not found. Falling back to 1920x1080.")
        print("Install using: pip install screeninfo")
        return 1920, 1080 # Fallback resolution
    except Exception as e:
        print(f"Warning: Error getting screen info ({e}). Falling back to 1920x1080.")
        return 1920, 1080

SCREEN_WIDTH, SCREEN_HEIGHT = get_screen_resolution()

# --- Gemini AI and Speech Functions ---

def get_word_suggestions(current_letters, context=None):
    """
    Get word suggestions based on current letters using Gemini AI.
    
    Args:
        current_letters: Current letters recognized
        context: Optional context from previous words
        
    Returns:
        List of suggested words
    """
    if not GEMINI_ENABLED or not current_letters:
        return []
    
    try:
        prompt = f"I'm using sign language and have entered these letters: '{current_letters}'. "
        if context:
            prompt += f"Previous words in my sentence are: '{context}'. "
        prompt += "Suggest 3 likely words I might be trying to spell, ordered by probability. Just respond with the words separated by commas, nothing else."
        
        response = gemini_model.generate_content(prompt)
        suggestions = response.text.strip().split(',')
        return [s.strip() for s in suggestions[:3]]  # Return up to 3 suggestions
    except Exception as e:
        print(f"Error getting word suggestions: {e}")
        return []

def check_spelling(word):
    """
    Check if a word might be misspelled and suggest corrections.
    
    Args:
        word: Word to check for spelling errors
        
    Returns:
        Suggested correction or None if no correction needed
    """
    if not GEMINI_ENABLED or not word:
        return None
    
    # Common words we always want to check (like your example "HACV")
    common_misspellings = {
        "HACV": "HAVE", 
        "EAYS": "EASY",
        "THW": "THE",
        "WIT": "WITH",
        "YOO": "YOU",
        "CANN": "CAN",
        "DEOS": "DOES",
        "WAT": "WHAT",
        "WNAT": "WANT",
        "THSI": "THIS",
        "TAHT": "THAT"
    }
    
    # First check against our common misspellings dictionary
    if word.upper() in common_misspellings:
        return common_misspellings[word.upper()]
    
    # If it's a very short word, only check with API if it's clearly not a common word
    if len(word) < 2:
        return None
    
    try:
        prompt = f"This word might be misspelled: '{word}'. If it's likely a misspelling, suggest the correct spelling. If it's likely correct or a proper noun, respond with 'CORRECT'. Just give the corrected word or 'CORRECT', nothing else."
        
        response = gemini_model.generate_content(prompt)
        suggestion = response.text.strip()
        
        # If the AI thinks it's correct or returns the same word, return None
        if suggestion == 'CORRECT' or suggestion.lower() == word.lower():
            return None
        
        return suggestion
    except Exception as e:
        print(f"Error checking spelling: {e}")
        return None

def process_sentence_with_ai(sentence):
    """
    Process a sentence with Gemini AI to fix grammar and improve coherence.
    
    Args:
        sentence: Raw sentence from sign language recognition
        
    Returns:
        Improved sentence
    """
    if not GEMINI_ENABLED or not sentence:
        return sentence
    
    try:
        prompt = f"Fix the grammar and improve coherence of this sentence from sign language: '{sentence}'. Respond with ONLY the corrected sentence, nothing else."
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error processing sentence: {e}")
        return sentence

def speak_text(text):
    """
    Convert text to speech and play it.
    
    Args:
        text: Text to speak
    """
    if not text:
        return
        
    def speak_thread():
        try:
            # Create temporary file for audio
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
                temp_filename = temp_file.name
            
            # Generate speech
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save(temp_filename)
            
            # Play the audio directly without opening media player
            if PLAYSOUND_AVAILABLE:
                playsound(temp_filename)
            else:
                # Fallback methods that try to play directly
                if os.name == 'posix':  # Linux/Mac
                    os.system(f"mpg123 -q {temp_filename}")
                else:  # Windows
                    from subprocess import call
                    call(["powershell", "-c", f"(New-Object Media.SoundPlayer '{temp_filename}').PlaySync()"], 
                         stdout=open(os.devnull, 'w'), stderr=open(os.devnull, 'w'))
                
            # Clean up temp file
            try:
                os.unlink(temp_filename)
            except:
                pass
        except Exception as e:
            print(f"Error in speech synthesis: {e}")
    
    # Run speech in separate thread to avoid blocking UI
    threading.Thread(target=speak_thread).start()

# Function to extract hand landmarks
def extract_landmarks(image):
    """
    Processes an image to detect hands and extract landmarks.

    Args:
        image: The input image (BGR format).

    Returns:
        A tuple containing:
        - left_landmarks (np.array): Landmarks for the left hand (size SINGLE_HAND_FEATURE_SIZE, padded with zeros if not detected).
        - right_landmarks (np.array): Landmarks for the right hand (size SINGLE_HAND_FEATURE_SIZE, padded with zeros if not detected).
        - image_with_drawings (np.array): The input image with landmarks drawn.
        - hands_detected (list): A list [bool, bool] indicating if [left, right] hands were detected.
    """
    image_to_process = image.copy() # Work on a copy
    image_rgb = cv2.cvtColor(image_to_process, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False # Improve performance
    results = hands.process(image_rgb)
    image_rgb.flags.writeable = True # Back to writeable
    image_with_drawings = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR) # Convert back for drawing

    # Initialize arrays using the global constant
    left_landmarks = np.zeros(SINGLE_HAND_FEATURE_SIZE)
    right_landmarks = np.zeros(SINGLE_HAND_FEATURE_SIZE)
    hands_detected = [False, False]  # [left, right]

    # --- Removed redundant local definition: single_hand_feature_size = NUM_LANDMARKS * LANDMARK_DIMS ---

    if results.multi_hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if hand_idx >= MAX_HANDS: # Should not happen with max_num_hands=2, but good practice
                break

            # Determine if it's left or right hand (requires multi_handedness)
            hand_type = "Unknown"
            if results.multi_handedness and len(results.multi_handedness) > hand_idx:
                 hand_type = results.multi_handedness[hand_idx].classification[0].label

            # Draw hand landmarks on the image
            mp_drawing.draw_landmarks(
                image_with_drawings,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Extract coordinates into a temporary flat array
            # Ensure we handle cases where landmarks might be missing (though unlikely with default settings)
            landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
            # Verify size before assignment (optional safety check)
            if landmarks_array.shape[0] != SINGLE_HAND_FEATURE_SIZE:
                print(f"Warning: Landmark array size mismatch for hand {hand_type}. Expected {SINGLE_HAND_FEATURE_SIZE}, got {landmarks_array.shape[0]}. Skipping hand.")
                continue # Skip this hand if size is wrong


            # Store landmarks based on hand type
            if hand_type == "Left":
                left_landmarks = landmarks_array
                hands_detected[0] = True
            elif hand_type == "Right":
                right_landmarks = landmarks_array
                hands_detected[1] = True
            # else: Handle "Unknown" if necessary, maybe assign based on position? For now, ignore.

    return left_landmarks, right_landmarks, image_with_drawings, hands_detected

# --- Data Collection ---

def list_available_cameras():
    """Lists all available camera devices."""
    available_cameras = []
    print("Searching for available cameras...")
    for i in range(10):  # Check first 10 indices
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    available_cameras.append(i)
                    print(f"Found camera {i}")
            cap.release()
        except Exception as e:
            print(f"Error checking camera {i}: {e}")
    
    if not available_cameras:
        print("No cameras were detected.")
        # Default to camera 0 even if not detected
        print("Adding default camera (0) to the list.")
        available_cameras.append(0)
    
    return available_cameras

def select_camera():
    """Allows user to select a camera device."""
    available_cameras = list_available_cameras()
    if not available_cameras:
        print("No cameras found!")
        return 0
    
    print("\nAvailable cameras:")
    for cam_idx in available_cameras:
        print(f"Camera {cam_idx}")
    
    while True:
        try:
            choice = int(input("\nSelect camera number (0-9): "))
            if choice in available_cameras:
                return choice
            print("Invalid camera number. Please try again.")
        except ValueError:
            print("Please enter a valid number.")

def collect_training_data(gesture_class, num_samples=100, camera_idx=0):
    """Collects training samples for a given gesture class."""
    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_idx}.")
        return [], []

    data = []
    labels = []
    count = 0

    # Ask if the gesture requires both hands
    print(f"\nGesture: {gesture_class}")
    print("Does this gesture require both hands?")
    print("1. Yes - Both hands together")
    print("2. No - One hand at a time")
    while True:
        choice = input("Enter your choice (1/2): ")
        if choice in ['1', '2']:
            break
        print("Invalid choice. Please enter 1 or 2.")
    requires_both_hands = (choice == '1')

    print("-" * 30)
    print(f"Collecting data for gesture: {gesture_class} ({num_samples} samples)")
    print("Press 'q' to stop early.")
    if requires_both_hands:
        print("Instruction: Show the gesture using BOTH hands simultaneously.")
    else:
        print("Instruction: Show the gesture using EITHER your left OR right hand.")
    print("Ensure the hand(s) are clearly visible in the center.")
    print("-" * 30)
    time.sleep(2) # Give user time to read

    # --- Fullscreen Window Setup ---
    window_name = f'Collecting Data: {gesture_class}'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # Attempt fullscreen - might not work perfectly on all OS/Window Managers
    # cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN) # Can be glitchy
    cv2.resizeWindow(window_name, SCREEN_WIDTH, SCREEN_HEIGHT) # Often more reliable
    cv2.moveWindow(window_name, 0, 0)


    # Get initial frame dimensions for scaling calculation
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read initial frame from camera.")
        cap.release()
        cv2.destroyAllWindows()
        return [], []
    frame_h, frame_w = frame.shape[:2]

    # Calculate scaling to fit frame within screen while preserving aspect ratio
    scale_w = SCREEN_WIDTH / frame_w if frame_w > 0 else 1
    scale_h = SCREEN_HEIGHT / frame_h if frame_h > 0 else 1
    scale = min(scale_w, scale_h)
    disp_w = int(frame_w * scale)
    disp_h = int(frame_h * scale)

    # Calculate offsets to center the display frame
    offset_x = (SCREEN_WIDTH - disp_w) // 2
    offset_y = (SCREEN_HEIGHT - disp_h) // 2

    # Info panel configuration (relative to the *display* area)
    panel_width = 400
    panel_start_x = disp_w - panel_width if disp_w > panel_width else 0 # Position within displayed frame
    panel_start_x_screen = offset_x + panel_start_x                  # Absolute screen X
    panel_end_x_screen = offset_x + disp_w                            # Absolute screen X
    panel_start_y_screen = offset_y                                   # Absolute screen Y
    panel_end_y_screen = offset_y + disp_h                            # Absolute screen Y
    text_x_start = panel_start_x_screen + 20                          # Text start position on screen
    text_panel_width_screen = panel_end_x_screen - text_x_start - 20  # Max text width on screen


    last_collection_time = time.time()
    collection_delay = 0.1 # Minimum seconds between collecting samples

    while count < num_samples:
        ret, frame = cap.read()
        if not ret:
            print("Warning: Dropped frame.")
            continue

        # --- Landmark Extraction (on original frame) ---
        # Process the *original* frame for accurate landmarks
        left_landmarks, right_landmarks, _, hands_detected = extract_landmarks(frame) # We don't need the drawn image from here

        # --- Display Frame Preparation ---
        display_canvas = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8) # Black canvas
        resized_frame = cv2.resize(frame, (disp_w, disp_h)) # Resize for display

        # Place resized frame onto the canvas
        display_canvas[offset_y:offset_y+disp_h, offset_x:offset_x+disp_w] = resized_frame

        # --- Draw Landmarks on Display Frame ---
        # We need to re-run detection/drawing on the *displayed portion* or scale landmarks.
        # Re-running is simpler for display purposes.
        display_frame_slice = display_canvas[offset_y:offset_y+disp_h, offset_x:offset_x+disp_w]
        # Make a copy to avoid modifying the canvas directly if drawing fails
        display_slice_copy = display_frame_slice.copy()
        display_slice_rgb = cv2.cvtColor(display_slice_copy, cv2.COLOR_BGR2RGB)
        display_slice_rgb.flags.writeable = False
        results_display = hands.process(display_slice_rgb)
        display_slice_rgb.flags.writeable = True
        # Draw onto the slice *within* the main canvas
        if results_display.multi_hand_landmarks:
            for hand_landmarks_disp in results_display.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    display_frame_slice, # Draw directly on the slice in the canvas
                    hand_landmarks_disp,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
        # No need to put slice back if we drew directly on it


        # --- Information Panel Overlay ---
        overlay = display_canvas.copy()
        cv2.rectangle(overlay, (panel_start_x_screen, panel_start_y_screen),
                      (panel_end_x_screen, panel_end_y_screen), (0, 0, 0), -1) # Black rectangle
        alpha = 0.6 # Transparency
        display_canvas = cv2.addWeighted(overlay, alpha, display_canvas, 1 - alpha, 0)


        # --- Data Collection Logic ---
        current_time = time.time()
        collected_this_frame = False
        can_collect = (current_time - last_collection_time) > collection_delay

        feature_vector = None
        collection_message = ""

        if requires_both_hands:
            if all(hands_detected):
                if can_collect:
                    feature_vector = np.concatenate([left_landmarks, right_landmarks])
                    collection_message = f"Collected {count + 1}/{num_samples} (Both Hands)"
                else:
                     collection_message = "Hold steady..."
            else:
                collection_message = "Waiting for BOTH hands..."
        else: # Single hand mode
            if hands_detected[0]: # Left hand detected
                if can_collect:
                    # --- Corrected: Use global constant ---
                    feature_vector = np.concatenate([left_landmarks, np.zeros(SINGLE_HAND_FEATURE_SIZE)])
                    collection_message = f"Collected {count + 1}/{num_samples} (Left Hand)"
                else:
                     collection_message = "Hold steady..."
            elif hands_detected[1]: # Right hand detected (and left wasn't)
                 if can_collect:
                    # --- Corrected: Use global constant ---
                    feature_vector = np.concatenate([np.zeros(SINGLE_HAND_FEATURE_SIZE), right_landmarks])
                    collection_message = f"Collected {count + 1}/{num_samples} (Right Hand)"
                 else:
                     collection_message = "Hold steady..."
            else:
                collection_message = "Waiting for ONE hand..."

        if feature_vector is not None:
            # Verify feature vector size before appending
            if feature_vector.shape[0] == FEATURE_VECTOR_SIZE:
                data.append(feature_vector)
                labels.append(gesture_class)
                count += 1
                collected_this_frame = True
                last_collection_time = current_time # Reset timer
                print(collection_message) # Print to console as well
            else:
                 print(f"Error: Feature vector size mismatch! Expected {FEATURE_VECTOR_SIZE}, got {feature_vector.shape[0]}")
                 collection_message = "Size Error!" # Show error on screen


        # --- Display Information Panel Text ---
        y_pos = panel_start_y_screen + 50
        line_height = 40
        text_color = (0, 255, 0) # Green

        def draw_text(text, y, size=0.8, color=text_color, bold=False):
            thickness = 2 if bold else 1
            try:
                cv2.putText(display_canvas, text, (text_x_start, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness, cv2.LINE_AA)
            except Exception as e:
                print(f"Error drawing text '{text}': {e}") # Handle potential errors during drawing
            return y + line_height

        y_pos = draw_text('ISL GESTURE COLLECTION', y_pos, size=1.0, bold=True)
        cv2.line(display_canvas, (text_x_start, y_pos - (line_height // 2)), (panel_end_x_screen - 20, y_pos - (line_height // 2)), text_color, 1)

        y_pos = draw_text(f'Gesture: {gesture_class}', y_pos, size=0.9, bold=True)
        y_pos = draw_text(f'Progress: {count}/{num_samples}', y_pos)
        y_pos += line_height // 2 # Extra space

        y_pos = draw_text('INSTRUCTIONS:', y_pos, bold=True)
        cv2.line(display_canvas, (text_x_start, y_pos - (line_height // 2)), (panel_end_x_screen - 20, y_pos - (line_height // 2)), text_color, 1)
        instr = ['Show gesture clearly.', 'Hold steady for collection.', 'Press "q" to quit.']
        if requires_both_hands: instr.insert(0,'Use BOTH hands.')
        else: instr.insert(0, 'Use ONE hand (Left or Right).')
        for line in instr: y_pos = draw_text(line, y_pos, size=0.7)
        y_pos += line_height // 2

        y_pos = draw_text('HAND STATUS:', y_pos, bold=True)
        cv2.line(display_canvas, (text_x_start, y_pos - (line_height // 2)), (panel_end_x_screen - 20, y_pos - (line_height // 2)), text_color, 1)
        status_color_left = (0, 255, 0) if hands_detected[0] else (0, 0, 255)
        status_color_right = (0, 255, 0) if hands_detected[1] else (0, 0, 255)
        y_pos = draw_text(f'Left Hand: {"Detected" if hands_detected[0] else "Not Detected"}', y_pos, color=status_color_left)
        y_pos = draw_text(f'Right Hand: {"Detected" if hands_detected[1] else "Not Detected"}', y_pos, color=status_color_right)
        y_pos += line_height // 2

        status_color = (0, 255, 0) if collected_this_frame else (0, 255, 255) # Green if collected, Yellow otherwise
        if "Error" in collection_message: status_color = (0,0,255) # Red for errors
        y_pos = draw_text(collection_message, y_pos, color=status_color, bold=True)

        # --- Show Frame ---
        cv2.imshow(window_name, display_canvas)

        # --- Handle Keypress ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\nCollection stopped early by user.")
            break
        # elif key == ord('f'): # Fullscreen toggle can be unreliable
        #      current_mode = cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN)
        #      cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
        #                            cv2.WINDOW_NORMAL if current_mode == cv2.WINDOW_FULLSCREEN else cv2.WINDOW_FULLSCREEN)

    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()
    print(f"Finished collecting data for {gesture_class}.")
    return data, labels

# --- Model Training ---

def train_model(data, labels):
    """Trains an SVM model on the collected data."""
    if not data or not labels:
        print("Error: No data provided for training.")
        return None

    X = np.array(data)
    y = np.array(labels)
    unique_labels = np.unique(y)

    if len(unique_labels) < 2:
        print(f"Error: Need data for at least two different gestures to train. Found only: {unique_labels}")
        return None

    # Check if any class has fewer samples than required for stratification (default is 2 for test_split)
    min_samples_per_class = 2
    counts = {label: np.sum(y == label) for label in unique_labels}
    can_stratify = all(count >= min_samples_per_class for count in counts.values())

    # Determine test size (e.g., 20%, but ensure at least 1 sample per class in test set if possible)
    test_size = 0.2

    print(f"Total samples: {len(X)}")
    print(f"Gestures: {counts}")


    if len(X) < 5 or not can_stratify: # Not enough samples overall or per class for reliable split/stratification
        print(f"Warning: Few samples or classes with < {min_samples_per_class} samples. Using all data for training, accuracy based on training data.")
        X_train, X_test, y_train, y_test = X, [], y, [] # No test set
    else:
        try:
             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
             print(f"Training with {len(X_train)} samples, Testing with {len(X_test)} samples (Stratified).")
        except ValueError as e:
             print(f"Warning: Stratified split failed ({e}). Using non-stratified split.")
             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
             print(f"Training with {len(X_train)} samples, Testing with {len(X_test)} samples (Non-Stratified).")


    print("Training SVM model...")
    # Use probability=True to get confidence scores later
    model = SVC(kernel='linear', probability=True, random_state=42, C=1.0) # C is regularization parameter
    try:
        model.fit(X_train, y_train)
    except Exception as e:
        print(f"Error during model training: {e}")
        return None

    # Evaluate accuracy
    if len(X_test) > 0:
        accuracy = model.score(X_test, y_test)
        print(f"Model accuracy on TEST set: {accuracy:.4f}")
    else:
        # If no test set, show training accuracy (likely optimistic)
        accuracy = model.score(X_train, y_train)
        print(f"Model accuracy on TRAINING set: {accuracy:.4f}")


    return model

# --- Data Management ---

MODEL_FILENAME = 'isl_gesture_model.pkl'
DATA_FILENAME = 'isl_gesture_data.pkl' # Store collected data separately

def save_data(all_data, all_labels):
    """Saves collected data and labels."""
    # Ensure data is serializable (convert numpy arrays if necessary, though pickle handles them)
    try:
        with open(DATA_FILENAME, 'wb') as f:
            pickle.dump({'data': all_data, 'labels': all_labels}, f)
        print(f"Data saved to {DATA_FILENAME}")
    except Exception as e:
        print(f"Error saving data to {DATA_FILENAME}: {e}")

def load_data():
    """Loads collected data and labels."""
    if os.path.exists(DATA_FILENAME):
        try:
            with open(DATA_FILENAME, 'rb') as f:
                saved_data = pickle.load(f)
            print(f"Data loaded from {DATA_FILENAME}")
            # Basic validation
            data = saved_data.get('data', [])
            labels = saved_data.get('labels', [])
            if isinstance(data, list) and isinstance(labels, list):
                 if len(data) == len(labels):
                     return data, labels
                 else:
                     print("Warning: Loaded data and labels have different lengths. Returning empty.")
                     return [], []
            else:
                 print("Warning: Loaded data is not in the expected list format. Returning empty.")
                 return [], []
        except Exception as e:
            print(f"Error loading data from {DATA_FILENAME}: {e}")
            return [], []
    else:
        # print("Data file not found.") # Reduce noise on startup
        return [], []

def save_model(model):
    """Saves the trained model."""
    if model:
        try:
            with open(MODEL_FILENAME, 'wb') as f:
                pickle.dump(model, f)
            print(f"Model saved to {MODEL_FILENAME}")
        except Exception as e:
            print(f"Error saving model to {MODEL_FILENAME}: {e}")

def load_model():
    """Loads the trained model."""
    if os.path.exists(MODEL_FILENAME):
        try:
            with open(MODEL_FILENAME, 'rb') as f:
                model = pickle.load(f)
            print(f"Model loaded from {MODEL_FILENAME}")
            # Check if the loaded object looks like a scikit-learn estimator
            if hasattr(model, 'predict') and hasattr(model, 'score'):
                return model
            else:
                print("Error: Loaded file does not appear to be a valid scikit-learn model.")
                return None
        except Exception as e:
            print(f"Error loading model from {MODEL_FILENAME}: {e}")
            # Optionally delete corrupted file
            # try: os.remove(MODEL_FILENAME) except OSError: pass
            return None
    else:
        # print("Model file not found.") # Reduce noise on startup
        return None

# --- Main Application Logic ---

def main():
    """Main function to handle data collection, training, and recognition."""

    model = None
    all_data, all_labels = [], []
    available_gestures = []
    camera_idx = 0  # Default camera index
    
    # Default timer settings
    settings = {
        'auto_speak_delay': 3.0,  # Default: 3 seconds for auto-speech
        'gesture_stability_threshold': 1.0,  # Default: 1 second for gesture stability
        'initial_delay': 2.0,  # Default: 2 seconds between gestures
        'word_delay': 2.0  # Default: 2 seconds between words
    }

    # --- Initial Setup ---
    print("\n" + "="*30)
    print("  ISL Gesture Recognition")
    print("="*30)

    # Try loading existing data and model first
    loaded_model = load_model()
    loaded_data, loaded_labels = load_data()

    if loaded_data and loaded_labels:
        all_data = loaded_data
        all_labels = loaded_labels
        available_gestures = sorted(list(np.unique(all_labels)))
        print(f"Loaded {len(all_data)} samples for gestures: {', '.join(available_gestures)}")
    elif loaded_data or loaded_labels:
         print("Warning: Inconsistent data loaded (data/labels mismatch). Ignoring loaded data.")
         all_data, all_labels = [], [] # Reset

    if loaded_model:
         # If we loaded data AND a model, assume the model corresponds to the data for now
         # A more robust system might store metadata (e.g., gestures included, feature size) with the model
         if not all_data: # Model exists but no data file - risky, model might be for different data
              print("Warning: Model file exists but no data file found. Model might be outdated or incompatible.")
              proceed = input("Use loaded model anyway? (y/n): ").lower()
              if proceed == 'y':
                  model = loaded_model
                  print("Proceeding with loaded model. Gesture labels might be inaccurate.")
                  # We don't know the gestures this model was trained on. Get them from the model itself.
                  if hasattr(model, 'classes_'):
                       available_gestures = sorted(list(model.classes_))
                       print(f"Model expects gestures: {', '.join(available_gestures)}")
                  else:
                       print("Could not determine gestures from loaded model.")
                       available_gestures = []

              else:
                  loaded_model = None # Discard the loaded model
         else:
              model = loaded_model # Both loaded, assume they match
              # Verify model classes match loaded data labels
              if hasattr(model, 'classes_'):
                    model_gestures = sorted(list(model.classes_))
                    if set(model_gestures) != set(available_gestures):
                         print("Warning: Model classes do not match loaded data labels!")
                         print(f"  Model has: {model_gestures}")
                         print(f"  Data has: {available_gestures}")
                         retrain = input("Retrain model with current data? (y/n): ").lower()
                         if retrain == 'y':
                              model = None # Force retrain later if chosen
                         else:
                              print("Keeping existing model. Predictions might be inconsistent.")
              else:
                    print("Warning: Could not verify classes of loaded model.")


    while True:
        print("\n--- Main Menu ---")
        if available_gestures:
             print(f"Current Gestures: {', '.join(available_gestures)}")
        else:
             print("No gesture data loaded or collected yet.")
        if model:
             print("Model is loaded/trained.")
        else:
             print("Model needs training.")
        print(f"Current camera: {camera_idx}")
        
        # Display current timer settings
        print(f"Auto-speak delay: {settings['auto_speak_delay']} seconds")
        print(f"Gesture recognition time: {settings['gesture_stability_threshold']} seconds")

        print("-" * 17)
        print("1. Collect/Add Gesture Data")
        print("2. Train/Retrain Model")
        print("3. Start Real-time Recognition")
        print("4. View Collected Data Summary")
        print("5. Clear ALL Data and Model")
        print("6. Settings")
        print("q. Quit")
        print("-" * 17)

        choice = input("Enter your choice: ").strip().lower()

        if choice == '1':
            gesture_name = input("Enter name for the gesture (e.g., 'Hello', 'A'): ").strip()
            if not gesture_name:
                print("Gesture name cannot be empty.")
                continue

            # Check if overwriting or adding new
            is_new_gesture = gesture_name not in available_gestures
            if not is_new_gesture:
                 overwrite = input(f"Gesture '{gesture_name}' exists. Overwrite samples? (y/n): ").lower()
                 if overwrite != 'y':
                     print(f"Keeping existing samples for '{gesture_name}'.")
                     continue # Go back to menu if not overwriting
                 else: # Remove existing data for this specific gesture
                     # Create new lists keeping only data for other gestures
                     temp_data = []
                     temp_labels = []
                     for i, label in enumerate(all_labels):
                         if label != gesture_name:
                             temp_data.append(all_data[i])
                             temp_labels.append(label)
                     all_data = temp_data
                     all_labels = temp_labels
                     print(f"Removed existing samples for '{gesture_name}'. Ready to collect new.")
                     # Update available gestures immediately if all samples were removed
                     available_gestures = sorted(list(np.unique(all_labels)))


            # Get number of samples
            while True:
                try:
                    num_samples_str = input(f"Enter number of samples for '{gesture_name}' (e.g., 100): ").strip()
                    if not num_samples_str: # Default if empty
                         num_samples = 100
                         print("Using default: 100 samples.")
                         break
                    num_samples = int(num_samples_str)
                    if num_samples > 0:
                        break
                    else:
                        print("Please enter a positive number.")
                except ValueError:
                    print("Invalid input. Please enter a number.")

            # Collect data with the current camera index
            new_data, new_labels = collect_training_data(gesture_name, num_samples, camera_idx)

            # Process collected data
            if new_data:
                all_data.extend(new_data)
                all_labels.extend(new_labels)
                available_gestures = sorted(list(np.unique(all_labels))) # Update list
                print(f"Finished collecting. Total samples for '{gesture_name}': {np.sum(np.array(all_labels) == gesture_name)}")
                print(f"Total samples overall: {len(all_labels)}")
                save_data(all_data, all_labels) # Save updated data
                model = None # Mark model as needing retraining
                print("-> Model needs retraining (use Option 2).")
            else:
                print(f"No new data was collected for '{gesture_name}'.")


        elif choice == '2':
            if not all_data:
                print("No data available to train. Please collect data first (Option 1).")
                continue
            if len(available_gestures) < 2:
                print("Need data for at least TWO different gestures to train a classifier.")
                continue

            print("Starting model training...")
            temp_model = train_model(all_data, all_labels)
            if temp_model:
                model = temp_model # Update the main model variable
                save_model(model) # Save the newly trained model
                print("Model training complete and saved.")
            else:
                 print("Model training failed.")

        elif choice == '3':
            if not model:
                print("Model not trained or loaded.")
                if all_data and len(available_gestures) >= 2:
                     retrain_now = input("Train model now with available data? (y/n): ").lower()
                     if retrain_now == 'y':
                          print("Starting model training...")
                          temp_model = train_model(all_data, all_labels)
                          if temp_model:
                              model = temp_model
                              save_model(model)
                              print("Model training complete and saved.")
                          else:
                               print("Model training failed. Cannot start recognition.")
                               continue # Back to menu
                     else:
                         print("Please train the model first (Option 2).")
                         continue # Back to menu
                else:
                     print("Insufficient data to train. Please collect more data (Option 1).")
                     continue # Back to menu


            if model: # Check again in case training just finished
                start_recognition(model, camera_idx, settings) # Pass settings to the function
            else:
                 print("Cannot start recognition without a valid model.")


        elif choice == '4':
             if not all_labels:
                 print("No data collected yet.")
             else:
                 print("\n--- Collected Data Summary ---")
                 unique_labels_summary, counts_summary = np.unique(all_labels, return_counts=True)
                 print(f"Total Samples: {len(all_labels)}")
                 print(f"Gestures ({len(unique_labels_summary)}):")
                 for label, count in zip(unique_labels_summary, counts_summary):
                     print(f"  - '{label}': {count} samples")
                 print(f"Expected feature vector size: {FEATURE_VECTOR_SIZE}")


        elif choice == '5':
             confirm = input("WARNING: This will delete saved data ("+DATA_FILENAME+") and model ("+MODEL_FILENAME+").\nAre you sure? (yes/no): ").lower()
             if confirm == 'yes':
                 deleted_files = []
                 try:
                     if os.path.exists(DATA_FILENAME):
                         os.remove(DATA_FILENAME)
                         deleted_files.append(DATA_FILENAME)
                     if os.path.exists(MODEL_FILENAME):
                         os.remove(MODEL_FILENAME)
                         deleted_files.append(MODEL_FILENAME)

                     if deleted_files:
                          print(f"Deleted: {', '.join(deleted_files)}")
                     else:
                          print("No files found to delete.")

                     # Reset runtime variables
                     all_data, all_labels = [], []
                     available_gestures = []
                     model = None
                     print("Runtime data and model cleared.")

                 except OSError as e:
                     print(f"Error deleting files: {e}")
             else:
                 print("Operation cancelled.")

        elif choice == '6':
            print("\n--- Settings ---")
            print("1. Change Camera")
            print("2. Timer Settings")
            print("b. Back to Main Menu")
            
            settings_choice = input("Enter your choice: ").strip().lower()
            
            if settings_choice == '1':
                available_cameras = list_available_cameras()
                if not available_cameras:
                    print("No cameras found!")
                    continue
                
                print("\nAvailable cameras:")
                for cam_idx in available_cameras:
                    print(f"Camera {cam_idx}")
                
                try:
                    new_camera = int(input("\nSelect camera number (0-9): "))
                    if new_camera in available_cameras:
                        camera_idx = new_camera
                        print(f"Camera changed to {camera_idx}")
                    else:
                        print("Invalid camera number. Camera not changed.")
                except ValueError:
                    print("Invalid input. Camera not changed.")
            
            elif settings_choice == '2':
                print("\n--- Timer Settings ---")
                print("Current settings:")
                print(f"1. Auto-speak delay: {settings['auto_speak_delay']} seconds")
                print(f"2. Gesture recognition time: {settings['gesture_stability_threshold']} seconds")
                print(f"3. Delay between gestures: {settings['initial_delay']} seconds")
                print(f"4. Delay between words: {settings['word_delay']} seconds")
                print("b. Back")
                
                timer_choice = input("Select setting to change (1-4): ").strip().lower()
                
                if timer_choice == 'b':
                    continue
                
                try:
                    if timer_choice == '1':
                        new_value = float(input("Enter new auto-speak delay (seconds): "))
                        if new_value > 0:
                            settings['auto_speak_delay'] = new_value
                            print(f"Auto-speak delay set to {new_value} seconds")
                        else:
                            print("Value must be positive. Setting not changed.")
                    
                    elif timer_choice == '2':
                        new_value = float(input("Enter new gesture recognition time (seconds): "))
                        if new_value > 0:
                            settings['gesture_stability_threshold'] = new_value
                            print(f"Gesture recognition time set to {new_value} seconds")
                        else:
                            print("Value must be positive. Setting not changed.")
                    
                    elif timer_choice == '3':
                        new_value = float(input("Enter new delay between gestures (seconds): "))
                        if new_value > 0:
                            settings['initial_delay'] = new_value
                            print(f"Delay between gestures set to {new_value} seconds")
                        else:
                            print("Value must be positive. Setting not changed.")
                    
                    elif timer_choice == '4':
                        new_value = float(input("Enter new delay between words (seconds): "))
                        if new_value > 0:
                            settings['word_delay'] = new_value
                            print(f"Delay between words set to {new_value} seconds")
                        else:
                            print("Value must be positive. Setting not changed.")
                    
                except ValueError:
                    print("Invalid input. Please enter a number.")
            
            elif settings_choice != 'b':
                print("Invalid choice.")
            
            continue  # Return to main menu after settings

        elif choice == 'q':
            print("Exiting program.")
            break

        else:
            print("Invalid choice. Please try again.")

# --- Real-time Recognition ---

def start_recognition(model, camera_idx=0, settings=None):
    """Starts the real-time gesture recognition loop."""
    # Use default settings if none provided
    if settings is None:
        settings = {
            'auto_speak_delay': 3.0,
            'gesture_stability_threshold': 1.0,
            'initial_delay': 2.0,
            'word_delay': 2.0
        }
    
    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_idx}.")
        return

    print("\nStarting Real-time Recognition...")
    print("Press 'q' in the window to stop.")
    print("Sentence Controls:")
    print("  - 'space': Add space between words")
    print("  - 'c': Clear sentence / Apply corrections")
    print("  - 'backspace': Remove last word")
    print("  - 's': Speak current sentence")
    print("  - 'w': Select suggested word")
    print(f"  - Auto-speak after {settings['auto_speak_delay']} seconds of no gesture detection")
    if GEMINI_ENABLED:
        print("Gemini AI is enabled for word suggestions and sentence processing")
    else:
        print("Gemini AI is NOT enabled - set GEMINI_API_KEY to enable")
    time.sleep(1)

    window_name = 'ISL Gesture Recognition'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, SCREEN_WIDTH, SCREEN_HEIGHT)
    cv2.moveWindow(window_name, 0, 0)

    # Get frame dimensions for scaling
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read initial frame.")
        cap.release()
        cv2.destroyAllWindows()
        return
    frame_h, frame_w = frame.shape[:2]

    # Calculate display scaling and offsets
    # Camera view will take up 60% of the width
    camera_width = int(SCREEN_WIDTH * 0.6)
    scale = camera_width / frame_w
    disp_w = int(frame_w * scale)
    disp_h = int(frame_h * scale)
    
    # Center the camera view vertically
    offset_x = 0  # Camera view starts from left
    offset_y = (SCREEN_HEIGHT - disp_h) // 2

    # Text panel configuration
    panel_width = SCREEN_WIDTH - camera_width  # Remaining width for text
    panel_start_x = camera_width  # Start after camera view
    panel_start_y = 0
    panel_end_x = SCREEN_WIDTH
    panel_end_y = SCREEN_HEIGHT
    text_x_start = panel_start_x + 20  # Text starts 20px from panel edge
    text_panel_width = panel_width - 40  # Leave 20px margin on both sides

    # Create a dark brown background for the text panel
    text_panel_bg = np.ones((SCREEN_HEIGHT, panel_width, 3), dtype=np.uint8)
    text_panel_bg[:] = (36, 28, 22)  # BGR for dark brown

    # Colors for text and lines (light beige)
    beige = (210, 200, 170)  # BGR for light beige
    timer_color = (100, 200, 255)  # Orange-yellow for timers
    timer_warning_color = (0, 0, 255)  # Red for timers when close to threshold
    ai_color = (50, 220, 150)  # Green for AI-suggested content
    ai_highlight_color = (50, 255, 200)  # Brighter green for selected AI content

    # Recognition variables - use values from settings
    last_high_conf_prediction = "None"
    last_confidence = 0.0
    display_prediction = "None" 
    confidence_threshold = 0.95 

    # Sentence formation variables
    current_sentence = []  
    current_letters = []   
    last_word_time = time.time()  
    word_delay = settings['word_delay']  # From settings
    last_added_word = None  
    countdown_start = last_word_time  
    
    # Word suggestion variables
    word_suggestions = []
    selected_suggestion_index = -1
    
    # AI processing state
    last_ai_processed_sentence = ""
    spelling_corrections = {}  
    spelling_check_interval = 200  
    
    # Gesture stability detection
    stable_gesture_start = None
    stable_gesture_threshold = settings['gesture_stability_threshold']  # From settings
    last_prediction = None
    prediction_stable = False
    initial_delay = settings['initial_delay']  # From settings
    initial_delay_start = time.time()
    is_initial_delay = True
    
    # Auto-speak after no gesture detection
    last_gesture_time = time.time()
    auto_speak_delay = settings['auto_speak_delay']  # From settings
    last_auto_speak_time = 0
    auto_spoken = False
    
    # Add flag to always show speech timer
    show_speech_timer = True

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Warning: Dropped frame during recognition.")
            continue

        # --- Landmark Extraction ---
        left_landmarks, right_landmarks, _, hands_detected = extract_landmarks(frame)
        
        # Track last time a hand was detected for auto-speak
        current_time = time.time()
        any_hand_detected = any(hands_detected)
        if any_hand_detected:
            last_gesture_time = current_time
            auto_spoken = False  # Reset auto-speak flag when gesture detected
        
        # Calculate time for auto-speech timer (always calculate, not just when hands missing)
        time_since_last_gesture = current_time - last_gesture_time
        time_since_last_auto_speak = current_time - last_auto_speak_time
        time_to_speak = max(0, auto_speak_delay - time_since_last_gesture)
        
        # Check if we should auto-speak due to no gesture detection
        should_auto_speak = (time_since_last_gesture >= auto_speak_delay and 
                            time_since_last_auto_speak >= auto_speak_delay and
                            not auto_spoken and
                            (current_sentence or current_letters))

        if should_auto_speak:
            # If we have current letters, speak just that word
            if current_letters:
                word = ''.join(current_letters)
                # Process and speak just this word
                if GEMINI_ENABLED:
                    processed_word = process_sentence_with_ai(word)
                    speak_text(processed_word)
                else:
                    speak_text(word)
                
                # Add the word to the sentence
                current_sentence.append(word)
                current_letters = []
                word_suggestions = []
                last_auto_speak_time = current_time
                auto_spoken = True
                print(f"Auto-spoke word: {word}")
            # Otherwise speak the full sentence
            elif current_sentence:
                full_text = ' '.join(current_sentence)
                # Process with AI if available
                if GEMINI_ENABLED:
                    processed_text = process_sentence_with_ai(full_text)
                    last_ai_processed_sentence = processed_text
                    speak_text(processed_text)
                else:
                    speak_text(full_text)
                
                last_auto_speak_time = current_time
                auto_spoken = True
                print(f"Auto-spoke sentence: {full_text}")

        # --- Prepare Display Canvas ---
        display_canvas = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
        
        # Add camera view
        resized_frame = cv2.resize(frame, (disp_w, disp_h))
        display_canvas[offset_y:offset_y+disp_h, offset_x:offset_x+disp_w] = resized_frame

        # Add text panel background
        display_canvas[panel_start_y:panel_end_y, panel_start_x:panel_end_x] = text_panel_bg

        # --- Draw Landmarks on Display ---
        display_frame_slice = display_canvas[offset_y:offset_y+disp_h, offset_x:offset_x+disp_w]
        display_slice_copy = display_frame_slice.copy()
        display_slice_rgb = cv2.cvtColor(display_slice_copy, cv2.COLOR_BGR2RGB)
        display_slice_rgb.flags.writeable = False
        results_display = hands.process(display_slice_rgb)
        display_slice_rgb.flags.writeable = True
        if results_display.multi_hand_landmarks:
            for hand_landmarks_disp in results_display.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    display_frame_slice,
                    hand_landmarks_disp,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

        # --- Prediction Logic ---
        feature_vector = None
        current_raw_prediction = "None"
        current_confidence = 0.0

        # Construct feature vector based on detected hands (matching training format)
        if all(hands_detected):
            feature_vector = np.concatenate([left_landmarks, right_landmarks])
        elif hands_detected[0]:
            feature_vector = np.concatenate([left_landmarks, np.zeros(SINGLE_HAND_FEATURE_SIZE)])
        elif hands_detected[1]:
            feature_vector = np.concatenate([np.zeros(SINGLE_HAND_FEATURE_SIZE), right_landmarks])

        if feature_vector is not None:
            if feature_vector.shape[0] == FEATURE_VECTOR_SIZE:
                feature_vector_reshaped = feature_vector.reshape(1, -1)
                try:
                    if hasattr(model, 'predict_proba'):
                        probabilities = model.predict_proba(feature_vector_reshaped)[0]
                        max_prob_index = np.argmax(probabilities)
                        current_raw_prediction = model.classes_[max_prob_index]
                        current_confidence = probabilities[max_prob_index]
                    else:
                        current_raw_prediction = model.predict(feature_vector_reshaped)[0]
                        current_confidence = 1.0

                except Exception as e:
                    print(f"Error during prediction: {e}")
                    current_raw_prediction = "Error"
                    current_confidence = 0.0
            else:
                print(f"Error: Feature vector size mismatch in recognition! Expected {FEATURE_VECTOR_SIZE}, got {feature_vector.shape[0]}")
                current_raw_prediction = "Size Error"
                current_confidence = 0.0

        # --- Gesture Stability Detection ---
        current_time = time.time()
        time_since_last_word = current_time - last_word_time
        countdown = max(0, word_delay - time_since_last_word)

        # Check initial delay
        if is_initial_delay:
            initial_delay_remaining = max(0, initial_delay - (current_time - initial_delay_start))
            if initial_delay_remaining <= 0:
                is_initial_delay = False
                stable_gesture_start = None
                last_prediction = None
                prediction_stable = False

        # Only check for stability if initial delay has passed
        if not is_initial_delay:
            if (current_raw_prediction == last_prediction and 
                current_raw_prediction not in ["Error", "Size Error", "None"] and
                current_confidence >= confidence_threshold):
                
                if stable_gesture_start is None:
                    stable_gesture_start = current_time
                elif current_time - stable_gesture_start >= stable_gesture_threshold:
                    prediction_stable = True
            else:
                stable_gesture_start = None
                prediction_stable = False

            last_prediction = current_raw_prediction

            # Handle letter capture for word building (for single-character gestures)
            if (prediction_stable and 
                time_since_last_word >= word_delay and
                current_raw_prediction != last_added_word and
                len(current_raw_prediction) == 1):  # If it's a single character (letter)
                
                current_letters.append(current_raw_prediction)
                # Update word suggestions if we have Gemini enabled
                if GEMINI_ENABLED and len(current_letters) > 0:
                    context = ' '.join(current_sentence) if current_sentence else None
                    word_suggestions = get_word_suggestions(''.join(current_letters), context)
                
                last_word_time = current_time
                countdown_start = current_time
                last_added_word = current_raw_prediction
                stable_gesture_start = None
                prediction_stable = False
                is_initial_delay = True
                initial_delay_start = current_time
            
            # Handle full word/gesture capture
            elif (prediction_stable and 
                  time_since_last_word >= word_delay and
                  current_raw_prediction != last_added_word and
                  len(current_raw_prediction) > 1):  # If it's a full word or multi-letter gesture
                
                # If we've been building a word with letters, finish it first
                if current_letters:
                    current_sentence.append(''.join(current_letters))
                    current_letters = []
                
                current_sentence.append(current_raw_prediction)
                last_word_time = current_time
                countdown_start = current_time
                last_added_word = current_raw_prediction
                stable_gesture_start = None
                prediction_stable = False
                is_initial_delay = True
                initial_delay_start = current_time
                word_suggestions = []  # Clear suggestions after adding a full word

        if current_confidence >= confidence_threshold:
            display_prediction = current_raw_prediction
            last_confidence = current_confidence
            last_high_conf_prediction = current_raw_prediction
        else:
            display_prediction = "None"
            last_confidence = 0.0
            last_high_conf_prediction = "None"

        # --- Display Information ---
        y_pos = 60  # Start from top of text panel
        line_height = 44
        section_gap = 32
        text_color = beige
        header_color = beige
        line_color = beige
        suggestion_color = (100, 255, 100)  # Green for suggestions
        selected_suggestion_color = (255, 255, 100)  # Yellow for selected suggestion
        font = cv2.FONT_HERSHEY_SIMPLEX

        def draw_text(text, y, size=1.0, color=text_color, bold=False):
            thickness = 3 if bold else 1
            try:
                cv2.putText(display_canvas, text, (text_x_start, y), font, size, color, thickness, cv2.LINE_AA)
            except Exception as e:
                print(f"Error drawing text '{text}': {e}")
            return y + line_height

        def draw_section_line(y):
            cv2.line(display_canvas, (text_x_start, y), (panel_end_x - 20, y), line_color, 2)
            return y + 10

        # Title
        y_pos = draw_text('ISL GESTURE', y_pos, size=1.6, color=header_color, bold=True)
        y_pos = draw_text('RECOGNITION', y_pos, size=1.6, color=header_color, bold=True)
        y_pos += 18

        # Current Sentence Section
        y_pos += 18
        y_pos = draw_text('CURRENT SENTENCE', y_pos + 8, size=1.1, color=header_color, bold=True)
        
        # Format the sentence nicely
        sentence_parts = []
        spelling_warning_positions = []  # Track positions of misspelled words
        
        # Process sentence to identify spelling corrections
        char_position = 0
        for word in current_sentence:
            # Skip spaces
            if word == ' ':
                sentence_parts.append(word)
                char_position += 1
                continue
                
            if word in spelling_corrections:
                # Mark this word as having a spelling suggestion
                spelling_warning_positions.append((char_position, len(word), word, spelling_corrections[word]))
            else:
                # Check for spelling errors in each word as we process it
                # This ensures all words get checked regardless of when they were added
                if GEMINI_ENABLED and len(word) >= 2:
                    correction = check_spelling(word)
                    if correction:
                        spelling_corrections[word] = correction
                        spelling_warning_positions.append((char_position, len(word), word, correction))
            
            sentence_parts.append(word)
            char_position += len(word)
        
        if current_letters:  # If we're in the middle of a word
            typed_word = ''.join(current_letters)
            # Check if we have a correction for current word
            if GEMINI_ENABLED and len(typed_word) > 1:
                # Only check periodically to avoid constant API calls
                current_time_ms = int(time.time() * 1000)
                if current_time_ms % spelling_check_interval < 50:  # More frequent checks
                    correction = check_spelling(typed_word)
                    if correction:
                        spelling_corrections[typed_word] = correction
            
            # Add current typing to display
            if sentence_parts:
                sentence_parts.append(typed_word)
            else:
                sentence_parts = [typed_word]
            
            # Mark position if there's a correction
            if typed_word in spelling_corrections:
                spelling_warning_positions.append((char_position, len(typed_word), typed_word, spelling_corrections[typed_word]))
            
        full_sentence = ''.join(sentence_parts)  # No spaces between parts for accurate position calculation
        sentence_display = full_sentence if full_sentence else "No words yet"
        
        # Calculate the position for sentence display
        sentence_y_pos = y_pos + 8
        y_pos = draw_text(sentence_display, sentence_y_pos, size=1.0, color=text_color, bold=False)
        
        # Display inline spelling corrections directly under the words
        if spelling_warning_positions:
            # Add more vertical space for corrections
            y_pos += 10
            
            # Draw a semi-transparent background for the correction line
            correction_line_y = sentence_y_pos + line_height
            overlay = display_canvas.copy()
            cv2.rectangle(overlay, 
                         (text_x_start - 10, correction_line_y - int(line_height * 0.8)), 
                         (panel_end_x - 20, correction_line_y + int(line_height * 0.5)),
                         (20, 60, 40),  # Dark green background
                         -1)  # Filled rectangle
            alpha = 0.3  # Transparency
            cv2.addWeighted(overlay, alpha, display_canvas, 1 - alpha, 0, display_canvas)
            
            # Draw corrections with connecting lines
            for pos, length, misspelled, correction in spelling_warning_positions:
                # Use more accurate text measurement method
                # Calculate starting X position for this text segment
                text_before = full_sentence[:pos]
                (text_width_before, _) = cv2.getTextSize(text_before, font, 1.0, 1)[0]
                
                # Calculate width of misspelled text
                (text_width_at_pos, _) = cv2.getTextSize(misspelled, font, 1.0, 1)[0]
                
                # Position the correction text
                correction_x = text_x_start + text_width_before
                correction_y = sentence_y_pos + line_height
                
                # Draw the correction text with a more visible background
                # First draw a background rectangle
                (corr_width, corr_height) = cv2.getTextSize(correction, font, 0.8, 1)[0]
                cv2.rectangle(display_canvas,
                            (correction_x - 2, correction_y - corr_height - 2),
                            (correction_x + corr_width + 2, correction_y + 2),
                            (0, 30, 0), -1)  # Dark green filled rectangle
                
                # Draw vertical connecting line
                line_start_x = correction_x + text_width_at_pos / 2
                cv2.line(display_canvas, 
                         (int(line_start_x), sentence_y_pos + 5), 
                         (int(line_start_x), correction_y - corr_height - 4), 
                         ai_highlight_color, 1)
                
                # Draw the correction text
                cv2.putText(display_canvas, correction, 
                           (correction_x, correction_y), 
                           font, 0.8, ai_highlight_color, 1, cv2.LINE_AA)
            
            # Update y_pos to account for the correction line
            y_pos += line_height
            
            # Add instruction to apply corrections
            y_pos += 6
            y_pos = draw_text("Press 'c' to apply all corrections", y_pos, size=0.8, color=ai_highlight_color, bold=True)
        
        # Display spelling correction summary
        if spelling_warning_positions:
            # If there are many corrections, summarize them
            if len(spelling_warning_positions) > 1:
                y_pos += 4
                y_pos = draw_text(f"{len(spelling_warning_positions)} spelling corrections available", 
                                 y_pos, size=0.9, color=ai_highlight_color, bold=True)
        
        # Add AI-processed sentence if available
        if GEMINI_ENABLED and last_ai_processed_sentence and last_ai_processed_sentence != full_sentence:
            y_pos = draw_text("AI Suggestion:", y_pos + 4, size=0.9, color=ai_color, bold=True)
            y_pos = draw_text(last_ai_processed_sentence, y_pos + 4, size=0.9, color=ai_color, bold=False)

        # Current Word Section
        y_pos += 18
        y_pos = draw_text('CURRENT WORD', y_pos + 8, size=1.1, color=header_color, bold=True)
        
        # Show current letters being typed
        if current_letters:
            current_word = ''.join(current_letters)
            y_pos = draw_text(f"Typing: {current_word}", y_pos + 8, size=1.0, color=text_color, bold=True)
        
        # Show current gesture
        y_pos = draw_text(f"Gesture: {display_prediction}", y_pos + 8, size=1.0, color=text_color, bold=True)
        y_pos = draw_text(f'Confidence: {last_confidence:.2f}', y_pos + 4, size=0.9, color=text_color, bold=False)
        
        # Timer for gesture recognition
        if stable_gesture_start and not prediction_stable:
            time_holding = current_time - stable_gesture_start
            time_left = max(0, stable_gesture_threshold - time_holding)
            progress_pct = (time_holding / stable_gesture_threshold) * 100
            
            # Change color to red when close to recognition threshold
            timer_display_color = timer_warning_color if progress_pct > 75 else timer_color
            y_pos = draw_text(f"Gesture Timer: {time_left:.1f}s", y_pos + 4, size=0.9, color=timer_display_color, bold=(progress_pct > 75))
        elif is_initial_delay:
            delay_left = max(0, initial_delay - (current_time - initial_delay_start))
            y_pos = draw_text(f"Reset Delay: {delay_left:.1f}s", y_pos + 4, size=0.9, color=timer_color, bold=False)
        
        # Word Suggestions Section - Enhanced display
        if word_suggestions:
            y_pos += 18
            y_pos = draw_text('AI WORD SUGGESTIONS', y_pos + 8, size=1.1, color=ai_highlight_color, bold=True)
            
            # Draw a box around the suggestions area
            suggestions_start_y = y_pos
            suggestions_height = len(word_suggestions) * line_height + 16
            
            # Create semi-transparent overlay for suggestions area
            overlay = display_canvas.copy()
            cv2.rectangle(overlay, 
                         (text_x_start - 10, suggestions_start_y), 
                         (panel_end_x - 20, suggestions_start_y + suggestions_height),
                         (20, 60, 40),  # Dark green background
                         -1)  # Filled rectangle
            alpha = 0.3  # Transparency
            cv2.addWeighted(overlay, alpha, display_canvas, 1 - alpha, 0, display_canvas)
            
            # Draw border
            cv2.rectangle(display_canvas, 
                         (text_x_start - 10, suggestions_start_y), 
                         (panel_end_x - 20, suggestions_start_y + suggestions_height),
                         ai_color,  # Green border
                         2)  # Border thickness
            
            # Label at top of suggestions
            y_pos += 8
            for i, suggestion in enumerate(word_suggestions):
                suggestion_text = f"{i+1}: {suggestion}" 
                color = selected_suggestion_color if i == selected_suggestion_index else ai_color
                thickness = 2 if i == selected_suggestion_index else 1
                y_pos = draw_text(suggestion_text, y_pos + 4, size=1.0, color=color, bold=(i == selected_suggestion_index))
            
            # Instructions for using suggestions
            y_pos = draw_text("Press 1-3 to select, 'w' to use", y_pos + 4, size=0.8, color=ai_color, bold=False)
            
            # Draw line below suggestions
            y_pos += 4
            cv2.line(display_canvas, (text_x_start - 10, y_pos), 
                    (panel_end_x - 20, y_pos), ai_color, 1)
            y_pos += 8

        # Hand Status Section
        y_pos += 18
        y_pos = draw_text('HAND STATUS', y_pos + 8, size=1.1, color=header_color, bold=True)
        y_pos = draw_text(f'Left: {"Detected" if hands_detected[0] else "Not Detected"}', y_pos + 8, size=1.0, color=text_color, bold=False)
        y_pos = draw_text(f'Right: {"Detected" if hands_detected[1] else "Not Detected"}', y_pos + 4, size=1.0, color=text_color, bold=False)
        
        # Auto-Speech Timer - Always show the timer if we have content
        if (current_sentence or current_letters):
            # Use warning color when close to speaking
            auto_speak_color = timer_warning_color if time_to_speak < 1.0 and not any_hand_detected else timer_color
            
            # When hands are detected, show "Speech Timer Paused", otherwise show countdown
            if any_hand_detected:
                y_pos = draw_text("Speech Timer: Paused", y_pos + 8, size=0.9, color=timer_color, bold=False)
            else:
                y_pos = draw_text(f"Auto-Speak in: {time_to_speak:.1f}s", y_pos + 8, size=0.9, 
                                 color=auto_speak_color, bold=(time_to_speak < 1.0))

        # Controls Section
        y_pos += 18
        y_pos = draw_text('CONTROLS', y_pos + 8, size=1.1, color=header_color, bold=True)
        y_pos = draw_text('q: Quit', y_pos + 8, size=1.0, color=text_color, bold=False)
        y_pos = draw_text('space: Add space', y_pos + 4, size=1.0, color=text_color, bold=False)
        y_pos = draw_text('c: Clear sentence / Apply corrections', y_pos + 4, size=1.0, color=text_color, bold=False)
        y_pos = draw_text('backspace: Remove last word', y_pos + 4, size=1.0, color=text_color, bold=False)
        y_pos = draw_text('s: Speak sentence', y_pos + 4, size=1.0, color=text_color, bold=False)
        if word_suggestions:
            y_pos = draw_text('1-3: Select AI suggestion', y_pos + 4, size=1.0, color=ai_color, bold=False)
            y_pos = draw_text('w: Use selected AI suggestion', y_pos + 4, size=1.0, color=ai_color, bold=False)

        # --- Show Frame ---
        cv2.imshow(window_name, display_canvas)

        # --- Handle Keypress ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Stopping recognition window.")
            break
        elif key == ord(' '):  # Space key
            # If building a word with letters, complete it before adding space
            if current_letters:
                word = ''.join(current_letters)
                
                # Check for spelling errors before adding the word
                if GEMINI_ENABLED and len(word) > 1:
                    correction = check_spelling(word)
                    if correction:
                        spelling_corrections[word] = correction
                
                current_sentence.append(word)
                current_letters = []
                word_suggestions = []
            
            if current_sentence:
                current_sentence.append(' ')  # Add space between words
        
        elif key == ord('c'):  # Clear sentence or apply corrections
            if spelling_corrections and current_sentence:
                # Apply all spelling corrections
                corrected_sentence = []
                for word in current_sentence:
                    if word in spelling_corrections:
                        corrected_sentence.append(spelling_corrections[word])
                    else:
                        corrected_sentence.append(word)
                
                current_sentence = corrected_sentence
                spelling_corrections = {}  # Clear corrections after applying
                print("Applied spelling corrections")
            else:
                # No corrections to apply, so clear the sentence
                current_sentence = []
                current_letters = []
                last_added_word = None
                word_suggestions = []
                last_ai_processed_sentence = ""
                spelling_corrections = {}
        
        elif key == 8:  # Backspace key
            if current_letters:  # If we're typing a word, remove the last letter
                current_letters.pop()
                # Update suggestions if we still have letters
                if current_letters and GEMINI_ENABLED:
                    context = ' '.join(current_sentence) if current_sentence else None
                    word_suggestions = get_word_suggestions(''.join(current_letters), context)
                elif not current_letters:
                    word_suggestions = []
            elif current_sentence:  # Otherwise remove the last word
                current_sentence.pop()
                last_added_word = None
        elif key == ord('s'):  # Speak sentence
            full_text = ' '.join(current_sentence)
            if current_letters:
                full_text += ' ' + ''.join(current_letters)
                
            if full_text:
                # Process with AI if available
                if GEMINI_ENABLED:
                    processed_text = process_sentence_with_ai(full_text)
                    last_ai_processed_sentence = processed_text
                    speak_text(processed_text)
                else:
                    speak_text(full_text)
        elif key in [ord('1'), ord('2'), ord('3')]:  # Select suggestion
            suggestion_idx = key - ord('1')
            if word_suggestions and suggestion_idx < len(word_suggestions):
                selected_suggestion_index = suggestion_idx
        elif key == ord('w'):  # Use selected suggestion
            if selected_suggestion_index >= 0 and selected_suggestion_index < len(word_suggestions):
                # Replace current letters with the selected suggestion
                selected_word = word_suggestions[selected_suggestion_index]
                if current_sentence and current_sentence[-1] == ' ':  # If last element is a space
                    current_sentence.append(selected_word)
                else:
                    current_sentence.append(selected_word)
                current_letters = []
                word_suggestions = []
                selected_suggestion_index = -1

    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()


# --- Entry Point ---
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n--- An Unexpected Error Occurred ---")
        import traceback
        traceback.print_exc()
        print("------------------------------------")
    finally:
        # Ensure mediapipe hands resources are released if an error occurs mid-operation
        if 'hands' in globals() and hasattr(hands, 'close'):
             hands.close()
        print("\nProgram finished.")
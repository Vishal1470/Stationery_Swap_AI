import streamlit as st
import joblib
import numpy as np
from PIL import Image
import cv2
import os
import pandas as pd
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from collections import Counter
import csv
import base64
from io import BytesIO

# Set page configuration
st.set_page_config(
    page_title="Intelligent Stationery Swap & Sell",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1E88E5;
    text-align: center;
    margin-bottom: 2rem;
    font-weight: 800;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
}
.prediction-box {
    background-color: #E3F2FD;
    padding: 20px;
    border-radius: 10px;
    margin: 20px 0;
    border-left: 5px solid #1E88E5;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
.confidence-bar {
    height: 20px;
    background-color: #64B5F6;
    border-radius: 10px;
    margin: 10px 0;
}
.stButton>button {
    background-color: #1E88E5;
    color: white;
    font-weight: bold;
    border-radius: 5px;
    padding: 10px 24px;
    transition: all 0.3s ease;
    border: none;
}
.stButton>button:hover {
    background-color: #0D47A1;
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}
.info-box {
    background-color: #FFF8E1;
    padding: 15px;
    border-radius: 10px;
    margin: 15px 0;
    border-left: 5px solid #FFC107;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
.success-box {
    background-color: #E8F5E9;
    padding: 15px;
    border-radius: 10px;
    margin: 15px 0;
    border-left: 5px solid #4CAF50;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
.warning-box {
    background-color: #FFEBEE;
    padding: 15px;
    border-radius: 10px;
    margin: 15px 0;
    border-left: 5px solid #F44336;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
.sidebar .sidebar-content {
    background-color: #f5f7f9;
}
.card {
    background-color: white;
    border-radius: 10px;
    padding: 15px;
    margin: 10px 0;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
.featured-item {
    transition: transform 0.3s ease;
    border-radius: 10px;
    overflow: hidden;
}
.featured-item:hover {
    transform: scale(1.03);
}
.progress-bar {
    height: 10px;
    border-radius: 5px;
    background-color: #e0e0e0;
    margin: 10px 0;
}
.progress-fill {
    height: 100%;
    border-radius: 5px;
    background-color: #1E88E5;
}
.login-container {
    max-width: 400px;
    margin: 50px auto;
    padding: 30px;
    background-color: transparent;
    border-radius: 10px;
    box-shadow: none;
}
.login-title {
    font-size: 2.5rem;
    color: #1E88E5;
    text-align: center;
    margin-bottom: 2rem;
    font-weight: 700;
}
.login-form {
    background-color: white;
    padding: 25px;
    border-radius: 10px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.1);
}
.main {
    background-color: #f8f9fa;
}
.color-box {
    display: inline-block;
    width: 20px;
    height: 20px;
    border-radius: 4px;
    margin-right: 10px;
    vertical-align: middle;
    border: 1px solid #ddd;
}
.color-palette {
    display: flex;
    gap: 10px;
    margin: 10px 0;
    flex-wrap: wrap;
}
.color-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    margin: 5px;
}
.color-swatch {
    width: 40px;
    height: 40px;
    border-radius: 8px;
    border: 2px solid white;
    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
}
</style>
""", unsafe_allow_html=True)

# User authentication system
def initialize_session_state():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'username' not in st.session_state:
        st.session_state.username = ''
    if 'register_mode' not in st.session_state:
        st.session_state.register_mode = False

# Sample user database (in real app, use a proper database)
users_db = {
    "admin": {"password": "admin123", "name": "Administrator", "email": "admin@example.com"},
    "user1": {"password": "password1", "name": "John Doe", "email": "john@example.com"},
    "user2": {"password": "password2", "name": "Jane Smith", "email": "jane@example.com"}
}

def login_user(username, password):
    if username in users_db and users_db[username]["password"] == password:
        st.session_state.logged_in = True
        st.session_state.username = username
        return True
    return False

def register_user(username, password, name, email):
    if username in users_db:
        return False, "Username already exists"
    users_db[username] = {"password": password, "name": name, "email": email}
    return True, "Registration successful"

def show_login_page():
    # Set page background
    st.markdown("""
    <style>
    .stApp {
        background-color: #f8f9fa;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Login page title
    st.markdown('<h1 class="login-title">📚 Intelligent Stationery Swap & Sell</h1>', unsafe_allow_html=True)
    
    if st.session_state.register_mode:
        # Registration form with transparent background
        st.markdown('<div class="login-form">', unsafe_allow_html=True)
        st.subheader("📝 Create Account")
        
        with st.form("register_form"):
            new_username = st.text_input("Username", key="reg_username")
            new_password = st.text_input("Password", type="password", key="reg_password")
            confirm_password = st.text_input("Confirm Password", type="password", key="reg_confirm")
            name = st.text_input("Full Name", key="reg_name")
            email = st.text_input("Email", key="reg_email")
            
            submitted = st.form_submit_button("Register")
            if submitted:
                if new_password != confirm_password:
                    st.error("Passwords do not match")
                elif len(new_password) < 6:
                    st.error("Password must be at least 6 characters")
                else:
                    success, message = register_user(new_username, new_password, name, email)
                    if success:
                        st.success(message)
                        st.session_state.register_mode = False
                    else:
                        st.error(message)
        
        st.markdown("Already have an account?")
        if st.button("Login instead"):
            st.session_state.register_mode = False
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
            
    else:
        # Login form with transparent background
        st.markdown('<div class="login-form">', unsafe_allow_html=True)
        st.subheader("🔐 Login to Your Account")
        
        with st.form("login_form"):
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            
            submitted = st.form_submit_button("Login")
            if submitted:
                if login_user(username, password):
                    st.success(f"Welcome back, {username}!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")
        
        st.markdown("Don't have an account?")
        if st.button("Create Account"):
            st.session_state.register_mode = True
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# Initialize session state
initialize_session_state()

# Check if user is logged in
if not st.session_state.logged_in:
    show_login_page()
    st.stop()

# Load the trained model and class labels
try:
    model = joblib.load("stationery_model.pkl")
    class_labels = joblib.load("class_labels.pkl")
    st.sidebar.success("✅ Model loaded successfully!")
except:
    st.error("❌ Model files not found. Please train the model first.")
    st.stop()

# Function to extract features from images (same as training)
def extract_image_features(img, resize_dim=(100, 100)):
    try:
        # Convert PIL Image to OpenCV format
        img = np.array(img)
        if len(img.shape) == 2:  # Grayscale image
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Resize
        img = cv2.resize(img, resize_dim)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Extract features (same as training)
        features = []
        
        # 1. Color features (mean of each channel)
        mean_color = np.mean(img, axis=(0, 1))
        features.extend(mean_color)
        
        # 2. Texture features (std deviation)
        std_color = np.std(img, axis=(0, 1))
        features.extend(std_color)
        
        # 3. Edge features (using Canny)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.mean(edges)
        features.append(edge_density)
        
        # 4. Histogram features
        hist = cv2.calcHist([gray], [0], None, [16], [0, 256])
        hist = hist.flatten() / hist.sum()  # Normalize
        features.extend(hist[:5])  # Use first 5 histogram bins
        
        return np.array(features)
        
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None

# Function to detect dominant colors in an image
def get_dominant_colors(image, num_colors=5):
    try:
        # Convert PIL Image to numpy array
        img = np.array(image)
        
        # Convert to RGB if not already
        if len(img.shape) == 2:  # Grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:  # RGBA
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        
        # Resize image for faster processing
        img = cv2.resize(img, (100, 100))
        
        # Reshape the image to be a list of pixels
        pixels = img.reshape(-1, 3)
        
        # Use KMeans to find dominant colors
        kmeans = KMeans(n_clusters=num_colors, n_init=10, random_state=42)
        kmeans.fit(pixels)
        
        # Get the colors and their percentages
        colors = kmeans.cluster_centers_.astype(int)
        counts = np.bincount(kmeans.labels_)
        percentages = (counts / len(pixels)) * 100
        
        # Sort by percentage (descending)
        sorted_indices = np.argsort(percentages)[::-1]
        dominant_colors = []
        
        for idx in sorted_indices:
            if percentages[idx] > 2:  # Only include colors with more than 2% presence
                color = colors[idx]
                hex_color = '#{:02x}{:02x}{:02x}'.format(color[0], color[1], color[2])
                dominant_colors.append({
                    'rgb': color,
                    'hex': hex_color,
                    'percentage': round(percentages[idx], 1)
                })
        
        return dominant_colors[:num_colors]  # Return top colors
        
    except Exception as e:
        st.error(f"Error detecting colors: {str(e)}")
        return None

# Function to get color name from RGB values
def get_color_name(rgb):
    # Simple color mapping
    color_map = {
        (255, 0, 0): "Red",
        (0, 255, 0): "Green",
        (0, 0, 255): "Blue",
        (255, 255, 0): "Yellow",
        (255, 0, 255): "Magenta",
        (0, 255, 255): "Cyan",
        (255, 255, 255): "White",
        (0, 0, 0): "Black",
        (128, 128, 128): "Gray",
        (255, 165, 0): "Orange",
        (128, 0, 128): "Purple",
        (165, 42, 42): "Brown",
        (255, 192, 203): "Pink"
    }
    
    # Find closest color
    min_distance = float('inf')
    closest_color = "Custom"
    
    for map_rgb, name in color_map.items():
        distance = np.sqrt(np.sum((np.array(rgb) - np.array(map_rgb)) ** 2))
        if distance < min_distance:
            min_distance = distance
            closest_color = name
    
    return closest_color

# Function to convert image to base64
def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# Function to save data to CSV (with UTF-8 encoding)
def save_to_csv(data):
    csv_file = "stationery_data.csv"
    file_exists = os.path.isfile(csv_file)
    
    # Use UTF-8 encoding to support special characters
    with open(csv_file, 'a', newline='', encoding='utf-8') as f:
        fieldnames = ['timestamp', 'username', 'filename', 'item_type', 'condition', 
                     'confidence', 'dominant_colors', 'estimated_value', 'image_data']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(data)

# Sidebar with navigation and user info
st.sidebar.markdown(f"### 📚 Intelligent Stationery Swap & Sell")
st.sidebar.markdown(f"### 👋 Welcome, {st.session_state.username}!")

if st.sidebar.button("🚪 Logout", key="logout_btn"):
    st.session_state.logged_in = False
    st.session_state.username = ''
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.title("📚 Navigation")
page = st.sidebar.radio("Go to", ["Home", "Classify Item", "Marketplace", "My Items", "Sustainability Impact", "Data Management"], key="nav_radio")

# Sample data for marketplace (in a real app, this would come from a database)
marketplace_items = [
    {"name": "Premium Notebook", "condition": "Excellent", "type": "Notebook", "value": 150, "owner": "User123", "days_ago": 2},
    {"name": "Artistic Pens Set", "condition": "Good", "type": "Pens", "value": 50, "owner": "ArtLover", "days_ago": 1},
    {"name": "Engineering Scale", "condition": "Excellent", "type": "Scale", "value": 20, "owner": "Engineer99", "days_ago": 3},
    {"name": "Color Pencils", "condition": "Fair", "type": "Pencils", "value": 10, "owner": "CreativeMind", "days_ago": 5},
    {"name": "Journal", "condition": "Good", "type": "Notebook", "value": 200, "owner": "WriterGirl", "days_ago": 1},
    {"name": "Mathematical Instruments", "condition": "Excellent", "type": "Scale", "value": 100, "owner": "MathWiz", "days_ago": 4}
]

# Sample user items
user_items = [
    {"name": "My Notebook", "condition": "Good", "type": "Notebook", "value": 100, "listed": True},
    {"name": "Blue Pen Set", "condition": "Excellent", "type": "Pens", "value": 50, "listed": False},
    {"name": "Old Pencil", "condition": "Fair", "type": "Pencils", "value": 5, "listed": True}
]

# Sample sustainability metrics
sustainability_metrics = {
    "items_saved": 80,
    "co2_saved": 40.5,  # kg
    "waste_reduced": 38.2,  # kg
    "money_saved": 15600  # Rs.
}

# Home page
if page == "Home":
    st.markdown('<h1 class="main-header">📚 Intelligent Stationery Swap & Sell System</h1>', unsafe_allow_html=True)
    
    # Hero section
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ## ♻️ Give Your Stationery a Second Life
        
        Our AI-powered platform helps you classify, value, and trade stationery items 
        to promote sustainability and save money.
        
        - **Smart Classification**: AI identifies items and assesses condition
        - **Fair Valuation**: Get accurate estimates of your items' worth
        - **Easy Trading**: Swap or sell with other eco-conscious users
        - **Track Impact**: See your environmental contribution
        - **Data Management**: All uploaded images are stored in CSV format for future reference
        """)
        
        st.button("🚀 Get Started", key="get_started_btn")
        
    with col2:
        st.image("https://cdn.pixabay.com/photo/2017/08/11/00/43/office-2629494_1280.png", 
                use_container_width=True)
    
    # Stats section
    st.markdown("---")
    st.subheader("📈 Our Community Impact")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class>', unsafe_allow_html=True)
        st.metric("Items Traded", "1,247", "+23 this week")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class>', unsafe_allow_html=True)
        st.metric("CO₂ Saved", "342 kg", "Equivalent to 15 trees")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class>', unsafe_allow_html=True)
        st.metric("Money Saved", "5,892 Rs.", "For our community")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class>', unsafe_allow_html=True)
        st.metric("Active Users", "893", "+12 today")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Featured items
    st.markdown("---")
    st.subheader("🔥 Featured Items")
    
    cols = st.columns(3)
    for idx, item in enumerate(marketplace_items[:3]):
        with cols[idx]:
            st.markdown(f'<div class="featured-item">', unsafe_allow_html=True)
            st.image(f"https://picsum.photos/300/200?random={idx}", use_container_width=True)
            st.subheader(item['name'])
            st.caption(f"Condition: {item['condition']} • {item['type']}")
            st.write(f"**Value:** {item['value']} Rs.")
            
            # Condition progress bar
            condition_value = 80 if item['condition'] == 'Excellent' else 60 if item['condition'] == 'Good' else 40
            st.markdown(f'<div class="progress-bar"><div class="progress-fill" style="width: {condition_value}%;"></div></div>', unsafe_allow_html=True)
            
            st.button("Make Offer", key=f"offer_{idx}")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # How it works
    st.markdown("---")
    st.subheader("🔄 How It Works")
    
    steps = st.columns(4)
    with steps[0]:
        st.markdown("""
        ### 1. Upload
        Take a clear photo of your stationery item
        """)
        st.image("https://cdn-icons-png.flaticon.com/512/3342/3342137.png", width=80)
    
    with steps[1]:
        st.markdown("""
        ### 2. Classify
        Our AI identifies and assesses the item
        """)
        st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=80)
    
    with steps[2]:
        st.markdown("""
        ### 3. Value
        Get a fair estimate of your item's worth
        """)
        st.image("https://cdn-icons-png.flaticon.com/512/3474/3474362.png", width=80)
    
    with steps[3]:
        st.markdown("""
        ### 4. Trade
        Swap or sell with our community
        """)
        st.image("https://cdn-icons-png.flaticon.com/512/1001/1001375.png", width=80)

# Classify Item page
elif page == "Classify Item":
    st.markdown('<h1 class="main-header">🔍 Classify Stationery Item</h1>', unsafe_allow_html=True)
    
    with st.expander("ℹ️ Classification Guide", expanded=True):
        st.markdown("""
        **For best results:**
        - Use clear, well-lit images
        - Photograph against a plain background
        - Capture the entire item
        - Show any wear or damage clearly
        
        **Supported items:** Notebooks, Pens, Pencils, Scales etc.
        
        **Note:** All uploaded images and their data will be stored in a CSV file for future reference.
        """)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload a stationery item image", 
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG"
    )

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)
        
        with col2:
            # Extract features and predict
            with st.spinner("Analyzing image..."):
                features = extract_image_features(image)
                
                if features is not None:
                    # Make prediction
                    prediction = model.predict([features])
                    predicted_class = prediction[0]
                    
                    # Get prediction probabilities
                    if hasattr(model, 'predict_proba'):
                        probabilities = model.predict_proba([features])[0]
                        confidence = max(probabilities) * 100
                        class_index = np.argmax(probabilities)
                    else:
                        confidence = 70  # Default confidence for models without probabilities
                        class_index = 0
                    
                    # Display results
                    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                    st.markdown("### 📊 Classification Results")
                    
                    # Split class into item and condition
                    if '_' in predicted_class:
                        item_type, condition = predicted_class.split('_')
                        st.markdown(f'<div class="success-box">', unsafe_allow_html=True)
                        st.success(f"**Item Type:** {item_type}")
                        st.success(f"**Condition:** {condition.capitalize()}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="success-box">', unsafe_allow_html=True)
                        st.success(f"**Prediction:** {predicted_class}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Confidence indicator
                    st.markdown(f"**Confidence:** {confidence:.2f}%")
                    
                    # Color code confidence level
                    if confidence > 80:
                        bar_color = "#4CAF50"  # Green
                    elif confidence > 60:
                        bar_color = "#FF9800"  # Orange
                    else:
                        bar_color = "#F44336"  # Red
                    
                    st.markdown(f'<div class="progress-bar"><div class="progress-fill" style="width: {confidence}%; background-color: {bar_color};"></div></div>', unsafe_allow_html=True)
                    
                    # Display appropriate emoji based on condition
                    condition_emojis = {
                        "excellent": "⭐⭐⭐",
                        "good": "⭐⭐", 
                        "fair": "⭐"
                    }
                    
                    condition = predicted_class.split('_')[-1] if '_' in predicted_class else ""
                    if condition in condition_emojis:
                        st.markdown(f"**Quality Rating:** {condition_emojis[condition]}")
                    
                    # Value estimation based on type and condition
                    value_ranges = {
                        "Notebook": {"excellent": "100-150", "good": "50-100", "fair": "20-50"},
                        "Pen": {"excellent": "30-60", "good": "10-30", "fair": "5-10"},
                        "Pencil": {"excellent": "20-40", "good": "10-20", "fair": "5-10"},
                        "Scale": {"excellent": "40-80", "good": "20-40", "fair": "10-20"}
                    }
                    
                    estimated_value = ""
                    if item_type in value_ranges and condition in value_ranges[item_type]:
                        value_range = value_ranges[item_type][condition]
                        estimated_value = f"Rs. {value_range}"
                        st.markdown(f"**Estimated Value:** {estimated_value}")
                    
                    # Color detection section
                    st.markdown("### 🎨 Color Analysis")
                    with st.spinner("Detecting colors..."):
                        dominant_colors = get_dominant_colors(image)
                        
                        if dominant_colors:
                            st.markdown("**Dominant Colors:**")
                            
                            # Create color palette display
                            st.markdown('<div class="color-palette">', unsafe_allow_html=True)
                            for color_info in dominant_colors:
                                color_name = get_color_name(color_info['rgb'])
                                st.markdown(f"""
                                <div class="color-item">
                                    <div class="color-swatch" style="background-color: {color_info['hex']};"></div>
                                    <div style="font-size: 12px; margin-top: 5px;">
                                        {color_name}<br>
                                        {color_info['percentage']}%
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.info("Could not detect dominant colors in this image.")
                    
                    # Save data to CSV
                    try:
                        # Convert image to base64
                        image_data = image_to_base64(image)
                        
                        # Prepare data for CSV
                        csv_data = {
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'username': st.session_state.username,
                            'filename': uploaded_file.name,
                            'item_type': item_type if '_' in predicted_class else predicted_class,
                            'condition': condition if '_' in predicted_class else "Unknown",
                            'confidence': f"{confidence:.2f}%",
                            'dominant_colors': ", ".join([f"{get_color_name(color['rgb'])} ({color['percentage']}%)" for color in dominant_colors]) if dominant_colors else "None detected",
                            'estimated_value': estimated_value,
                            'image_data': image_data
                        }
                        
                        # Save to CSV
                        save_to_csv(csv_data)
                        st.success("✅ Data saved to CSV successfully!")
                        
                    except Exception as e:
                        st.error(f"Error saving data to CSV: {str(e)}")
                    
                    # Suggested action based on condition
                    st.markdown("### 💡 Recommendations")
                    
                    if condition == "excellent":
                        st.markdown('<div class="info-box">', unsafe_allow_html=True)
                        st.info("""
                        **Premium Value Item!**
                        - Perfect for swapping or selling at premium value
                        - Consider asking for higher-value items in exchange
                        - Great condition attracts more potential swappers
                        """)
                        st.markdown('</div>', unsafe_allow_html=True)
                    elif condition == "good":
                        st.markdown('<div class="info-box">', unsafe_allow_html=True)
                        st.info("""
                        **Good Value Item**
                        - Suitable for fair swapping or reasonable selling price
                        - Good condition appeals to most users
                        - Consider combining with other items for better swaps
                        """)
                        st.markdown('</div>', unsafe_allow_html=True)
                    elif condition == "fair":
                        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                        st.warning("""
                        **Needs Attention**
                        - Consider refurbishing before swapping
                        - Best for swapping with similar condition items
                        - Might need to combine with other items for better value
                        """)
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.info("Item condition recognized. Consider its appropriate value for swapping.")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Action buttons
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("🔄 List for Swap", use_container_width=True, key="swap_btn"):
                            st.success("Item listed for swapping!")
                    with col2:
                        if st.button("💰 List for Sale", use_container_width=True, key="sale_btn"):
                            st.success("Item listed for sale!")
                    
                    # Show all class probabilities if available
                    if hasattr(model, 'predict_proba') and st.checkbox("Show detailed probabilities", key="prob_check"):
                        st.markdown("#### 📈 Detailed Predictions")
                        probs_dict = {class_labels[i]: float(probabilities[i]) * 100 for i in range(len(class_labels))}
                        
                        # Sort by probability
                        sorted_probs = sorted(probs_dict.items(), key=lambda x: x[1], reverse=True)
                        
                        for class_name, prob in sorted_probs:
                            if prob > 5:  # Only show probabilities above 5%
                                st.write(f"{class_name.replace('_', ' ').title()}: {prob:.2f}%")
                else:
                    st.error("Could not process the image. Please try another image.")

# Marketplace page
elif page == "Marketplace":
    st.markdown('<h1 class="main-header">🛍️ Stationery Marketplace</h1>', unsafe_allow_html=True)
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        item_filter = st.selectbox("Item Type", ["All", "Notebook", "Pen", "Pencil", "Scale"], key="item_filter")
    with col2:
        condition_filter = st.selectbox("Condition", ["All", "Excellent", "Good", "Fair"], key="condition_filter")
    with col3:
        sort_by = st.selectbox("Sort By", ["Newest", "Best Value", "Highest Rated"], key="sort_filter")
    
    # Marketplace items
    st.subheader("Available Items")
    
    filtered_items = marketplace_items
    if item_filter != "All":
        filtered_items = [item for item in filtered_items if item["type"] == item_filter]
    if condition_filter != "All":
        filtered_items = [item for item in filtered_items if item["condition"] == condition_filter]
    
    # Display items in a grid
    cols = st.columns(3)
    for idx, item in enumerate(filtered_items):
        with cols[idx % 3]:
            st.markdown(f'<div class>', unsafe_allow_html=True)
            st.image(f"https://picsum.photos/300/200?random={idx+10}", use_container_width=True)
            st.subheader(item['name'])
            st.caption(f"By {item['owner']} • {item['days_ago']} days ago")
            st.write(f"**Type:** {item['type']}")
            st.write(f"**Condition:** {item['condition']}")
            st.write(f"**Value:** {item['value']} Rs.")
            
            # Condition indicator
            condition_progress = {
                "Excellent": 90,
                "Good": 60,
                "Fair": 30
            }
            progress_value = condition_progress.get(item['condition'], 50)
            st.markdown(f'<div class="progress-bar"><div class="progress-fill" style="width: {progress_value}%;"></div></div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.button("Swap", key=f"swap_{idx}", use_container_width=True)
            with col2:
                st.button("Buy", key=f"buy_{idx}", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

# My Items page
elif page == "My Items":
    st.markdown('<h1 class="main-header">📦 My Stationery Items</h1>', unsafe_allow_html=True)
    
    # Stats
    total_items = len(user_items)
    listed_items = sum(1 for item in user_items if item['listed'])
    total_value = sum(item['value'] for item in user_items if item['listed'])
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Items", total_items)
    with col2:
        st.metric("Listed Items", listed_items)
    with col3:
        st.metric("Total Value", f"{total_value} Rs.")
    
    # User's items
    st.subheader("Your Inventory")
    
    for idx, item in enumerate(user_items):
        with st.expander(f"{item['name']} - {item['type']} ({item['condition']})", expanded=True):
            col1, col2 = st.columns([1, 3])
            with col1:
                st.image(f"https://picsum.photos/200/150?random={idx+20}", use_container_width=True)
            with col2:
                st.write(f"**Condition:** {item['condition']}")
                st.write(f"**Estimated Value:** {item['value']} Rs.")
                st.write(f"**Listed:** {'Yes' if item['listed'] else 'No'}")
                
                if item['listed']:
                    st.success("✅ Currently listed on marketplace")
                    if st.button("Remove Listing", key=f"remove_{idx}"):
                        st.warning("Listing removed")
                else:
                    st.warning("Not listed on marketplace")
                    if st.button("List Item", key=f"list_{idx}"):
                        st.success("Item listed on marketplace!")
    
    # Add new item
    st.subheader("Add New Item")
    with st.form("add_item_form"):
        col1, col2 = st.columns(2)
        with col1:
            item_name = st.text_input("Item Name", key="item_name")
            item_type = st.selectbox("Item Type", ["Notebook", "Pen", "Pencil", "Scale", "Other"], key="item_type")
        with col2:
            item_condition = st.selectbox("Condition", ["Excellent", "Good", "Fair"], key="item_condition")
            item_value = st.slider("Estimated Value (Rs.)", 10, 50, 100, key="item_value")
        
        submitted = st.form_submit_button("Add Item")
        if submitted:
            st.success(f"Added {item_name} to your inventory!")

# Sustainability Impact page
elif page == "Sustainability Impact":
    st.markdown('<h1 class="main-header">🌱 Sustainability Impact</h1>', unsafe_allow_html=True)
    
    # Personal impact
    st.subheader("Your Environmental Contribution")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Items Saved", sustainability_metrics['items_saved'])
    with col2:
        st.metric("CO₂ Saved", f"{sustainability_metrics['co2_saved']} kg")
    with col3:
        st.metric("Waste Reduced", f"{sustainability_metrics['waste_reduced']} kg")
    with col4:
        st.metric("Money Saved", f"{sustainability_metrics['money_saved']} Rs.")
    
    # Impact visualization - Using Streamlit's native line chart instead of Plotly
    st.subheader("Impact Over Time")
    
    # Sample data for chart
    dates = [datetime.now() - timedelta(days=i) for i in range(30, 0, -1)]
    items_saved = np.cumsum(np.random.randint(1, 5, size=30))
    
    # Create a DataFrame for the chart
    chart_data = pd.DataFrame({
        'Date': dates,
        'Items Saved': items_saved
    })
    
     # Display line chart using Streamlit's native chart
    st.line_chart(chart_data.set_index('Date'))
    
    # Environmental facts
    st.subheader("Did You Know?")
    
    facts = [
        "📊 Producing a single notebook generates approximately 1.2 kg of CO₂ emissions",
        "💧 Manufacturing one pen uses about 13 gallons of water",
        "🌳 Recycling stationery items can reduce carbon footprint by up to 70%",
        "💰 The average student spends 500-1000 Rs. on stationery each year",
        "🔄 Swapping instead of buying new extends the life of products by 2-3 years"
    ]
    
    for fact in facts:
        st.info(fact)
    
    # Community leaderboard
    st.subheader("Community Leaders")
    
    leaders = [
        {"name": "EcoWarrior22", "items": 47, "impact": 125},
        {"name": "GreenThumb", "items": 42, "impact": 118},
        {"name": "SustainableSally", "items": 38, "impact": 105},
        {"name": "PlanetProtector", "items": 35, "impact": 98},
        {"name": "RecycleRick", "items": 32, "impact": 92}
    ]
    
    for i, leader in enumerate(leaders):
        st.markdown(f"{i+1}. **{leader['name']}** - {leader['items']} items saved, {leader['impact']} Rs. impact")

# Data Management page
elif page == "Data Management":
    st.markdown('<h1 class="main-header">📊 Data Management</h1>', unsafe_allow_html=True)
    
    st.subheader("Uploaded Stationery Data")
    
    # Check if CSV file exists
    csv_file = "stationery_data.csv"
    if os.path.exists(csv_file):
        # Load and display data with UTF-8 encoding
        try:
            df = pd.read_csv(csv_file, encoding='utf-8')
        except UnicodeDecodeError:
            # If UTF-8 fails, try other encodings
            try:
                df = pd.read_csv(csv_file, encoding='latin-1')
            except:
                st.error("Could not read the CSV file with available encodings.")
                df = pd.DataFrame()
        
        if not df.empty:
            st.dataframe(df)
            
            # Show statistics
            st.subheader("Data Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", len(df))
            with col2:
                st.metric("Unique Users", df['username'].nunique())
            with col3:
                st.metric("Most Common Item", df['item_type'].mode()[0] if not df.empty else "N/A")
            
            # Download button
            st.download_button(
                label="Download CSV",
                data=df.to_csv(index=False),
                file_name="stationery_data.csv",
                mime="text/csv"
            )
            
            # Clear data button
            if st.button("Clear All Data", key="clear_data"):
                if os.path.exists(csv_file):
                    os.remove(csv_file)
                    st.success("All data has been cleared!")
                    st.rerun()
        else:
            st.info("No data available in the CSV file.")
    else:
        st.info("No data available yet. Upload some images on the Classify Item page to get started.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Intelligent Stationery Swap & Sell System © 2025 | "
    "</div>", 
    unsafe_allow_html=True
)
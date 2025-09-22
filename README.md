# Intelligent Stationery Swap & Sell System
An AI-powered web application for classifying, valuing, and trading stationery items to promote sustainability and reduce waste.

# 🌟 Features
Smart Image Classification: AI-powered recognition of stationery items and condition assessment

Color Analysis: Automatic detection of dominant colors in uploaded images

Valuation System: Estimated pricing based on item type and condition

Marketplace: Platform for swapping or selling stationery items

Sustainability Tracking: Environmental impact metrics and reporting

User Authentication: Secure login and registration system

Data Management: CSV export of all classified items with image data

# 🛠️ Technologies Used
Frontend: Streamlit

Machine Learning: Scikit-learn, OpenCV, K-Means clustering

Image Processing: OpenCV, PIL

Data Handling: Pandas, NumPy

Authentication: Custom user management system

Data Storage: CSV with image encoding

# 📦 Installation
Clone the repository:

bash
git clone https://github.com/yourusername/stationery-swap-sell.git
cd stationery-swap-sell
Create a virtual environment:

bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies:

bash
pip install -r requirements.txt
📋 Requirements
The application requires the following Python packages:

streamlit

scikit-learn

opencv-python

pillow

numpy

pandas

joblib

# 🚀 Usage
Train the model (if not already trained):

bash
python train_model.py
Run the application:

bash
streamlit run app.py
Open your browser and navigate to http://localhost:8501

# 📁 Project Structure
text
stationery-swap-sell/
├── app.py # Main application file

├── train_model.py         # Model training script

├── stationery_model.pkl   # Trained model (generated after training)

├── class_labels.pkl       # Class labels (generated after training)

├── stationery_data.csv    # Data storage (generated during use)

├── requirements.txt       # Python dependencies

└── README.md             # Project documentation

# 🎯 How It Works
Upload Images: Users upload images of stationery items

AI Classification: The system classifies items and assesses condition

Color Analysis: Dominant colors are detected and displayed

Valuation: Items are given estimated market values

Trading: Users can list items for swap or sale

Impact Tracking: Environmental savings are calculated and displayed

# 🔧 Model Training
To train the classification model:

Prepare a dataset of stationery images organized by category and condition

Run the training script:

bash
python train_model.py
The script will generate stationery_model.pkl and class_labels.pkl files

# 🤝 Contributing
We welcome contributions! Please feel free to submit pull requests or open issues for bugs and feature requests.

# 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

# 🙏 Acknowledgments
Icons and images from Flaticon and Pixabay

Built with Streamlit and Scikit-learn

Inspired by sustainability initiatives in educational environments

# 📞 Support
For support or questions, please open an issue in the GitHub repository or contact the development team.


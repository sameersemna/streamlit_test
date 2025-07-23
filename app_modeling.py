import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import json
import models
from common import custom_stopwords, display_paired_images_in_reports_folder, prdtypes, prdtypes_en, select_h5_file, word_grouping
from functions import display_html_file, show_pdf_page
import keras
from sklearn.metrics import confusion_matrix, classification_report

import cv2
from PIL import Image
import random
import glob
from rembg import remove
import warnings
import io

warnings.filterwarnings('ignore')



from image_preprocessing import (load_original_image, get_random_image_path, baseline_preprocessing, 
        advanced_augmentation_preprocessing, background_removal_preprocessing, smart_crop_preprocessing)


st.title("🖼️ Image Preprocessing Methods Demo")


st.markdown("### 📊 Basic Image Properties")

col1, col2 = st.columns(2)

with col1:
    st.metric("Total Images", "84,916")
    st.metric("Missing Images", "0")
    st.metric("Average Sharpness", "671.93")
    st.metric("Sharp Images (%)", "86%")

with col2:
    st.metric("Average Brightness", "43.47")
    st.metric("Bright Images (%)", "35%")
    st.metric("Average Contrast", "62.93")


st.markdown("*Compare 4 different image preprocessing approaches*")

# Sidebar controls
st.sidebar.header("Controls")

# Random image button
if st.sidebar.button("🎲 Select Random Image", type="primary"):
    st.session_state.current_image = get_random_image_path()
    st.session_state.processed_images = None

# Initialize session state
if 'current_image' not in st.session_state:
    st.session_state.current_image = get_random_image_path()
if 'processed_images' not in st.session_state:
    st.session_state.processed_images = None

# Display current image info
st.info(f"**Current Image:** {os.path.basename(st.session_state.current_image)}")

# Show original image continuously at the top
if st.session_state.current_image:
    original = load_original_image(st.session_state.current_image)
    if original is not None:
        st.subheader("📸 Original Image")
        
        # Make the original image bigger by using columns
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.image(original, caption=f"Current: {os.path.basename(st.session_state.current_image)}", use_container_width=True)

# Method descriptions - always visible as dropdown
st.markdown("---")
st.subheader("🔍 Method Descriptions")

with st.expander("📖 Click to learn about each method"):
    st.markdown("""
    **⚡ Baseline Preprocessing:**
    - Maintains aspect ratio with intelligent scaling
    - Creates uniform 500×500 output with white padding
    - RGB conversion and normalization to [0,1]
    
    **🎭 Background Removed:**
    - Uses AI-based rembg library for automatic background removal
    - Replaces background with white
    
    **✂️ Smart Crop:**
    - Uses edge detection and contour analysis
    - Automatically crops to product boundaries
    - Applies histogram equalization for better contrast
    
    **🌟 Advanced Augmentation:**
    - Multi-stage enhancement pipeline
    - Fast Non-Local Means Denoising
    - CLAHE (Contrast Limited Adaptive Histogram Equalization)
    - Custom sharpening kernel and color balance
    """)

# Show process instruction if not processed yet
if not st.session_state.processed_images:
    st.info("👆 Click 'Process Image' to see the preprocessing results!")

# Process images button
if st.sidebar.button("🔄 Process Image", type="secondary"):
    with st.spinner("Processing image with all methods... This may take a moment."):
        
        if original is not None:
            # Process with all methods
            baseline = baseline_preprocessing(st.session_state.current_image)
            background_removed = background_removal_preprocessing(st.session_state.current_image)
            smart_crop = smart_crop_preprocessing(st.session_state.current_image)
            advanced = advanced_augmentation_preprocessing(st.session_state.current_image)
            
            # Store in session state
            st.session_state.processed_images = {
                'baseline': baseline,
                'background_removed': background_removed,
                'smart_crop': smart_crop,
                'advanced': advanced
            }
            
            st.success("✅ Image processed successfully!")
        else:
            st.error("Failed to load the selected image.")

# Display results
if st.session_state.processed_images:
    st.markdown("---")
    st.subheader("📊 Processing Results")
    
    # Create 4 columns for the processed images
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)
    
    
    columns = [col1, col2, col3, col4]
    methods = [
        ('baseline', '⚡ Baseline'),
        ('background_removed', '🎭 Background Removed'),
        ('smart_crop', '✂️ Smart Crop'),
        ('advanced', '🌟 Advanced Augmentation')
    ]
    
    for i, (method, title) in enumerate(methods):
        with columns[i]:
            st.markdown(f"**{title}**")
            
            img = st.session_state.processed_images[method]
            if img is not None:
                # Convert to display format if needed
                if img.dtype == np.float32:
                    display_img = (img * 255).astype(np.uint8)
                else:
                    display_img = img
                
                st.image(display_img, use_container_width=True)
                
                # Show image stats
                st.caption(f"Shape: {img.shape}")
                if img.dtype == np.float32:
                    st.caption(f"Range: [{img.min():.3f}, {img.max():.3f}]")
                else:
                    st.caption(f"Range: [{img.min()}, {img.max()}]")
            else:
                st.error("Failed to process with this method")

else:
    st.markdown("---")


st.header("Modeling")

text_model = "text-cnn-epochs-100-lr-0.001-testing.keras"
text_history = "text-cnn-epochs-100-lr-0.001-testing_history.json"
text_report = 'training_text-cnn-epochs-100-lr-0.001-testing_f1-0.7257_t-0721_1642.pdf'

img_model = ''
img_history = ''

multimodal_model = "multimodal-text-mobilenetv2-epochs-100-lr-0.0005_valf1-0.762.keras"
multimodal_history = "multimodal-text-mobilenetv2-epochs-100-lr-0.0005_valf1-0.762_history.json"
multimodal_report = 'training_multimodal-text-mobilenetv2-epochs-100-lr-0.0005_valf1-0.762_f1-0.7620_t-0720_0148.pdf'
# Model selection dropdown
model_options = [
    "Select a model",
    "Basic ML Models", 
    "Custom CNN Text",
    "MobileNet for Images", 
    "Multimodal Fusion Model"
]

selected_model = st.selectbox("Choose a model:", model_options)

if selected_model == "Basic ML Models":
    st.subheader("Basic ML Models")
    display_html_file('basic_models.html')
    
elif selected_model == "Custom CNN Text":
    st.subheader("Custom CNN Text Model")
    
    # Load specific model files
    model_file = text_model
    history_file = text_history
    
    model_path = f"models/{model_file}"
    history_path = f"models/{history_file}"
    
    st.info(f"Loading model: {model_file}")
    st.info(f"Loading history: {history_file}")
    
    # Load Keras model
    try:
        model = keras.models.load_model(model_path)
        st.success(f"Complete Keras model loaded from: {model_path}")
    except Exception as e:
        st.error(f"Error loading Keras model: {e}")
        st.stop()
    
    # Load history from JSON
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history_data = json.load(f)
        st.success(f"Training history loaded from {history_path}")
        
        # Display model info
        model_name = "Custom CNN Text Model"
        st.title(f"CNN Text Model Results: {model_name}")
        
        # Create visualization
        fig = plt.figure(figsize=(12, 8))
        
        # Loss curve
        plt.subplot(2, 2, 1)
        plt.plot(history_data['loss'], label='Train Loss', linewidth=2)
        plt.plot(history_data['val_loss'], label='Val Loss', linewidth=2)
        plt.title('Training Loss', fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # F1 Score curve (main focus)
        plt.subplot(2, 2, 2)
        if 'f1_score' in history_data:
            plt.plot(history_data['f1_score'], label='Train F1 Macro', linewidth=2)
            plt.plot(history_data['val_f1_score'], label='Val F1 Macro', linewidth=2)
            plt.title('F1 Score (Macro) - Primary Metric', fontweight='bold')
            plt.legend()
        else:
            plt.text(0.5, 0.5, 'F1 Macro not tracked', ha='center', va='center')
            plt.title('F1 Score (N/A)')
        plt.grid(True, alpha=0.3)
        
        # Learning Rate (if available)
        plt.subplot(2, 2, 3)
        if 'learning_rate' in history_data:
            plt.plot(history_data['learning_rate'], linewidth=2, color='red')
            plt.title('Learning Rate', fontweight='bold')
            plt.yscale('log')
        else:
            plt.text(0.5, 0.5, 'LR not tracked', ha='center', va='center')
            plt.title('Learning Rate (N/A)')
        plt.grid(True, alpha=0.3)
        
        # Model summary info
        plt.subplot(2, 2, 4)
        plt.axis('off')
        
        final_val_f1 = history_data.get('val_f1_score', ['N/A'])[-1] if 'val_f1_score' in history_data else 'N/A'
        final_train_f1 = history_data.get('f1_score', ['N/A'])[-1] if 'f1_score' in history_data else 'N/A'
        final_val_loss = history_data['val_loss'][-1]
        epochs = len(history_data['loss'])
        
        summary_text = f"""MODEL SUMMARY

Epochs: {epochs}
Final Train F1 Macro: {final_train_f1 if final_train_f1 != 'N/A' else 'N/A'}
Final Val F1 Macro: {final_val_f1 if final_val_f1 != 'N/A' else 'N/A'}
Final Val Loss: {final_val_loss:.3f}

Parameters: {model.count_params():,}
Architecture: Custom CNN Text
"""
        
        plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.suptitle(f'Training Results: {model_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        
        # Show detailed metrics
        st.subheader("Training History Details")
        with st.expander("View Raw Training History"):
            st.json(history_data)
            
    else:
        st.error(f"Training history file not found: {history_path}")

    pdf_path = 'models/'+ text_report
    show_pdf_page(pdf_path, pnum=2)
        
elif selected_model == "Multimodal Fusion Model":
    st.subheader("Multimodal Fusion Model")

    display_html_file('multimodal.html')
    display_html_file('technic.html')
    # Load specific multimodal model files
    model_file = multimodal_model
    history_file = multimodal_history
    
    model_path = f"models/{model_file}"
    history_path = f"models/{history_file}"
    
    st.info(f"Loading model: {model_file}")
    st.info(f"Loading history: {history_file}")
    
    # Load Keras model
    try:
        model = keras.models.load_model(model_path)
        st.success(f"Complete Keras model loaded from: {model_path}")
    except Exception as e:
        st.error(f"Error loading Keras model: {e}")
        st.stop()
    
    # Load history from JSON
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history_data = json.load(f)
        st.success(f"Training history loaded from {history_path}")
        
        # Display model info
        model_name = "Multimodal Fusion Model (Text + MobileNetV2)"
        st.title(f"Multimodal Model Results: {model_name}")
        
        # Create visualization
        fig = plt.figure(figsize=(12, 8))
        
        # Loss curve
        plt.subplot(2, 2, 1)
        plt.plot(history_data['loss'], label='Train Loss', linewidth=2)
        plt.plot(history_data['val_loss'], label='Val Loss', linewidth=2)
        plt.title('Training Loss', fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # F1 Score curve (main focus)
        plt.subplot(2, 2, 2)
        if 'f1_score' in history_data:
            plt.plot(history_data['f1_score'], label='Train F1 Macro', linewidth=2)
            plt.plot(history_data['val_f1_score'], label='Val F1 Macro', linewidth=2)
            plt.title('F1 Score (Macro) - Primary Metric', fontweight='bold')
            plt.legend()
        else:
            plt.text(0.5, 0.5, 'F1 Macro not tracked', ha='center', va='center')
            plt.title('F1 Score (N/A)')
        plt.grid(True, alpha=0.3)
        
        # Learning Rate (if available)
        plt.subplot(2, 2, 3)
        if 'learning_rate' in history_data:
            plt.plot(history_data['learning_rate'], linewidth=2, color='red')
            plt.title('Learning Rate', fontweight='bold')
            plt.yscale('log')
        else:
            plt.text(0.5, 0.5, 'LR not tracked', ha='center', va='center')
            plt.title('Learning Rate (N/A)')
        plt.grid(True, alpha=0.3)
        
        # Model summary info
        plt.subplot(2, 2, 4)
        plt.axis('off')
        
        final_val_f1 = history_data.get('val_f1_score', ['N/A'])[-1] if 'val_f1_score' in history_data else 'N/A'
        final_train_f1 = history_data.get('f1_score', ['N/A'])[-1] if 'f1_score' in history_data else 'N/A'
        final_val_loss = history_data['val_loss'][-1]
        epochs = len(history_data['loss'])
        
        summary_text = f"""MODEL SUMMARY

Epochs: {epochs}
Final Train F1 Macro: {final_train_f1 if final_train_f1 != 'N/A' else 'N/A'}
Final Val F1 Macro: {final_val_f1 if final_val_f1 != 'N/A' else 'N/A'}
Final Val Loss: {final_val_loss:.3f}

Parameters: {model.count_params():,}
Architecture: Multimodal (Text CNN + MobileNetV2)
"""
        
        plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.suptitle(f'Training Results: {model_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        
        # Show detailed metrics
        st.subheader("Training History Details")
        with st.expander("View Raw Training History"):
            st.json(history_data)
            
    else:
        st.error(f"Training history file not found: {history_path}")

    pdf_path = 'models/'+ multimodal_report
    show_pdf_page(pdf_path, pnum=2)

elif selected_model == "MobileNet for Images":
    st.subheader("MobileNet for Images")
    st.info("MobileNet implementation coming soon...")
    # Add your MobileNet implementation here
    
else:
    st.write("Please select a model from the dropdown above.")


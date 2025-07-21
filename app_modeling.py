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


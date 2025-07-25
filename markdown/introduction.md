# 📦 E-commerce Product Classification with Multimodal Data

## 🧾 Introduction

### 🔍 Overview
This project focuses on the automatic classification of e-commerce products using multimodal data—specifically textual and visual information.

In online shopping, it's essential to organize products using their names, descriptions, and images.  
This process of cataloging products plays a crucial role in enabling:
- Personalized search results  
- Recommendation systems  
- Improved customer navigation  

### 🎯 Project Goal
The goal of this project was to build a **multimodal classifier** to predict product type codes from Rakuten’s e-commerce listings, using both textual (title and description) and visual (image) data.

This classification is critical for improving product search, recommendation, and catalog management in large online marketplaces.

The dataset, provided by **Rakuten France**, included nearly **99,000 products** across over **1,000 categories**, posing challenges such as:
- Class imbalance  
- Missing descriptions  
- Noisy labels  

Products were described in **French** and accompanied by images, requiring **natural language processing** and **computer vision** techniques.

To address this, benchmark models were used:
- 🖼️ A **ResNet50 CNN** trained on product images achieved a **weighted F1-score of 0.5534**
- 📝 A **CNN text classifier** using product titles reached a higher **F1-score of 0.8113**

These results highlight the strength of text-based classification in this dataset. However, combining both text and image modalities (**multimodal learning**) was proposed to further enhance performance, particularly for ambiguous or sparsely described products.

The challenge also emphasized:
- Scalability  
- Robustness  
- Real-world applicability for e-commerce platforms handling vast, diverse product inventories  


## 📂 Dataset

- **Text data size:** ~60 MB  
- **Image data size:** ~2.2 GB  
- **Data source:** Rakuten Institute of Technology  

Rakuten France provides a dataset split into:
- **Training data:** 84,916 items  
- **Test data:** 13,812 items  

Each product includes:
- **Text data:** designation (title), optional description  
- **Metadata:** `productid`, `imageid`, and internal `id`  
- **Images:** Located in `images.zip`, structured into:
  - `image_training/` for training images  
  - `image_test/` for test images  

📁 The provided files include:
- Combined training inputs and outputs for data processing  
- A separate test input file  
- A ZIP archive containing product images for both training and testing  

🔍 **Objective:** Predict the `prdtypecode` for each item in the test set  



## 💡 Applications
- Automated product categorization  
- Improved recommendation engines  
- Smarter, more personalized search capabilities for e-commerce platforms  



## 📊 Evaluation and Benchmarks

Although this is not an explicit benchmark task, we adopted a **multimodal approach** that does not directly align with standard benchmarks.

Given the classification nature of the problem, we defined **weighted F1-score** as our evaluation metric to assess model performance.

The benchmark models achieved the following weighted F1-scores:
- 📝 **Text modality:** 81.13%  
- 🖼️ **Image modality:** 55.34%  



## 🔧 Project Steps

- **Data Discovery:** Checked for missing values, duplicates, and unique entries per column  
- **Data Exploration:**  
  - Created word clouds to visualize product types  
  - Identified class imbalance  
  - Found high missing data in book-related categories  
  - Detected few duplicates in childcare categories  

- **Image Processing:**  
  - Applied baseline resizing  
  - Performed background removal and smart cropping  
  - Introduced advanced augmentation techniques  

- **Modeling:**  
  - Used basic ML classifiers and Text CNN  
  - Built a **multimodal fusion model** using Text + MobileNetV2  
  - Trained using AdamW optimizer, regularization, class weights, and data augmentation  

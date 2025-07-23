# Overview

This project focuses on the automatic classification of e-commerce products using **multimodal data**—
specifically textual and visual information.  

In online shopping, it's essential to organize products using their names, descriptions, and images.  
This process of cataloging products plays a crucial role in enabling:
- Personalized search results  
- Recommendation systems  
- Improved customer navigation


# Objective

The goal is to **predict the product type code** using two types of input data:
- **Textual data**: Product designation (name) and description  
- **Image data**: Visual representation of the product  

By accurately classifying products based on these inputs, we aim to enhance:

- Product indexing and catalog structuring  
- Search result relevance (personalized search)  
- Product discovery (recommendation systems)


# Context

This project is based on the **Rakuten France Multimodal Product Data Classification Challenge**,  
which provides a real-world dataset to apply and test **multimodal machine learning techniques**  
in the context of large-scale e-commerce.


# Dataset

- **Total samples**: ~99,000 products  
- **Number of classes**: Over 1,000 unique product categories  
- **Text data size**: ~60 MB  
- **Image data size**: ~2.2 GB  
- **Data source**: [Challenge website](https://challengedata.ens.fr/challenges/35)

## Data Description

Rakuten France provides a dataset of ~99,000 product listings, split into:

- **Training set:** 84,916 entries  
- **Test set:** 13,812 entries

Each product includes:

- **Text data**: designation (title), description (optional)
- **Metadata**: productid, imageid, and an internal "id"
- **Images**: Located in images.zip, organized into:
- image_training/ for training images
- image_test/ for test images

The image filenames follow the format:  
image_<imageid>_product_<productid>.jpg

The files provided:

- X_train.csv: Product features (text + metadata)
- Y_train.csv: Product categories (prdtypecode)
- X_test.csv: Test features (same structure as X_train)
- images.zip: All associated product images

🔍 **Objective**: Predict the prdtypecode for each item in the test set.

# Key Challenges

The following weighted F1-scores were achieved using the benchmark models described above:
- Text modality: 81.13%
- Image modality: 55.34%

# Applications

- Automated product categorization  
- Improved recommendation engines  
- Smarter, more personalized search capabilities for e-commerce platforms


# Classification Benchmarks

As we are dealing with a **classification problem**, we defined the following **evaluation metrics** to benchmark model performance:

The following weighted F1-scores were achieved using the benchmark models described above:
- Text modality: 81.13%
- Image modality: 55.34%

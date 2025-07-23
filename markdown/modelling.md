**Simple text-based modelling:** run on the basic text tokens

```json
{‘Logistic Regression’: {‘model’: LogisticRegression(max\_iter=1000, solver=‘liblinear’), ‘f1\_score’: 0.7734938224667732, ‘f1\_weighted’: 0.7902879939112404, ‘predictions’: array(\[2705, 1302, 1560, ..., 1560, 1302, 1300\], shape=(16984,))}, ‘SGD Classifier’: {‘model’: SGDClassifier(loss=‘log\_loss’), ‘f1\_score’: 0.7342957373976658, ‘f1\_weighted’: 0.7529368805858434, ‘predictions’: array(\[1281, 1302, 1560, ..., 1560, 2280, 2280\], shape=(16984,))}, ‘Linear SVM’: {‘model’: LinearSVC(), ‘f1\_score’: 0.783122845011161, ‘f1\_weighted’: 0.8044003932739417, ‘predictions’: array(\[1280, 1302, 1560, ..., 1560, 1302, 1300\], shape=(16984,))}, ‘Random Forest’: {‘model’: RandomForestClassifier(max\_depth=20, n\_jobs=-1), ‘f1\_score’: 0.6164494092352807, ‘f1\_weighted’: 0.6270541941377651, ‘predictions’: array(\[1280, 1302, 1560, ..., 1560,  10,  10\], shape=(16984,))}}
```

**Text only CNN:** Here’s a bullet point summary of this CNN text classification model:
            
**Model Architecture:**
            
• Multi-branch CNN with parallel Conv1D layers using different kernel sizes (default: 2, 3, 4, 5)

• Embedding layer (180 dimensions) followed by convolutional branches with global max pooling

• Two dense layers (128 and 64 units) with batch normalization and dropout for regularization

• Softmax output layer for multi-class product classification
            
**Key Features:**

• tokenized text sequences with configurable vocabulary limit (15,000 tokens)

• class weighting to handle imbalanced datasets

• early stopping and learning rate reduction callbacks for training optimization

• AdamW optimizer with L2 regularization throughout the network
            
**Training Configuration:**

• Configurable hyperparameters via command line arguments (epochs, batch size, learning rate, etc.)

• Train/validation split (80/20) with stratified sampling

• F1 score monitoring for model evaluation and early stopping
            
**Analysis & Reporting:**
            
• Includes occlusion-based feature importance analysis for model interpretabilityMultimodal:

**Model Architecture:**

*   **Dual-branch multimodal system** combining text CNN and image CNN (MobileNetV2) for product classification
    
*   **Text branch**: Multi-kernel Conv1D layers (kernels 2,3,4,5) with embedding layer and global max pooling
    
*   **Image branch**: Pre-trained MobileNetV2 backbone with global average pooling and dense layers
    
*   **Fusion layer**: Concatenates text features (128D) and image features (256D) for joint classification
    

**Key Multimodal Features:**

*   Processes both tokenized French text and product images simultaneously
    
*   Custom data generator handles batch loading of text sequences and image files from disk
    
*   MobileNetV2 preprocessing with optional ImageNet normalization or standard scaling
    
*   Image augmentation support for training data (disabled for validation)
    

**Training Configuration:**

*   Command-line configurable hyperparameters for both text and image processing
    
*   Class-weighted training for imbalanced datasets with stratified train/val split
    
*   Early stopping and learning rate reduction based on validation F1 score
    
*   AdamW optimizer with weight decay and L2 regularization throughout
    

**Technical Implementation:**

*   **Data handling**: Single dataframe with pre-processed text tokens and image file paths
    
*   **Memory efficiency**: Batch-wise image loading prevents memory overflow
    
*   **Freezing strategy**: Optional MobileNetV2 backbone freezing for transfer learning
    
*   **Model saving**: Automatic saving when validation F1 score exceeds 0.7 threshold
    

**Performance & Evaluation:**

*   F1 score monitoring (macro-averaged) as primary evaluation metric
    
*   Comprehensive training report generation with hyperparameter tracking
    
*   Training history saved in JSON format for analysis
    
*   Higher complexity model targeting >70% validation F1 performance
    

**Data Pipeline:**

*   Custom MultimodalDataGenerator loads text sequences and images per batch
    
*   Handles image preprocessing, resizing (224x224), and optional augmentation
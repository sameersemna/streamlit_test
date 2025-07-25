### Main Results
- Our best-performing **text-only CNN model** achieved a weighted F1-score of **71%**, which falls short of the benchmark target.
- The **multimodal model**, combining both text and image data, achieved a weighted F1-score of **79%**, successfully surpassing the benchmark. Although more complex, it offers greater flexibility and robustness across varied inputs.

### Limitations and Potential Causes of Underperformance (Text-Only Model)
- Class imbalance, leading to biased predictions toward dominant classes  
- Overfitting, potentially due to limited training data or inadequate regularization  
- Short product descriptions, which may not provide enough context for accurate classification  
- Model architecture not fully optimized for the brevity and structure of the input text  

### Proposed Next Steps for Improvement
- Apply undersampling of dominant classes to address imbalance  
- Explore alternative text preprocessing techniques (e.g., stemming, lemmatization, n-gram features)  
- Incorporate pretrained language models (e.g., BERT, DistilBERT) for richer text embeddings  
- Conduct a broader grid search covering regularization parameters, network layer sizes, and other key hyperparameters  

### Conclusion
The **multimodal approach** improves prediction coverage, especially when one modality (e.g., text or image) is weak or missing, thereby enhancing overall model robustness. Despite some overfitting challenges, the multimodal model **outperforms the benchmark**.

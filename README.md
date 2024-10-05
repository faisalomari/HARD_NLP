# Arabic BERT Sentiment Classification on Hotel Reviews

This project implements a sentiment classification model using **ArabicBERT** for analyzing Arabic hotel reviews. The goal is to classify the reviews as either **positive** or **negative** based on their associated ratings.

## Dataset

The dataset used for this project is the **HARD Arabic Hotel Reviews** dataset, which contains user reviews in Arabic along with ratings and other metadata. The ratings are classified as follows:

- Ratings 4 and 5: Positive sentiment
- Ratings 1 and 2: Negative sentiment
- Rating 3 reviews are ignored

## Key Features

1. **Data Preprocessing**:  
   - The dataset is filtered to include only reviews with ratings of 1, 2, 4, and 5.
   - Reviews are classified into **positive** or **negative** based on their ratings.

2. **Fine-Tuning ArabicBERT**:  
   - The pre-trained ArabicBERT model is fine-tuned for binary sentiment classification.
   - The tokenizer used is from the **aubmindlab/bert-base-arabertv02** model.

3. **Model Training and Evaluation**:  
   - The dataset is split into 70% for training and 30% for testing.
   - The model is trained for 5 epochs using the Hugging Face `Trainer` API.
   - Accuracy is evaluated on the test set, and confusion matrices are plotted to visualize performance.

4. **Batch Inference**:  
   - The trained model is used to classify reviews in batches, with results saved to a CSV file.
   - Model accuracy is calculated, and true/false positives and negatives are analyzed.

## Project Structure

- `classified.csv`: Contains the classified reviews.
- `train_reviews.csv`, `test_reviews.csv`: Training and testing datasets split from the original dataset.
- `arabic_bert_review_classifier/`: The directory where the fine-tuned ArabicBERT model and tokenizer are saved.
- `model_predictions.csv`: Contains the model predictions, along with the actual classification and whether the prediction was correct.
- Python scripts to load, train, and evaluate the model, along with visualizations such as confusion matrix and loss plots.

## Installation and Requirements

To run this project, the following dependencies are required:

```bash
pip install pandas sklearn torch transformers datasets matplotlib seaborn
```

##

1. Load and Preprocess Data:
- Load the Arabic hotel reviews, classify them based on ratings, and split them into training and testing sets.

2. Train the Model:
- Fine-tune ArabicBERT using Hugging Face Trainer API for sentiment classification.

3. Evaluate the Model:
- Evaluate the model on the test set, visualize losses, and plot the confusion matrix.

4. Save Predictions:
- The predictions from the model are saved to a CSV file, along with accuracy scores and true/false positive and negative examples.

## Evaluation Metrics
- Accuracy: The percentage of correctly classified reviews.
- Confusion Matrix: Visualizes the classification performance, including true positives, true negatives, false positives, and false negatives.
- Loss Plot: Training and evaluation loss curves are plotted to show the model's learning progress.

## Results
- The model achieved an accuracy of 94% on the test set.
- Examples of true and false positives/negatives are included in the output for further analysis.

## Citation
If you use this code or the ArabicBERT model in your work, please cite the original authors:
```bash
Antoun, W., Baly, F., & Hajj, H. (2020). ArabicBERT: Transformer-based Model for Arabic Language Understanding. Proceedings of the 2020 International Conference on Arabic Computational Linguistics.
```
## Contact
Faisal Omari
Email: faisalomari321@gmail.com
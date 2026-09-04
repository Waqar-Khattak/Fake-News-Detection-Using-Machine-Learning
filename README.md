# Fake News Detection Using Machine Learning

An NLP classification project that predicts whether a news article is real or fake using the WELFake dataset and three machine-learning models.

## Project Overview

The notebook combines each article's title and body, cleans the text, converts it into TF-IDF features, and trains multiple classifiers:

- Multinomial Naive Bayes
- Linear Support Vector Machine (LinearSVC)
- Logistic Regression

The models are evaluated using accuracy, precision, recall, F1-score, classification reports, confusion matrices, and a comparison chart.

## Dataset

This project uses the [WELFake dataset](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification), stored locally as `WELFake_Dataset.csv`.

The dataset contains news titles, article text, and binary labels:

- `0`: Real news
- `1`: Fake news

The notebook removes rows with missing titles or article text before training.

## Requirements

- Python 3.9 or later
- Jupyter Notebook or Visual Studio Code with the Jupyter extension
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- nltk

Install the dependencies with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk jupyter
```

## How to Run

1. Clone this repository:

   ```bash
   git clone <your-repository-url>
   cd <repository-folder>
   ```

2. Confirm that `WELFake_Dataset.csv` is in the same directory as `Fake_News_Detection.ipynb`.

3. Open the notebook in Jupyter or Visual Studio Code.

4. Run the cells from top to bottom.

The first cell downloads the required NLTK resources. The notebook then creates these image files:

- `label_distribution.png`
- `word_count_distribution.png`
- `confusion_matrices.png`
- `model_comparison.png`

## Processing Pipeline

1. Load the CSV dataset.
2. Remove records missing a title or article body.
3. Combine the title and body into one text field.
4. Convert text to lowercase.
5. Remove URLs, HTML tags, punctuation, and numbers.
6. Remove English stopwords and apply Porter stemming.
7. Split the data into 80% training and 20% testing sets.
8. Generate unigram and bigram TF-IDF features.
9. Train and evaluate three classification models.
10. Select the model with the highest F1-score.

## Results

The saved notebook outputs report the following test-set performance:

| Model | Accuracy | Precision | Recall | F1-score |
| --- | ---: | ---: | ---: | ---: |
| Naive Bayes | 89.71% | 89.76% | 89.71% | 89.70% |
| LinearSVC | 97.76% | 97.77% | 97.76% | 97.76% |
| Logistic Regression | 96.36% | 96.36% | 96.36% | 96.36% |

Based on these results, **LinearSVC** is the best-performing model in this experiment.

## Important Notes

- This project identifies language patterns associated with the dataset labels; it does not fact-check news against external sources.
- Reported results depend on the dataset, preprocessing choices, random seed, and train-test split.
- Verify the label meanings if using a different copy or version of the dataset.
- The sample predictions in the notebook are illustrative and should not be treated as reliable fact verification.
- For production use, the preprocessing pipeline and trained model should be saved together so new articles are transformed consistently.

## Project Files

```text
.
|-- Fake_News_Detection.ipynb
|-- WELFake_Dataset.csv
|-- README.md
```

## Author

Waqar Ahmad

# Shopping Intent Classifier

## Background

Online shopping has become an integral part of modern life, yet not all users who visit an online shopping website end up making a purchase. In fact, the majority of visitors leave without completing a transaction. Understanding user intent can be a powerful tool for online retailers, enabling them to tailor the shopping experience to individual users. For instance, if a website predicts that a user is unlikely to make a purchase, it could display a discount offer or other incentives to encourage conversion.

This project leverages **machine learning** to predict whether a user intends to make a purchase based on their browsing behavior. By analyzing features such as the number of pages visited, the type of web browser used, and whether the session occurred on a weekend, we aim to build a **nearest-neighbor classifier** that can make informed predictions about user intent. While no model can perfectly capture the complexities of human behavior, our classifier seeks to outperform random guessing and provide meaningful insights.

## Objectives

The primary goals of this project are:

1. **Build a Classifier**: Develop a nearest-neighbor classifier to predict whether a user will complete a purchase.
2. **Evaluate Performance**: Measure the classifier's performance using two key metrics:
   - **Sensitivity (True Positive Rate)**: The proportion of users who made a purchase and were correctly identified.
   - **Specificity (True Negative Rate)**: The proportion of users who did not make a purchase and were correctly identified.
3. **Optimize for Balance**: Create a model that performs reasonably well on both sensitivity and specificity, avoiding the pitfalls of overly simplistic classifiers.

## Why Sensitivity and Specificity?

Accuracy alone can be misleading in imbalanced datasets. For example, if only 15% of users make a purchase, a classifier that always predicts "no purchase" would achieve 85% accuracy but fail to provide any meaningful insights. By focusing on sensitivity and specificity, we ensure that our model captures both positive and negative cases effectively, providing a more nuanced understanding of user behavior.

## Dataset

The dataset consists of approximately **12,000 user sessions** from an online shopping website. Each session includes features such as:
- Number of pages visited
- Whether the session occurred on a weekend
- The web browser used
- Other relevant behavioral data

This data will be split into training and testing sets to evaluate the model's performance.

## Methodology

1. **Data Preprocessing**: Clean and normalize the dataset to ensure compatibility with the nearest-neighbor algorithm.
2. **Feature Selection**: Identify the most relevant features for predicting user intent.
3. **Model Training**: Implement a nearest-neighbor classifier using the training data.
4. **Performance Evaluation**: Test the model on unseen data and calculate sensitivity and specificity to assess its effectiveness.

## Challenges

- **Imbalanced Data**: With a small proportion of users completing purchases, the dataset is inherently imbalanced, requiring careful handling to avoid biased predictions.
- **Human Behavior**: Predicting human intent is inherently complex and subject to numerous external factors that may not be captured in the dataset.

## Conclusion

This project demonstrates the potential of machine learning to enhance the online shopping experience by predicting user intent. By focusing on sensitivity and specificity, we aim to build a classifier that provides actionable insights for online retailers, helping them optimize their platforms and improve conversion rates.

---

**Developed by:** Jovani Velasco  
**Institution:** Harvard University  
**Course:** cs50ai  
**Date:** January 2023

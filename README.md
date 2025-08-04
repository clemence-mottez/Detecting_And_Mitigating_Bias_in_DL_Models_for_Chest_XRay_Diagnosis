# From Detection to Mitigation: Addressing Bias in Deep Learning Models for Chest X-Ray Diagnosis
Paper in review by Pacific Symposium on Biocomputing (PSB) 2026  

## Overview
Deep learning models have shown promise in improving diagnostic accuracy from chest X-rays, but they also risk perpetuating healthcare disparities when performance varies across demographic groups. In this work, we present a comprehensive bias detection and mitigation framework targeting sex, age, and race-based disparities when performing diagnostic tasks with chest X-rays. We extend a recent CNN–XGBoost pipeline to support multi-label classification and evaluate its performance across four medical conditions. We show that replacing the final layer of CNN with an eXtreme Gradient Boosting classifier improves the fairness of the subgroup while maintaining or improving the overall predictive performance. To validate its generalizability, we apply the method to different backbones, namely DenseNet-121 and ResNet-50, and achieve similarly strong performance and fairness outcomes, confirming its model-agnostic design. We further compare this lightweight adapter training method with traditional full-model training bias mitigation techniques, including adversarial training, reweighting, data augmentation, and active learning, and find that our approach offers competitive or superior bias reduction at a fraction of the computational cost. Finally, we show that combining eXtreme Gradient Boosting retraining with active learning yields the largest reduction in bias across all demographic subgroups, both in and out of distribution on the CheXpert and MIMIC datasets, establishing a practical and effective path toward equitable deep learning deployment in clinical radiology.

## Conda Environment

To create a Conda environment for this project, run the following command in your terminal:

```bash
conda create --name mitigate-bias python=3.8.19
```
To open the environment, run the following command in your terminal:

```bash
conda activate mitigate-bias
```

## How to use our code
- Bias Detection folder contains notebooks to inspect the bias and determine where does it come from
- CNN from scratch folder contains python scripts to train our own DenseNet121 from scratch as well as scripts to extract disease predictions and image embeddings
- XGBoost Bias Mitigation contains notebook to retrain the head of the CNN with a XGBoost classifier as well as a notebook to combine Active Learning with the XGBoost head retraining

## Labels in the data
  
"sex": {"Male": 0, "Female": 1},  
"race": {"White": 0, "Asian": 1, "Black": 2},  
"age": {"Young (<70)": 0, "Old (>70)": 1}  

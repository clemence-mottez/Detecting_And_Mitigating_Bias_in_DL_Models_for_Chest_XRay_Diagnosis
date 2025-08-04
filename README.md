# Detecting_And_Mitigating_Bias_in_DL_Models_for_Chest_XRay_Diagnosis

## Overview
This project aims to detect and mitigate bias in Deep Learning models for Chest X-Ray Diagnosis

## Creating a Conda Environment

To create a Conda environment for this project, run the following command in your terminal:

```bash
conda create --name mitigate-bias python=3.8.19
```
To open the environment, run the following command in your terminal:

```bash
conda activate mitigate-bias
```

## Labels in the data
  
"sex": {"Male": 0, "Female": 1},  
"race": {"White": 0, "Asian": 1, "Black": 2},  
"age": {"Young (<70)": 0, "Old (>70)": 1}  

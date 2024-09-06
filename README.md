# Telecommunication dataset analysis
# Data Analysis and User Engagement Analysis

## Overview

This repository contains the code and resources for performing two key data analysis tasks related to telecommunications:

1. **User Overview Analysis**: This involves exploring user behavior based on data sessions and identifying trends in handset usage and user engagement metrics.
2. **User Engagement Analysis**: This focuses on analyzing user engagement metrics, normalizing data, and applying clustering techniques to segment users based on their engagement.

## Project Structure
├── .vscode/
│   └── settings.json
├── .github/
│   └── workflows
│       ├── unittests.yml
├── .gitignore
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
├── notebooks/
│   ├── Task1.ipynb
│   ├── Task2.ipynb
│   ├── __init__.py
│   └── README.md
├── tests/
│   ├── __init__.py
└── scripts/
    ├── __init__.py
    └── README.md


## Getting Started

### Prerequisites

- Python 3.x
- Required Python libraries listed in `requirements.txt`

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yohannestayez/Telecommunication-dataset-analysis.git
Navigate into the project directory:

cd your-repository
Install the required libraries: pip install -r requirements.txt


## Tasks
### Task 1: User Overview Analysis
Objective: Understand user behavior and identify key metrics from data sessions.

**Steps**:

Import and clean the data.
Identify the top 10 handsets used by customers.
Determine the top 3 handset manufacturers and the top 5 handsets per manufacturer.
Perform exploratory data analysis (EDA) on xDR data.
Handle missing values and outliers.
Provide insights and recommendations to the marketing team.
**Notebook**:
notebooks/Task1_User_Overview_Analysis.ipynb: Contains detailed steps, code, and visualizations for analyzing handset usage and performing EDA.

### Task 2: User Engagement Analysis
Objective: Evaluate user engagement based on session metrics and apply clustering techniques.

**Steps**:

Aggregate metrics per customer ID (MSISDN).
Normalize engagement metrics.
Perform K-Means clustering to classify customers into engagement groups.
Determine the optimized number of clusters using the Elbow method.
Visualize cluster results and summarize findings.
**Notebook**:
notebooks/Task2_User_Engagement_Analysis.ipynb: Contains detailed steps, code, and visualizations for aggregating metrics, normalizing data, and applying clustering.

## Usage
Run the analysis notebooks to generate insights and perform clustering:

Open notebooks/Task1_User_Overview_Analysis.ipynb to start analyzing user overview.
Open notebooks/Task2_User_Engagement_Analysis.ipynb to perform user engagement analysis.


## License
This project is licensed under the MIT License - see the LICENSE file for details.

For detailed documentation and additional information, please refer to the notebooks/ directory and the individual notebooks for each task.

This updated `README.md` specifies that Task 1 and Task 2 are in Jupyter notebooks within the `notebooks/` f
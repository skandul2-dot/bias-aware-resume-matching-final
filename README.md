# Bias-Aware Resume Matching

This project studies whether an automated resume-job matching system gives different scores when only demographic signals in a resume are changed. The main model uses Sentence-BERT to calculate resume-job similarity scores, and the project audits whether those scores change when the applicant's name, pronouns, or university are modified.

The project also includes a blind-screening mitigation experiment and an open-source LLM comparison to address the project feedback about examining a current language model.

## Project Overview

Automated hiring systems are often used to filter job applications before a human reviews them. While these systems can improve efficiency, they may also reflect or amplify bias if the model gives different scores to equally qualified applicants based on demographic cues.

In this project, we create counterfactual resume variants where only one demographic signal changes at a time. Then we compare each resume version with job descriptions using SBERT cosine similarity scores.

## Main Components

### 1. SBERT Resume-Job Matching

The notebook `sbert_matching_final.ipynb` uses Sentence-BERT to convert resumes and job descriptions into embeddings. Cosine similarity is then used to calculate a matching score for each resume-job pair.

Output:

`results/sbert_scores.csv`

### 2. Counterfactual Fairness Analysis

The notebook `fairness_analysis_final.ipynb` compares original resume scores with counterfactual resume scores. The score difference shows whether changing only the name, pronouns, or university affects the matching score.

Outputs:

`results/fairness_comparison.csv`  
`results/fairness_summary.csv`

### 3. Blind Screening Mitigation

The notebook `blind_screening_final.ipynb` implements a blind-screening mitigation step by removing identity-related information such as name, pronouns, and university from the resume text. SBERT scores are then recalculated to check whether score differences are reduced.

Outputs:

`results/blind_sbert_scores.csv`  
`results/blind_fairness_comparison.csv`  
`results/blind_fairness_summary.csv`

### 4. Open-Source LLM Comparison

The notebook `llm_demo_final.ipynb` uses an open-source instruction-tuned language model to classify selected resume-job pairs as Strong match, Partial match, or Weak match. This LLM step is used as a qualitative comparison, while SBERT remains the main numerical model for fairness analysis.

Output:

`results/llm_match_decisions.csv`

## Repository Structure

```text
data/
  base_resumes.csv
  jobs.csv
  resume_variants.csv

notebooks/
  sbert_matching_final.ipynb
  fairness_analysis_final.ipynb
  blind_screening_final.ipynb
  llm_demo_final.ipynb

results/
  sbert_scores.csv
  fairness_comparison.csv
  fairness_summary.csv
  blind_sbert_scores.csv
  blind_fairness_comparison.csv
  blind_fairness_summary.csv
  llm_match_decisions.csv

report/
README.md
```

## Dataset

The dataset contains resumes and job descriptions across five domains:

- Software Engineering
- Finance
- Marketing
- Healthcare
- Education

Each base resume has counterfactual versions where only one demographic signal is changed:

- Name
- Pronouns
- University

All other resume qualifications remain the same, so score differences can be attributed to the changed demographic signal.

## Methods

### Sentence-BERT Matching

Sentence-BERT is used to create dense text embeddings for resumes and job descriptions. Cosine similarity is used as the resume-job matching score.

### Counterfactual Fairness Audit

For each resume-job pair, the project compares the original resume score with the changed resume score.

The score difference is calculated as:

`score_difference = changed_score - original_score`

The absolute difference is used to measure how much the score changed regardless of direction.

### Blind Screening

Blind screening removes identity-related information before scoring. This helps test whether removing demographic cues reduces score differences across counterfactual resume versions.

### LLM Comparison

An open-source instruction-tuned LLM is used to provide qualitative match decisions for selected resume-job pairs. This addresses the project feedback to examine a current LLM in addition to Sentence-BERT.

## Key Findings

The original SBERT fairness analysis showed that changing demographic signals can slightly change resume-job matching scores. Name changes produced the largest average score shift, followed by university changes and pronoun changes.

After blind screening, the counterfactual score differences were reduced. This suggests that removing identity-related cues can help reduce demographic sensitivity in this dataset.

The LLM comparison showed how a current instruction-tuned language model can provide a qualitative resume-job match judgment. However, SBERT was kept as the main scoring model because it provides consistent numerical scores for fairness analysis.

## Limitations

This project uses a small dataset, so the results should be treated as an audit-style study rather than a final conclusion about real-world hiring systems. The LLM component is qualitative and is not used as the main fairness metric. Also, blind screening was implemented as the mitigation method, while other methods such as data augmentation and adversarial debiasing can be explored in future work.

## Future Work

Future improvements can include:

- Expanding the dataset with more resumes and job descriptions
- Testing more advanced LLMs for resume-job matching
- Implementing data augmentation using counterfactual pairs
- Implementing adversarial debiasing
- Comparing accuracy and fairness tradeoffs across mitigation methods

## Technologies Used

- Python
- Google Colab
- pandas
- scikit-learn
- Sentence-BERT
- Hugging Face Transformers

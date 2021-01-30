# riiid-prediction
## Challenge

This repo covers my participation in the [Riiid AIEd Challenge 2020](https://www.kaggle.com/c/riiid-test-answer-prediction/overview)

Final submission was ranked [# 54](https://www.kaggle.com/c/riiid-test-answer-prediction/leaderboard) out of 3,395 participants (top 2%).


From the Kaggle description:
> Riiid Labs launched an AI tutor based on deep-learning algorithms in 2017 that attracted more than one million South Korean students. This year, the company released EdNet, the world’s largest open database for AI education containing more than 100 million student interactions.

> In this competition, your challenge is to create algorithms for "Knowledge Tracing," the modeling of student knowledge over time. The goal is to accurately predict how students will perform on future interactions. You will pair your machine learning skills using Riiid’s EdNet data.

> Submissions are evaluated on AUC between the predicted probability and the observed target.  Expect about 2.5 million questions in the hidden test set.


## Training data
- 102 Million interaction, covering ~3 years of history of students preparing for the [TOEIC test](https://www.ets.org/toeic)
- 393,656 users
- 13,523 questions, 65% answered correctly
- clean and homogenous data: no gaps, no regime changes, no data missing at random

## Approach

The approach was adapted from a research papers by the company, for example [Riiid SAINT+](https://arxiv.org/abs/2010.12042), Shin et al., 2020
Key considerations:
- answer correctness is a function of both 
  - user's learning trajectory (non-Markovian, needs to incroporate long-term dependencies)
  - question features, e.g. a measure of how hard the question generally is, which TOEIC part it belongs to, etc.
- we need to keep track of each individual students' learning trajectory
- cross-validate to mimic a consecutive time period. i.e. a mix of old users and new users


Transformer architecture is following the original architecture per [Attention Is All You Need](https://arxiv.org/abs/1706.03762), Vaswani et al., 2017, with both Encoder and Decoder parts present:

![alt text](https://drive.google.com/uc?id=14AEscOTfCNBfEJwPBOMpQYO6u2XByHta)


## Model research
There are many possible model architectures beyond transformers, such as LSTMs, stacked architectures, state space models (widely used for knowledge tracing in the past), etc. But even within transformers, there are many high-level design choices we have to make which strongly affect the model performance on Riiid data set. 

The focus of model research was therefore not on hyperparameter tuning within a narrow family but a broader exploration of possible structures.

Some of modifications tested:
- XGBoost, LightGBM: provided a robust memory-less benchmark
- stacked model (Transformer + LightGBM)
- incremental retraining 
- split the users into light and heavy, train separately
- concatenate, not add, the model depth components (questions embedding, the rest of the inputs) - **Otpimal**

Feature engineering experiments:
- add question's overall correctness across users - **Adds value**
- add a repeat question indicator - **Adds value**
- add time_since_lecture for the relevant TOEIC parts
- add user's correctness ratio for the relevant TOEIC part
- add a 'stability' measure, how methodical the user is in going through the sections
- add question tags embedding (Riiid-defined categories)
- add prior_question_had_explanation

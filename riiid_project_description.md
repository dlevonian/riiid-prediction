## Challenge

This repo covers my participation in the Riiid AIEd Challenge 2020 [Kaggle page](https://www.kaggle.com/c/riiid-test-answer-prediction/overview)

Final submission was ranked [# 54](https://www.kaggle.com/c/riiid-test-answer-prediction/leaderboard) out of 3,395 participants (top 2%).


From the Kaggle description:

> Riiid Labs launched an AI tutor based on deep-learning algorithms in 2017 that attracted more than one million South Korean students. This year, the company released EdNet, the world’s largest open database for AI education containing more than 100 million student interactions.

> In this competition, your challenge is to create algorithms for "Knowledge Tracing," the modeling of student knowledge over time. The goal is to accurately predict how students will perform on future interactions. You will pair your machine learning skills using Riiid’s EdNet data.

> Submissions are evaluated on AUC between the predicted probability and the observed target.  Expect about 2.5 million questions in the hidden test set.



Training set:
- 102 Million interaction, covering ~3 years of history of students preparing for the [TOEIC test](https://www.ets.org/toeic)
- 393,656 users
- 13,523 questions, 65% answered correctly
- clean and homogenous data: no gaps, no regime changes, no data missing at random

## Approach

The approach was adapted from a research papers by the company, for example [Riiid SAINT+](https://arxiv.org/abs/2010.12042), Shin et al., 2020

Key considerations:
- answer correctness is a function of both 
  - the student features, e.g. his/her past history, and 
  - the question features, e.g. a measure of how hard the question generally is, which TOEIC part it belongs to, etc.
- we need to keep track of each individual students' learning trajectory
- cross-validate to mimic a consecutive time period. i.e. a mix of old users and new users


Transformer architecture is following the original architecture per [Attention Is All You Need](https://arxiv.org/abs/1706.03762), Vaswani et al., 2017, with both Encoder and Decoder parts present:

![alt text](https://drive.google.com/uc?id=14AEscOTfCNBfEJwPBOMpQYO6u2XByHta)

The base case (benchmark) should be as close to SAINT+ as possible.  
List all implicit assumptions not mentioned in the paper:
- positional encoding as in AIAYN
- part embedding relates to question's part (1-7). Question tags are more granular description of the question, apparently not included in SAINT+
- lecture information not included
- prior_question_had_explanation not included
- all embeddings have dimesnion D_MODEL and are apparently added as opposed to concatenated -- needs to be tested






Modifications explored:
- stacked model (Transformer + LightGBM)
- incremental retraining 
- add time_since_lecture for the relevant TOEIC parts
- add user's correctness ratio for the relevant TOEIC part
- add a 'stability' measure, how methodical the user is in going through the sections
- split the users into light and heavy
- concatenate, not add, the questions embedding with the rest of the inputs - **Otpimal!**
- encoder: add question tags (Riiid-defines categories) in addition to or instead of part embedding
- decoder: add prior question had explanation

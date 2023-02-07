# Do Explanations Improve the Quality of AI-assisted Human Decisions? An Algorithm-in-the-Loop Analysis of the Effects of Factual \& Counterfactual Explanations

This repository is the official implementation of [Do Explanations Improve the Quality of AI-assisted Human Decisions? An Algorithm-in-the-Loop Analysis of the Effects of Factual \& Counterfactual Explanations]() published in [The 22nd International Conference on Autonomous Agents and Multiagent Systems](https://aamas2023.soton.ac.uk/) in London, 29 May - 2 June 2023. 

## Directory Structure
* `analysis`: contains all the analysis files used to generate data and figures used in publication
  * `datasets`: contains three csv files. `defendants.csv` is the file with the 300 defendants presented to participants in the experiment, `results.csv` is the file with all the prediction data collected from the experiments, `participants.csv` is the file with information on each of the respondents from the experiment.
  * `accuracy.ipynb`: analyses under the first desideratum -> accuracy
  * `reliability.ipynb`: analyses under the second desideratum -> reliability
  * `fairness.ipynb`: analyses under the third desideratum -> fairness
  * `effective-explanations.ipynb`: analyses under the fourth desideratum -> effective explanations
  * `survey-summary`: summary of some intro and exit survey responses
* `model-and-exp`: contains files to train and test risk assessment model using COMPAS data, and generate explanations (via SHAP and DiCE)
  * `datasets`: contains COMPAS data (`compas-scores-two-years.csv`), crime categorization file (`crime-categories.csv`), explanation files (`shap_exp.csv`, `diff_sel.csv`, `diff_div.csv`), the model test set (`narratives.csv`), and the defendant sample used in the experiment (`sample.csv`).
* `requirements.txt`: required python libraries for model training, explanation generation, and analysis

## Running Analysis
### System Requirements

* To install Python 3, follow these [instructions](https://realpython.com/installing-python/). 
* To install Pip, follow these [instructions](https://pip.pypa.io/en/stable/installing/).
* To install Jupyter Lab/Notebook, follow these [instructions](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html). To run Jupyter Lab/Notebook, follow these [instructions](https://jupyter.readthedocs.io/en/latest/running.html). 
* To set up a virtual environment and use it in Jupyter Lab/Notebook, follow these [instructions](https://janakiev.com/blog/jupyter-virtual-envs/).

* To install requirements:

1. Clone this github repository 
```
git clone <url-to-this-repo>
cd <cloned-repo>
cd public-repo
```
2. Get Python requirements needed
```
pip3 install -r requirements.txt
```
### Understanding Datasets
There are three datasets used in the analysis: 
1. [defendants.csv](/analysis/datasets/defendants.csv): file with information on the 300 defendants sampled and presented to participants in the experiment
   * `id`: unique defendant identifier
   * `age`: defendant age
   * `sex`: defendant sex (male/female)
   * `race`: defendant race (Caucasian/African-American)
   * `priors_count`: defendant number of prior convictions
   * `juv_fel_count`: defendant number of juvenile felony charges
   * `juv_misd_count`: defendant number of juvenile misdemeanor charges
   * `c_charge_degree`: defendant criminal charge degree (felony/misdemeanor)
   * `offense_type`: defendant offense type (one of 8 categories)
   * `real_outcome`: whether or not the defendant reoffended (recidivism = 1, no recidivism = 0)
   * `alg_outcome`: whether or not the model predicted the defendant will reoffend (recidivism = 1, no recidivism = 0)
   * `alg_risk_score`: probability of the defendant reoffending predicted by the model
   * `alg_risk_score_decile`: `alg_risk_score` as a decile score
   * `influence_all`: influence of the risk assessment model on participants making predictions about the defendant across all treatments
   * `influence_1`: influence of the risk assessment model on participants making predictions about the defendant in treatment 1 (unexplained risk assessment model)
   * `influence_2`: influence of the risk assessment model on participants making predictions about the defendant in treatment 2 (diverse counterfactual)
   * `influence_3`: influence of the risk assessment model on participants making predictions about the defendant in treatment 3 (selective counterfactual)
   * `influence_4`: influence of the risk assessment model on participants making predictions about the defendant in treatment 4 (complete feature attribution)
   * `influence_5`: influence of the risk assessment model on participants making predictions about the defendant in treatment 5 (selective feature attribution)
3. [results.csv](/analysis/datasets/results.csv): file with information on all the predictions made in the experiment
   * `session_id`: unique identifier of an experiment session (30 predictions by a unique participant)	
   * `response_id`: unique identifier of each prediction in the results	
   * `treatment`: the treatment each session belongs to	
   * `defendant_id`: unique defendant identifier
   * `defendant_race`: defendant race (Caucasian/African-American)
   * `defendant_age`: defendant age	
   * `defendant_sex`: defendant sex (male/female)	
   * `defendant_priors`: defendant number of prior convictions	
   * `defendant_juv_fel_count`: defendant number of juvenile felony charges
   * `defendant_juv_misd_count`: defendant number of juvenile misdemeanor charges
   * `defendant_charge_degree`: defendant criminal charge degree (felony/misdemeanor)
   * `defendant_offense_type`: defendant offense type (one of 8 categories)
   * `task_order`: the order of the prediction task within the 30 tasks	
   * `ra_score`: decile risk score predicted by the risk assessment model	
   * `participant_score`: decile risk score predicted by the participant
   * `participant_gender`: participant gender
   * `participant_age`: participant age	
   * `participant_degree`: participant education degree
   * `participant_ethnicity`: participant ethnicity
   * `participant_politics`	: participant political party affiliation
   * `actual_outcome`: actual recidivism outcome of defendant	
   * `influence`: average influence of the risk assessment model on the participant over all the 30 predictions	
   * `deviation`: amount of deviation of participant score from risk assessment model score
   * `task_sub_time`: timestamp of task submission
5. [participants.csv](/analysis/datasets/participants.csv): file with information on participants and their survey responses. All multiple choice questions (MCQ) are on a 5-point Likert scale: (1) Not at all, (2) Slightly, (3) Moderately, (4) Very, (5) Extremely, except for the question on `accountability`
   * `session_id`: unique identifier of an experiment session (30 predictions by a unique participant)	
   * `treatment`: the treatment each session belongs to	
   * `participant_gender`: participant gender
   * `participant_age`: participant age	
   * `participant_degree`: participant education degree
   * `participant_ethnicity`: participant ethnicity
   * `participant_politics`	: participant political party affiliation
   * `ml_fam`: MCQ answer to this survey question, "How familiar are you with machine learning?"
   * `cj_fam`: MCQ answer to this survey question, "How familiar are you with the U.S. Criminal Justice System?"	
   * `confidence`: MCQ answer to this survey question, "How confident were you in your decisions?"	
   * `relative_confidence`: MCQ answer to this survey question, "How well do you think you did compared to other experiment participants?"
   * `self_reported_influence`: MCQ answer to this survey question, "How much did the algorithm's risk score influence your decision?"	
   * `self_reported_exp_usefulness`: MCQ answer to this survey question, "For each defendant, you were presented with an explanation shedding light on why the algorithm predicted a specific score for the defendant. How useful was that explanation?"	
   * `self_reported_ra_accuracy`: MCQ answer to this survey question, "How accurate do you think the risk score algorithm is?"	
   * `self_reported_ra_fairness`: MCQ answer to this survey question, "How fair (i.e. neutral and unbiased) do you think the risk score algorithm is?"	
   * `self_reported_exp_ability`: MCQ answer to this survey question, "If one of the decisions you made goes wrong or is questioned, how well can you explain how you arrived at that decision?"	
   * `accountability`: MCQ answer to this survey question, "If one of the decisions you made goes wrong or is questioned, how much accountability do you think you should face?"	 Options: (1) None, (2) Less than the developers of the algorithm, (3) Equal to the developers of the algorithm, (4) More than the developers of the algorithm, (5) I should face accountability, but the developers of the algorithm should not
   * `open_response_1`: open-response answer to this survey question, "Could you tell us how you incorporated the algorithm's risk scores in your decisions (if at all)?"
   * `open_response_2`	open-response answer to this survey question, "Could you tell us how you incorporated those explanations in your decisions (if at all)?"
   * `influence`: influence of risk assessment model on participant predictions over 30 predictions	
   * `participant_brier_score`: (1 - brier loss) of participant over 30 predictions	
   * `false_positive_participant`: overall participant false positive rate
   * `ra_brier_score`: (1 - brier loss) of the risk assessment model over 30 predictions	
   * `false_positive_ra_black`: risk assessment model false positve rates for black defendants	
   * `false_positive_ra_white`: risk assessment model false positve rates for white defendants
   * `false_positive_ra_diff`: difference in risk assessment model false positve rates for black vs white defendants
   * `session_submit_time`: timestamp of session submission	
   
   
### Running Jupyter Notebooks
All the analysis notebooks used to generate the figures and results used in the publication can be found in this [folder](/analysis):
- [datasets](/analysis/datasets)
- [accuracy.ipynb](/analysis/accuracy.ipynb)
- [reliability.ipynb](/analysis/reliability.ipynb)
- [fairness.ipynb](/analysis/fairness.ipynb)
- [effective-explanations.ipynb](/analysis/effective-explanations.ipynb)
- [survey-summary.ipynb ](/analysis/survey-summary.ipynb)

To train the gradient boosted model used and to generate explanations, follow the instructions in this [notebook](/model-and-exp/model-and-exp.ipynb). 


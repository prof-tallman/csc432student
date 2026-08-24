# Clinical Prediction Model #

## Overview ##

A hospital is evaluating a new treatment for a serious medical condition. The treatment is expensive and requires specialized staff, so not every patient can receive it. Physicians currently review patient records and estimate which patients are the most likely to benefit.

The hospital has also been experimenting with predictive models. Early prototypes can evaluate patients much faster than physicians and have performed about as well as--or slightly better than--human judgment alone. However, the models have also made some significant mistakes by producing incorrect recommendations that the physicians would have easily spotted.

The hospital does not yet know how, or whether, such a model should eventually be used in treatment decisions. Your job is to develop and evaluate a predictive model using historical patient data. You are not in a position to decide whether the hospital should adopt said model, only to create the best one possible.

## The Data ##

You have received a worldwide dataset containing records from a large number of previous patients. Each record includes the patient's eventual treatment outcome along with information such as:

- age and other demographic information;
- medical measurements and test results;
- disease severity and medical history;
- ancestry group and geographic region;
- financial or access-related information; and
- several other potentially pertinent variables.

Do not assume that there is one universally correct way to clean or engineer the data. The patient data is incomplete due to inconsistent data collection practices, equipment outages, incomplete testing (e.g., patient no-shows), and recording errors. How you decide to handle these inconsistencies is an important part of the assignment.

## Your Model ##

Develop a model that estimates whether a patient is likely to benefit from the treatment. You have considerable freedom in your approach. For example, you may build a regression model that estimates a probability of success; a classification model that assigns categories such as *Treat*, *Do Not Treat*, or *Human Review*; or a model that produces an entirely different type of output. Additionally, your model might be based on traditional machine-learning algorithms or it might be built on a neural network architecture. Keep in mind that complex models are not automatically better models.

You are responsible for deciding how the data should be prepared and represented. In particular, you will need to make decisions about:

- which features to include;
- how to handle missing values;
- how to represent categorical variables;
- the best way to handle demographic information;
- whether any new features should be derived from the existing data;
- how to divide the data for training, validation, and/or testing; and
- which measurements you will use to evaluate your model.

For missing data, possible approaches include removing incomplete records, substituting a fixed value, using a mean or median, estimating values from other data, or any other reasonable technique. Different variables do not necessarily need to be handled in the same way.

## Evaluate Your Decisions ##

Do not stop after producing a model with a good overall score. Test at least two reasonable approaches to handling missing data and compare their effects. You should also investigate whether important modeling or preprocessing decisions change the model's predictions for certain subsets of the patient population.

The hospital is especially concerned about cases near the boundary between one recommendation and another. A preprocessing decision that barely affects overall accuracy could still have a major effect on an individual patient. Your model has the potential to save lives or to deny somebody life-saving treatment.

You are not expected to discover a single objectively "correct" model. You are expected to make informed decisions, understand their consequences, and be able to justify your decisions. Your goal is to document what you did and what you discovered, not to provide a full analysis.

## Submission ##

Submit your code or a notebook along with a brief technical summary that identifies:

1. the type of model you created;
2. the features you used and any important feature engineering;
3. how you represented categorical data;
4. how you handled missing data;
5. the alternative missing-data approach(es) you tested;
6. how you evaluated the model; and
7. any important differences you observed between the approaches.

Keep the summary concise. Ideally, you would fit it all on a single page. Its purpose is to document what you did and what you discovered, not to provide a full analysis.

## Grading — 50 Points ##

Your grade is based primarily on the quality of your data-processing decisions, how carefully you evaluate their consequences, and how well you explain and justify those decisions. A model with slightly lower overall accuracy may earn a higher grade if it demonstrates better reasoning and more thoughtful evaluation.

* **~25–30 points:** You have made substantial progress, but the model or analysis is incomplete, poorly evaluated, or does not yet support a clear explanation of your decisions.

* **~30–40 points:** You produce a working predictive model and make reasonable preprocessing and modeling choices. You test at least two approaches to missing data and provide a basic evaluation of model performance.

* **40–45 points:** Your model is well designed and carefully evaluated. You justify important choices involving features, missing data, categorical variables, model selection, and evaluation, and you examine how these choices affect different patients or subsets of the population.

* **45–50 points:** Your work demonstrates particularly strong technical judgment. You compare reasonable alternatives, investigate important tradeoffs and borderline cases, evaluate consequences beyond a single overall accuracy score, and clearly explain why your final approach is defensible.

The goal is not to discover one perfectly correct model. The strongest submissions will demonstrate that you understand how technical choices can change predictions and can make and defend those choices responsibly.

## Generative AI ##

Generative AI may be used for ordinary programming assistance, debugging, explanations, and research related to this assignment. You may use AI to explain general Python code and neural-network ideas; to help diagnose errors; and, perhaps most importantly, to help you understand the statistical consequences of different preprocessing approaches.

However, the important modeling decisions must be your own. You may use AI to research individual options, but you should evaluate those options yourself and be able to explain why you made each important decision. AI should not simply decide for you:

- which features to include or exclude;
- how to handle missing data;
- how to encode categorical variables;
- how to treat demographic, ancestry, regional, or financial information;
- which model to use;
- how to define decision thresholds; or
- how to interpret the consequences of different approaches.

You are responsible for understanding all code and analysis that you submit and for checking AI-generated suggestions for errors.

Please add the appropriate references and attribution for any use of AI. And don't forget to credit any other authors whose work helps you with this assignment.

## Attribution and Acknowledgements ##

This model is the first half of a larger project. Prof. Tallman is unlikely to be the first professor to ever think of such a project. However, the ideas were devised by him independently of any other authors with inspiration and help coming from other Concordia faculty.

*Generative AI was used to assist with brainstorming and drafting portions of the instructions. The assignment concept, learning objectives, dataset design requirements, and final revisions were developed by Prof. Tallman.*

The dataset itself was created by generative AI to fit the constraints of the assignment. Prof. Tallman evaluated the dataset to verify that it fit the project constraints and then modified samples accordingly to emphasize the project requirements.

# Clinical Prediction Recommendation #

## Overview ##

The hospital is now considering how predictive models should be used in making treatment decisions. You have received records for 20 current patients who are candidates for treatment.

Your task is to revisit the model you developed in the previous assignment, apply it to these patients, and write a 3-5 page paper addressed to hospital administrators and physicians recommending how the hospital should use predictive modeling in its treatment process.

According to the hospital's previous evaluations, its prototype models can evaluate patients much faster than physicians and have performed about as well as--or slightly better than--physicians overall. However, they have also made significant mistakes. Relying entirely on physicians would avoid some algorithmic errors but would limit how many patients can be thoroughly evaluated. Neither approach is without costs.

Your recommendation should therefore consider not merely whether the model works, but what responsible use of the model would look like when its decisions affect individual patients.

## Reconsider Your Model ##

You are free to keep the model and data-processing decisions from the previous assignment, but you are **not** required to do so. The new patient records may give you reasons to reconsider how you handled missing data, categorical variables, feature selection, thresholds, or other aspects of your model.

If you make significant changes, explain what you changed, why you changed it, how you evaluated the revised approach, and whether the change affected overall model performance or the predictions for particular patients. You should pay particular attention to patients whose recommendations change under different reasonable modeling or preprocessing decisions. In contrast, if you retain your original model, explain why the new patient records did not give you sufficient reason to change it.

## Your Recommendation ##

Your paper should explain and defend both your technical decisions and your recommendation to the hospital.

Provide enough technical detail that administrators and physicians can understand why you trust your final model. Explain the important decisions involved in preparing the data, the evidence supporting those decisions, and any important limitations or uncertainties you discovered. Your discussion should consider issues such as missing data, categorical variables, ancestry and regional information, financial or access-related variables, false positives and false negatives, and cases in which reasonable modeling decisions lead to different recommendations.

Then recommend how much authority the model should have. For example, should it make treatment recommendations directly, identify patients for human review, serve only as an advisory tool, or be used in some other way? Your recommendation should acknowledge the tradeoffs involved rather than assuming that either humans or models are automatically more trustworthy. You should also make clear who ultimately makes the treatment decision, when human review is required, and what should happen when the model and physician disagree.

## Christian Faith and Professional Responsibility ##

Your recommendation must also demonstrate a thoughtful understanding of how the Christian faith connects to the decisions in this project.

You are not required to personally believe or affirm the Christian faith. You may agree with its teachings, question them, critique them, or argue that particular Christian principles should not influence the hospital's decision. Whatever position you take, your discussion should represent the Christian ideas accurately and seriously and connect them directly to the technical and human dilemmas involved in the model.

Two especially relevant Christian teachings are **love of neighbor** and **vocation**. Love of neighbor raises questions about how technical decisions affect the particular people represented by the data, while vocation raises questions about the responsibilities of data scientists, physicians, and administrators whose professional decisions affect others. You are welcome to use other biblical texts or Christian doctrines instead.

As part of this discussion, use and cite at least two relevant biblical texts and engage at least one Christian teaching or doctrine in enough depth to demonstrate that you understand it. You should also distinguish between claims that come from Christian revelation (i.e., the Bible) and claims that come from statistical evidence, medical knowledge, human reason, or experience. Scripture will not tell you which imputation method or machine-learning algorithm is best, but Christian teaching may still have something important to say about how people should use that knowledge and treat the people affected by their decisions.

## Sources ##

You may use the course readings, including Max Tegmark's *Life 3.0*, Arvind Narayanan and Sayash Kapoor's *AI Snake Oil*, John Lennox's *2084*, or other credible sources when they help support your analysis. You are not required to use every course reading or to force them into the paper.

## Writing Process and Submission ##

Before writing the paper, develop an outline that shows the major technical, ethical, and Christian claims you plan to make. **Take this outline to the Writing Studio for review before completing your final draft.**

Submit:

1. your **3-5 page paper** addressed to hospital administrators and physicians;
2. your reviewed outline;
3. the final version of the code or notebook used to produce your analysis and predictions for the 20 current patients.

Note: *The 3-5 page limit applies to the text of the paper itself; it does not include figures, tables, or other references.*

The paper should demonstrate enough technical rigor that a knowledgeable reader can evaluate your modeling decisions, but it should not become a line-by-line description of your code. Focus on the decisions that matter, the evidence supporting them, their consequences for particular patients, and the reasoning behind your final recommendation.

## Final Thoughts ##

> **Your goal is not to discover a perfectly accurate model or an easy ethical answer. Your goal is to demonstrate that you can use technical knowledge, careful reasoning, and an informed understanding of the Christian faith to make and defend a responsible professional judgment.**

## Attribution and Acknowledgements ##

This paper is the second half of a larger project. Prof. Tallman is unlikely to be the first professor to ever think of such a project. However, the ideas were devised by him independently of any other authors with inspiration and help coming from other Concordia faculty.

> *Generative AI was used to assist with brainstorming and drafting portions of the instructions. The assignment concept, learning objectives, dataset design requirements, and final revisions were developed by Prof. Tallman.*

The dataset itself was created by generative AI to fit the constraints of the assignment. Prof. Tallman evaluated the dataset to verify that it fit the project constraints and then modified samples accordingly to support his project goals.

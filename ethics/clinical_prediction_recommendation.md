# Clinical Prediction Recommendation #

## Overview ##

The hospital is now considering how predictive models should be used in making treatment decisions. You have received records for 20 current patients who are candidates for treatment.

Your task is to revisit the model you developed in the previous assignment, apply it to these patients, and write a 3-5 page paper recommending how the hospital should use predictive modeling in its treatment process.

According to the hospital's previous evaluations, its prototype models can evaluate patients much faster than physicians and have performed about as well as--or slightly better than--physicians overall. However, the models have also made significant mistakes. Relying entirely on physicians would avoid some algorithmic errors but would limit the number of patients who can be evaluated. Both approaches have significant merits and drawbacks.

Therefore, your recommendation should consider not merely whether the model works, but what responsible use of the model would look like when its decisions affect individual people.

## Reconsider Your Model ##

You are free to keep your original model and data-processing decisions from the previous assignment, but you are not required to do so. The new patient records may give you reasons to revisit and explore the consequences of how you handled missing data, categorical variables, feature selection, thresholds, or other aspects of your model.

Changing a model after examining how it behaves on these 20 patients creates an important methodological problem. Once these cases influence your modeling decisions, they can no longer serve as independent evidence that the revised model will perform well on new patients. Repeatedly adjusting a model in response to a small set of cases also risks overfitting to those particular examples.

Nevertheless, examining edge cases and testing how reasonable modeling choices affect them is an important part of model evaluation. The purpose of this assignment is therefore somewhat different from a standard model-development pipeline. You are using these patients to explore the consequences of choices you could reasonably have made—not to demonstrate that a revised model is more accurate. In practice, if this analysis were to lead you to revise the model, you would need new, independent evaluation data before making claims about the revised model's performance.

For the purposes of this assignment, if you make significant changes, explain what you changed and why you changed it. How did you evaluate the revised model and any intermediate approaches? Did your changes affect the overall model performance? You should pay particular attention to patients whose recommendations change under different preprocessing decisions. In contrast, if you decide to retain your original model, explain why the new patient records did not give you sufficient reason to change.

## Your Recommendation ##

Your paper should explain and defend both your technical decisions and your recommendation to the hospital.

Provide enough technical detail that administrators and physicians can understand why you trust your final model. Explain the important decisions involved in preparing the data, the evidence supporting those decisions, and any important limitations or uncertainties you discovered. Your discussion should consider issues such as missing data, categorical variables, ancestry and regional information, financial variables, false positives and false negatives, and cases in which reasonable modeling decisions lead to different recommendations. Keep in mind that nobody knows the true treatment outcome for the 20 new patients.

Then recommend how much authority the model should have. For example, should it make treatment recommendations directly, identify patients for human review, serve only as an advisory tool, or be used in some other way? Your recommendation should acknowledge the tradeoffs involved rather than assuming that either humans or models are automatically more trustworthy. You should also make clear who ultimately makes the treatment decision, when human review is required, and what should happen when the model and physician disagree.

## Christian Faith and Professional Responsibility ##

Your recommendation must demonstrate a thoughtful understanding of how the Christian faith connects to the decisions in this project.

You are not required to personally believe or affirm the Christian faith. You may agree with its teachings, question them, critique them, or argue that particular Christian principles should not influence the hospital's decision. Whatever position you take, your discussion should represent the Christian ideas accurately and seriously and connect them directly to the dilemmas involved in your model.

Two especially relevant Christian teachings are love for neighbor and vocation. Love of neighbor raises questions about how technical decisions affect the particular people represented by the data, while vocation raises questions about the responsibilities of data scientists, physicians, and administrators whose professional decisions affect others. You are welcome to use other biblical texts or Christian doctrines instead.

As part of this discussion, **use and cite at least two relevant biblical texts** and engage **at least one Christian teaching or doctrine** in enough depth to demonstrate that you understand it. You should also distinguish between claims that come from Christian revelation (i.e., the Bible) and claims that come from statistical evidence, medical knowledge, human reason, or experience. Scripture will not tell you which machine-learning algorithm is best or the mathematical steps to handle missing values, but Christian teaching certainly provides guidance on how people should use that knowledge and treat the people affected by their decisions.

## Sources ##

You may use the course readings, including Max Tegmark's *Life 3.0*, Arvind Narayanan and Sayash Kapoor's *AI Snake Oil*, John Lennox's *2084*, or other credible sources when they help support your analysis. You are not required to use every course reading or to force them into the paper.

## Writing Process and Submission ##

Before writing the paper, develop an outline that shows the major technical, ethical, and Christian claims you plan to make. Take this outline to the Writing Studio for review before completing your final draft.

When finished, submit:

1. your reviewed outline;
2. your 3-5 page paper; and
3. the final version of the code or notebook used to produce your analysis and predictions for the 20 current patients.

Note: *The 3-5 page limit applies to the text of the paper itself; it does not include figures, tables, or other references.*

The paper should demonstrate enough technical rigor that a knowledgeable reader can evaluate your modeling decisions, but it should not become a line-by-line description of your code. Focus on the decisions that matter, the evidence supporting them, their consequences for particular patients, and the reasoning behind your final recommendation.

## Grading Rubric ##

Your grade is based on the quality of your technical analysis, the care with which you evaluate the consequences of your modeling decisions, the strength of your recommendation, and the depth of your engagement with Christian teaching. Strong papers will integrate these elements rather than treating the technical, ethical, and Christian portions as unrelated sections.

- **~55-69 points:** Your paper demonstrates substantial progress but has important weaknesses. The technical analysis may be incomplete or insufficiently justified, the model may be evaluated only superficially, or the recommendation may not follow clearly from the evidence. Christian ideas may be mentioned but applied only loosely, inaccurately, or without a clear connection to the dilemmas raised by the model.

- **~70-79 points:** Your paper presents a working and reasonably evaluated model, explains the major preprocessing and modeling decisions, and offers a defensible recommendation to the hospital. You recognize important tradeoffs and consequences for patients. You accurately engage relevant biblical material and at least one Christian teaching, and you make a reasonable connection between Christian commitments and professional responsibility.

- **80-89 points:** Your paper demonstrates strong technical and professional judgment. You carefully justify important choices involving missing data, categorical variables, features, model selection, and evaluation; examine how those choices affect particular patients or borderline cases; and make a thoughtful recommendation that acknowledges uncertainty and competing concerns. Your Christian analysis is accurate, substantive, and integrated into the argument, with a clear distinction between claims grounded in Christian revelation and those grounded in statistics, medicine, reason, or experience.

- **90-100 points:** Your paper demonstrates exceptional integration of technical rigor, professional judgment, and Christian literacy. You compare reasonable alternatives, investigate meaningful tradeoffs and individual consequences, and use evidence to defend both your final model and your recommendation for how it should be used. Your discussion of Scripture and Christian teaching is accurate, nuanced, and directly relevant to the decisions at hand, whether you ultimately affirm, critique, or reject their use in hospital policy. The paper shows especially mature judgment about what technical evidence can establish, what it cannot establish, and the responsibilities of those whose decisions affect individual patients.

The strongest submissions will not necessarily use the most complicated model or achieve the highest overall accuracy. They will demonstrate that you understand how technical choices can change outcomes for individual people and can explain and defend those choices with rigor, humility, and responsible professional judgment.

## Generative AI ##

Generative AI use is limited for this assignment. You may use AI as a research or feedback tool, but the reasoning in the paper should be your own. You should personally wrestle with the tradeoffs, decide what you believe responsible professional practice requires, and explain how the technical evidence, ethical concerns, and Christian teachings fit together.

You may use AI to:

- help explain technical concepts that you do not understand;
- help you locate or understand relevant background information;
- diagnose problems in the error messages in the code used to analyze the 20 patients; and
- provide counterarguments to ideas, theses, evidence, and arguments that you consider and then address on your own.

However, AI may not evaluate your model's results for you or determine what conclusions you should draw from its accuracy, errors, subgroup performance, or predictions for the 20 patients. You should perform and interpret that analysis yourself. AI may not generate the substance of the paper.  In particular, it should not:

- decide what recommendation you should make to the hospital;
- determine how much authority the predictive model should have;
- decide which patients deserve human review;
- generate your ethical analysis;
- generate your Christian reflection;
- interpret biblical texts on your behalf; or
- write the final paper.

You are responsible for all claims, citations, analysis, and recommendations in your final submission. If you use AI to help research a Christian doctrine or biblical passage, verify its claims against credible theological or biblical sources rather than treating AI itself as an authority.

Please add references and attribution where appropriate.

## Final Thoughts ##

**Your goal is not to discover a perfectly accurate model or an easy ethical answer. Your goal is to demonstrate that you can use technical knowledge, careful reasoning, and an informed understanding of the Christian faith to make and defend a responsible professional judgment.**

## Attribution and Acknowledgements ##

This paper is the second half of a larger project. Prof. Tallman is unlikely to be the first professor to ever think of such a project. However, the ideas were devised by him independently of any other authors with inspiration and help coming from other Concordia faculty.

*Generative AI was used to assist with brainstorming and drafting portions of the instructions. The assignment concept, learning objectives, dataset design requirements, and final revisions were developed by Prof. Tallman.*

The dataset itself was created by generative AI to fit the constraints of the assignment. Prof. Tallman evaluated the dataset to verify that it fit the project constraints and then modified samples accordingly to support his project goals.

# Baseball Game Predictor #

Create a machine learning regression model to predict the final score of an upcoming baseball game. Everyone in the class will have an opportunity to briefly explain how his or her model works and then run it so that everyone can see its prediction.

In addition to creating the model, every student will also:

1. randomly generate a final score; and
2. guess the final score using his or her own intuition

After the game, we will compare the predictions from our machine learning models, human intuition, and random chance. I have absolutely no idea what the results will be.

I am trying to design this project so that people who are really into baseball, statistics, or machine learning can go crazy and have a lot of fun with it... while people who could not care less about baseball do not have to become experts.

## Assignment ##

Create a machine learning regression model that predicts the final score of a baseball game. Assume that the model will run immediately before any given baseball game so that information about the starting lineup could potentially be given to the model. For example, knowing the starting pitcher might be benefitial.

Your model should predict the number of runs scored by both teams. You may use two separate regression models or another reasonable approach that produces a predicted score for both teams. The model must use at least three features. Possible features might include:

- average runs scored;
- average runs allowed;
- recent offensive performance;
- batting statistics;
- starting pitcher statistics;
- bullpen performance;
- home-field advantage;
- winning percentage;
- days rest; and
- any other statistics that you believe might help predict the score.

You are not required to use any particular statistics. Part of the project is deciding which information you believe is useful.

## Hints ##

1. Regression algorithms provide continuous results, but baseball does not have fractional runs. Make sure your final prediction is expressed as a valid baseball score.

2. Finding a few key statistics that really affect run scoring may be more useful than collecting every baseball statistic you can find. This favors the baseball fans.

3. Baseball has an enormous amount of statistical data available. More features do **not** automatically produce a better model. Simpler is probably better. This favors the non-baseball fans.

4. Be careful about data leakage. When training on a historical game, your model should only use information that would reasonably have been available before that game was played. For example, using a team's full-season statistics to predict a game played halfway through that same season may accidentally include information from the future.

5. Starting pitchers can have a substantial effect on baseball games. If your model uses starting-pitcher information, think carefully about how you will associate historical games with their starting pitchers and what statistics would have been known before each game.

6. Free APIs and statistical websites sometimes have download limits or take significant time to query. Download historical data when appropriate and save it in a JSON or CSV file. Your model, which you may run dozens or hundreds of times during development, can then read the locally cached data instead of repeatedly downloading the same information.

7. Extra innings happen. Do not worry about restricting your prediction to nine innings. Predict the game's final score.

## Data ##

The models may use instructor-provided data, publicly available baseball statistics, APIs, or a combination of these sources. You are welcome to collect additional data if you believe it will improve your model. Make sure that you understand where your data came from and how it is being used.

If your model downloads statistics from a webservice at runtime, save a local copy whenever practical so that your project does not depend on repeatedly accessing an external service.

## Generative AI ##

You are encouraged to use Generative AI to help parse, clean, and organize baseball data in a form that is useful for machine learning. AI can be particularly useful for repetitive tasks such as:

- reading unfamiliar JSON or CSV formats;
- combining data from multiple files;
- converting data into a Pandas DataFrame;
- cleaning column names or values;
- writing ordinary data-processing code; and
- helping you understand unfamiliar baseball statistics.

However, the strategy behind the model should be yours. You must understand all code that ends up in your project and examine AI-generated code for errors. My suggestion is to combine AI's ability to crank out normal or repetitive code with your own strategy and thinking.

Please add references and attribution where appropriate.

## In-Class Prediction ##

During class you will briefly describe your model and run it to produce your official prediction. You will submit three predictions:

1. Random Prediction: a score generated randomly;
2. Human Prediction: your own guess at the final score; and
3. Model Prediction: the score predicted by your regression model;

The goal of this comparison is not to prove that one approach is always superior. A single baseball game contains a great deal of randomness. We are simply going to see what happens.

## Grading — 50 Points ##

Your grade is based primarily on the quality of your model and your reasoning, not on whether you correctly predict the actual score. Baseball is unpredictable, and a well-designed model can still make a terrible prediction for one particular game.

- **~25–30 points:** You have a model that seems basically appropriate but still has bugs that prevent it from running or producing a usable prediction.

- **~30–40 points:** Your model produces a valid score prediction, although it may be poorly trained, use fairly simple features, or depend heavily on manually prepared or hardcoded CSV files.

- **40–45 points:** Your model uses reasonable features, processes and normalizes data appropriately when necessary, avoids obvious data leakage, and produces a reasonably sound prediction. At least some of the data preparation or retrieval is automated.

- **45–50 points:** Your model is well designed, uses appropriate features and evaluation techniques, handles its data carefully, and relies substantially on automated data retrieval or preparation so that it could be adapted relatively easily to predict future baseball games.

The student with the most accurate model will win bragging rights. If a model predicts the score will be Home 3 - Visitor 2 and the final score turns out to be Home 5 - Visitor 1, then it would have an accuracy of +/-6 runs.

## Groups ##

No groups for this project.

You are welcome to bounce ideas off of each other and share data, but I want each person to develop a unique model. If you work closely with another student, make sure that your models differ in at least one meaningful feature or modeling decision.

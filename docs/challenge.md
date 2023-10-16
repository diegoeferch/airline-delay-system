# PART 1

The chosen model to complete this part was the XGBoost Classifier that uses the top 10 features
and the class balancing.

Even though, the precision and recall metrics are quite similar with the Logistic Regression model that
also uses the top 10 features and class balancing, there is a noticeable difference between the number of samples
correctly classified.

At least in the "On time" (or 0) category, there is a difference of more than 60 samples being misclassified as
"Delayed" on the Logistic Regression model.
Almost 37% of the samples labeled as "On time."

With all the other metrics being so similar, this was the reason to use the XGBoost Classifier for this task.
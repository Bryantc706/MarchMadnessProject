# MarchMadnessProject
This project centers on predicting NCAA March Madness game outcomes using advanced machine learning techniques. Multiple datasets—including teams, games, seeding, and tournament details—were integrated and thoroughly cleaned, setting the stage for effective feature engineering. The process involved transforming raw data into informative predictive variables, ensuring the dataset was robust enough for modeling. A variety of models were explored throughout the project, with a particular emphasis on ensemble methods. The final approach utilized a stacking ensemble strategy that combined tree-based models such as Gradient Boosting and Random Forest, enhanced with parameter-tuned XGBoost and LightGBM to capture diverse aspects of the game dynamics. In some iterations, a neural network was also incorporated, further diversifying the ensemble and capturing more complex patterns from the data.

To assess the performance of these models, evaluation metrics such as accuracy, ROC AUC, and the Brier score were employed, with the Brier score playing a pivotal role in ensuring reliable probability estimates. The experimental results demonstrated that combining multiple models tended to yield better-calibrated and more accurate predictions compared to single-model approaches. Finally, the project culminated in the preparation of a submission file formatted according to the competition's guidelines. Leveraging the provided template, the final file contained 131,407 rows, each with a unique game identifier and a corresponding probability prediction. Overall, the project provides a comprehensive demonstration of how robust data preparation, innovative model combination, and careful evaluation can come together to tackle the challenging task of predicting March Madness outcomes.







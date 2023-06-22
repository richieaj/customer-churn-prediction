# Predicting & Analysing Customer Churn using Machine Learning Algorithms
- In an era of increasing customer expectations and fierce competition, understanding customer behavior and optimizing customer retention is paramount for any successful bank. Our project aims to leverage advanced machine learning techniques to predict customer churn for targeted marketing strategies.

- The project starts by analyzing a comprehensive dataset containing various attributes of bank customers, including credit scores, estimated salaries, balances, and geographical locations.

- We have employed logistic regression and Random forest, the two powerful classification algorithms to predict customer churn. By training a logistic regression and Random forest model on historical customer data, we can accurately identify patterns and factors contributing to customer attrition. This enables banks to retain at-risk customers and optimize their retention strategies proactively.

- Our project leverages advanced machine learning techniques, specifically random forest, and logistic regression models, to analyze customer behavior in the banking industry. Through meticulous analysis and evaluation, we assess the performance of these models in their respective territories, providing valuable insights for customer retention and strategic decision-making.

# Explanation of the dataset

The dataset used in our project consists of customer data from a banking institution. It includes various features such as customer demographics, financial indicators, and banking interactions. Our analysis aims to understand customer behavior, identify patterns, and predict customer churn. By exploring the dataset, we gain insights into factors influencing customer decisions, enabling the bank to tailor their services, improve customer satisfaction, and optimize retention strategies. The dataset provides a comprehensive view of the customer landscape, empowering the bank to make data-driven decisions for sustainable growth and competitive advantage.

# Pre-processing techniques used

In our project, we employed several pre-processing techniques to prepare the dataset for analysis. Firstly, we handled missing values by imputing them with appropriate values or removing them from the dataset. Next, we performed feature scaling to standardize the numerical features, ensuring they have similar scales and preventing any bias in the analysis. We also applied one-hot encoding to convert categorical variables into numerical representations, allowing us to incorporate them into our models effectively.

# Data analysis
## Churn Rate by Gender
  ![image](https://github.com/richieaj/customer-churn-prediction/assets/87382894/d2cd2ea0-004f-44ae-8092-8a6b0dd0cb3f)

- Based on the analysis of customer churn count by gender using a bar chart, it can be concluded that among the 10,000+ data points, females exhibit a slightly higher churn rate than males. The churn count for females is approximately 1,000, while for males, it is around 800. This could be attributed to differences in customer preferences, needs, or satisfaction levels.

## Variation in customer needs and preferences: 
 - Females may have different expectations or experiences with the company's products or services, leading to a higher likelihood of churn.
## Marketing and targeting strategies: 
- The company's marketing campaigns or targeting strategies may have resonated more with males, resulting in a relatively lower churn rate for that segment.
Customer satisfaction and engagement: Factors such as customer support, product quality, or engagement initiatives may have a varying impact on different gender groups, influencing their churn behavior.

## Churn Distribution by Age Group
![image](https://github.com/richieaj/customer-churn-prediction/assets/87382894/95e3197f-9da1-4a21-9c93-c490f90f76db)

- Based on the pie chart depicting the distribution of churn percentages across different age groups, it can be observed that age groups above 30, particularly the age group of 40-49, exhibit a higher churn percentage. These age groups, including those above 30, are likelier to churn than younger age groups.
- The analysis suggests that customers in these age ranges may experience various life events or changing needs contributing to their higher propensity to churn. Understanding the underlying factors driving churn within these age groups can help businesses develop targeted retention strategies to address these customers' specific needs and concerns and reduce churn rates.

## Churn Distribution by Countries

![image](https://github.com/richieaj/customer-churn-prediction/assets/87382894/a8460a66-2a9e-47da-a3c2-95200e80c14a)

- Based on the churn distribution by countries, it is evident that France is the location from which most churns originate. 
- While the other two locations have a relatively lower percentage of churn, it is essential to note that they still have a significant number of churn instances.
- This suggests that location may or may not be a significant contributing factor for churn prediction. Other factors such as customer demographics, product offerings, customer service, or competitive landscape could influence churn behavior to a more significant extent.

## Churn Distribution by number of products

![image](https://github.com/richieaj/customer-churn-prediction/assets/87382894/c7efdd7e-5f47-47b9-92f7-1ddc5213e14b)

- The graph illustrating the distribution of churns based on the number of products subscribed by customers reveals that the number of products is a significant factor contributing to churn. The highest churn rate is observed among customers who have subscribed to only one product. On the other hand, customers with four products show the lowest churn rate.

- One possible reason for this pattern is that customers with only one product may not be fully engaged or satisfied with the service, making them more likely to churn. They might perceive limited value or lack sufficient options to meet their needs. In contrast, customers with four products will likely have a more comprehensive relationship with the service provider. They may benefit from a wider range of offerings, enjoying more value, and experiencing greater satisfaction, reducing their likelihood of churning.

- It is crucial for the service provider to assess the underlying reasons behind the higher churn among customers with a single product and consider strategies to enhance the value proposition for this segment. This could involve improving single product features, benefits, or pricing or offering incentives to encourage customers to subscribe to additional products.

## Churn percentage Has Credit card 

![image](https://github.com/richieaj/customer-churn-prediction/assets/87382894/b432dfac-f5de-4df0-94f7-5e7a7fad8658)

- The analysis of churn based on whether customers have a credit card or not indicates that this factor has not been a significant contributor to churn in the past. Both customers who have credit card and those who do not exhibit a churn rate of approximately 20%. This suggests that the presence or absence of a credit card has not played a prominent role in influencing customer churn.
  
- While having a credit card may have its own advantages or benefits, it does not appear to be a decisive factor in customer churn within the given dataset. Other factors such as product satisfaction, service quality, or customer engagement may substantially impact churn rates. To better understand the reasons behind customer churn, it would be beneficial to explore other variables or factors that could have a stronger correlation with churn, such as customer demographics, product features, or customer satisfaction metrics.

## Churn Percentage based on active memmbers 

![image](https://github.com/richieaj/customer-churn-prediction/assets/87382894/b8d59795-59ea-42d5-86b1-949f0a7d585f)

- The analysis of churn based on customer activity as a bank member indicates that this factor plays a significant role in customer churn.
- The graph demonstrates that customers who are active members have a lower churn rate than those who are not actively participating in the bank. This suggests that customer engagement and involvement in bank activities positively impact customer retention.
- Active members are less likely to churn, while non-active members are more likely to churn. This implies that customers who actively engage with the bank, utilize its services, and participate in various activities are likelier to remain loyal and continue their relationship with the bank.
- To reduce churn and improve customer retention, the bank must foster customer engagement, provide personalized services, and encourage active participation through various programs, offers, and incentives. The likelihood of churn can be effectively reduced by strengthening the bond between the bank and its customers.


# Machine learning algorithms: logistic regression and random forest

## Logistic Regression:

- Logistic Regression is a popular classification algorithm used in machine learning. It is particularly suited for binary classification tasks, which aim to predict whether an instance belongs to one class or another. The algorithm calculates the probability of an instance belonging to a particular class by applying a logistic function to a linear combination of the input features. The decision boundary is determined based on this probability, classifying the instances accordingly.


- The logistic regression model uses the logistic function (also known as the sigmoid function) to transform the linear combination of the independent variables into a probability value between 0 and 1. This probability represents the likelihood of the binary outcome occurring. The logistic function has an "S" shaped curve, which allows the model to capture non-linear relationships between the independent variables and the outcome.
An optimization algorithm called maximum likelihood estimation is used to train a logistic regression model. This algorithm adjusts the model's parameters to maximize the likelihood of observing the actual outcomes given the predicted probabilities. The parameters are estimated using iterative methods, such as gradient descent, which iteratively updates the parameter values to minimize the difference between the predicted probabilities and the actual outcomes.


## Random Forest:

- Random Forest is an ensemble learning algorithm that combines multiple decision trees to make predictions. It creates a set of decision trees using a random subset of features and data samples, and then combines their predictions to generate the final output. Random Forest is known for its ability to handle high-dimensional data and capture complex interactions between features. It can handle both classification and regression tasks and is robust against overfitting.

- Random Forest implementations involve several steps in the backend of machine learning. Initially, the data undergoes preprocessing, including cleaning, handling missing values, encoding categorical variables, and scaling or normalizing numerical features. The training phase involves constructing decision trees by recursively splitting the data based on selected features and thresholds using an algorithm like CART (Classification and Regression Trees). Each decision tree is trained on a subset of the training data, incorporating bagging and feature randomness to create an ensemble. During prediction, new data points are passed through each decision tree, and the final prediction is determined by aggregating the individual tree predictions, either through majority voting for classification or averaging for regression. Finally, the trained Random Forest model is evaluated using metrics like accuracy, precision, recall, F1-score, or mean squared error (MSE) to assess its performance.


## Importance of standardizing and hyperparameter tuning

Standardizing and hyperparameter tuning are two important steps in machine learning model development. 

Standardizing, also known as feature scaling or normalization, is crucial to ensure that all features or variables in a dataset are on a similar scale. It involves transforming the data so that it has zero mean and unit variance. Standardizing the data helps in several ways: 

1. It prevents features with larger scales from dominating the learning process, as some machine learning algorithms are sensitive to the scale of the input features.

2. It ensures that all features contribute equally to the model training process, avoiding biased results.

3. It helps improve the convergence and performance of many optimization algorithms.

Hyperparameter tuning involves selecting the optimal values for the hyperparameters of a machine-learning model. Hyperparameters are parameters that are set before the learning process begins and control the behavior and performance of the model. Examples of hyperparameters include the learning rate, regularization strength, number of hidden layers in a neural network, etc. 

Hyperparameter tuning is important because it allows us to fine-tune the model to achieve the best performance on the given dataset. By systematically exploring different combinations of hyperparameter values, we can optimize the model's performance, improve accuracy, reduce overfitting, and enhance generalization to new, unseen data.

Overall, standardizing the data ensures that features are on a consistent scale, while hyperparameter tuning helps optimize the model's performance and generalization capabilities, leading to more accurate and robust predictions.

Conclude: final inference from the model: accuracy, MSE, better model, and why.

Based on the project analysis, we compared the performance of Logistic Regression and Random Forest models for customer segmentation. Here is a comparative study between the two models:
1. Accuracy: 
   - Logistic Regression: Before hyperparameter tuning, the accuracy was 81.16%, which improved slightly to 81.21% after tuning.
   - Random Forest: The accuracy before tuning was 85.97%, and after hyperparameter tuning, it increased to 86.32%.
   - In terms of accuracy, both models performed reasonably well, with Random Forest exhibiting a slightly higher accuracy after hyperparameter tuning.

2. Mean Squared Error (MSE):
   - Logistic Regression: The MSE for Logistic Regression was 0.188, indicating a moderate prediction error level.
   - Random Forest: The MSE for Random Forest was 0.137, suggesting a lower error than Logistic Regression.
   - Random Forest performed better regarding MSE, indicating more accurate predictions and a better fit to the data.

3. Better Model:
   - Based on the analysis, the Random Forest model outperformed Logistic Regression regarding accuracy and MSE.
   - Random Forest showed higher accuracy after hyperparameter tuning and a lower MSE, indicating improved performance and more accurate predictions.
   - Therefore, our project considers the Random Forest model the better model for customer segmentation.

The choice of the better model depends on the specific requirements and objectives of the project. In this case, Random Forest demonstrated superior accuracy and error metrics performance, suggesting its suitability for the customer segmentation task. However, it is essential to consider other factors, such as model interpretability, computational efficiency, and business constraints, when selecting the final deployment model.







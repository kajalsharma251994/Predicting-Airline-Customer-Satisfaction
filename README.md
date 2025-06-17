# Predicting-Airline-Customer-Satisfaction

**Project Overview**

The airline industry operates in a highly competitive environment where customer satisfaction
serves as a critical determinant of success. In various literature, customer satisfaction is often
defined as an emotional response which stems from a cognitive process of the service received in
relation to the costs incurred to obtain it (Park et al., 2019). Moreover, it is through the
experience of positive emotions that customers will have higher satisfaction whereas those
experiencing negative emotions in regard to a particular airline issue will face obstacles when it
comes to achieving satisfaction (Park et al., 2019). Thus, many airlines must strive to deliver
exceptional service to maintain customer loyalty, attract new passengers, and maintain a
competitive edge. Satisfied customers are more likely to recommend airlines to others, increasing
brand reputation and market share.

Despite technological advancements and service improvements, the industry faces persistent
challenges, including delays, overbooking, and inconsistent service quality, which can negatively
impact customer satisfaction. Understanding these dynamics is vital as airlines recover from
certain disruptions such as the COVID-19 pandemic and adapt to constantly evolving customer
expectations.

**Research Questions and Relevance**

To address these challenges, our project focuses on two critical research questions:

1. What are the most influential factors that determine customer satisfaction in airline
travel?
2. How can airlines leverage predictive modeling to anticipate satisfaction levels and
prioritize service enhancements?

Answering these questions is crucial because it allows airlines to address specific pain points and
areas for improvement. That is, understanding whether factors such as cleanliness, seat comfort,
or in-flight entertainment have the greatest impact on satisfaction can empower airlines to make
targeted investments in their operations. Predictive modeling provides a data-driven approach to
anticipate customer concerns, allowing airlines to address potential issues proactively before they
escalate.

By analyzing airline customer satisfaction data through machine learning techniques, this project
contributes to the broader goal of service excellence in the industry. The findings can inform
strategic decisions, enhance resource allocation, and improve customer-centric practices,
benefiting both passengers and airline operators.

**Data Characteristics**

The "Airline Customer Satisfaction" dataset, sourced from Kaggle, comprises 22 variables and
129,880 observations.The dataset includes both categorical and numerical features related to
customer satisfaction and airline services. Key variables include customer demographics (e.g.,
customer type, age, and gender), flight details (e.g., flight distance, class, and departure/arrival
delays), service ratings (e.g., seat comfort, in-flight entertainment, food quality), and operational
metrics (e.g., baggage handling, online boarding, cleanliness). These features provide a
comprehensive view of the factors influencing customer satisfaction in our project.

The target variable, "satisfaction," is categorical with two levels: "satisfied" and "dissatisfied."
This binary outcome makes the dataset particularly well-suited for classification tasks. The
independent variables, consisting of a mix of numerical (e.g., flight distance, service ratings) and
categorical (e.g., customer type, class) data, are potential predictors for customer satisfaction.
To ensure consistency and reproducibility in model evaluation, the dataset was split into an
80:20 training and testing set, using a random seed of 42.

The dataset directly aligns with the two core research questions.

Firstly, by including a wide range of features, it provides a comprehensive view of the variables
that influence customer satisfaction. Key variables such as service quality ratings (e.g., seat
comfort, in-flight entertainment, food and drink) and operational metrics (e.g., delays, flight
distance) capture essential aspects of the customer experience, offering insights into what drives
customer satisfaction.

Secondly, the dataset helps explore how predictive modeling can be used to forecast customer
satisfaction. By using classification algorithms, airlines can spot patterns in the data, focus on
service improvements and enhance overall customer experience.

**Variable Table:**
<img width="847" alt="Screenshot 2025-06-16 at 22 33 43" src="https://github.com/user-attachments/assets/3283cfa8-e410-42e4-a7ef-c2a7e17349d9" />

**Model Construction**

Before building the models, feature selection was performed to identify the most influential
variables and assess its impact on performance. Random Forest, an ensemble learning technique
that combines predictions from multiple decision trees, was used to capture nonlinear
relationships and assess feature importance. XGBoost, an optimized gradient boosting model,
was employed for its speed and effectiveness in handling complex, imbalanced datasets.

**Random Forest**

Justification: Random Forest handles both categorical and continuous variables effectively,
accommodates high-dimensional data, and is resistant to overfitting. Additionally, it provides
feature importance rankings, allowing stakeholders to pinpoint the factors most critical to
customer satisfaction. As for use in the airline industry, Random Forest identifies nonlinear
relationships among service quality factors, helping airlines prioritize areas like inflight
entertainment and seat comfort to enhance passenger satisfaction.

Model Building: To build the Random Forest model, the target variable (satisfaction) was
converted to numeric values using LabelEncoder to ensure compatibility with the
RandomForestClassifier. In addition, each categorical feature was one-hot encoded, creating
binary columns for each category, and to prevent multicollinearity, the first category of each
variable was excluded. Further, a variance threshold of 0.03 was applied to drop low variance
features. After splitting the dataset into 80% training and 20% testing, the model achieved 96%
overall accuracy. For the “dissatisfied” class, Precision was 0.94, Recall 0.96, and F1-Score 0.95;
for the “satisfied” class, Precision was 0.99, Recall 0.95, and F1-Score 0.96. Feature importance
graph indicates that Inflight Entertainment and Seat Comfort were key predictors.

<img width="660" alt="Screenshot 2025-06-16 at 22 33 32" src="https://github.com/user-attachments/assets/5abe56ed-bdae-4d15-af16-35b52c88b701" />


ROC Curve: The ROC curve displayed strong discriminative power with a steep initial rise and
a near-flat horizontal section at a high True Positive Rate (TPR). The curve approached a TPR of
0.99, indicating near-perfect sensitivity, with a small increase in false positives as the threshold
decreased.

<img width="546" alt="Screenshot 2025-06-16 at 22 33 24" src="https://github.com/user-attachments/assets/978a3303-488b-46c6-8052-a180e03a7366" />


Confusion Matrix: The confusion matrix showed that the model correctly identified 13,606
satisfied passengers and 11,236 dissatisfied passengers. However, it misclassified 439
dissatisfied passengers as satisfied and 695 satisfied passengers as dissatisfied, demonstrating
strong performance with minimal errors.

<img width="497" alt="Screenshot 2025-06-16 at 22 33 17" src="https://github.com/user-attachments/assets/0d190e99-7f30-4de1-bd40-aef1b1ed9e2b" />


**XGBoost**

Justification: XGBoost excels in capturing complex relationships between features, particularly
non-linear, making it well-suited for predicting customer satisfaction, where the number of
satisfied vs. dissatisfied passengers may not be equal. XGBoost works efficiently with both
numerical and categorical features and is highly effective at improving prediction accuracy. As
for use in the airline industry, XGBoost is appropriate for exploring weaker interactions between
features. For example: the type of customer, flight distance, seat comfort and satisfaction level
which are nonlinear and low order.


The XGBoost model was built to classify customer satisfaction using a structured approach. The
target variable was encoded using LabelEncoder, while a parameter grid was defined for
hyperparameter tuning, including settings for learning rate, the number of estimators, maximum
tree depth, and sampling techniques. A GridSearchCV with 5-fold cross-validation identified the
best parameters, which included a learning rate of 0.2, a maximum depth of 7, and 200
estimators. The model was then evaluated using precision-recall and ROC curves across
thresholds (0.3, 0.5, and 0.7), providing insights into its accuracy, precision, recall, and AUC.
This process highlighted the model's performance and allowed for threshold-specific metrics and
visualizations.


The performance evaluation of the XGBoost model highlights its strengths across various
thresholds. At a threshold of 0.3, the model achieves high recall (0.9741), making it effective for
minimizing false negatives, though precision (0.9340) is slightly lower. A threshold of 0.5 offers
the best balance, with high precision (0.9653) and recall (0.9550), resulting in the highest
accuracy (0.9563) and an excellent F1 score (0.9601). At a threshold of 0.7, the model achieves
the highest precision (0.9852), reducing false positives but at the cost of lower recall (0.9261).
Overall, threshold 0.5 is ideal for general applications, while 0.3 suits scenarios prioritizing
recall, and 0.7 is better for high-stakes cases where precision is critical.

The confusion matrices for the XGBoost model at thresholds of 0.3, 0.5, and 0.7 illustrate key
trade-offs between sensitivity and precision. A 0.3 threshold maximizes sensitivity (13,931 true
positives) but allows more false positives (984), while a 0.7 threshold prioritizes precision (199
false positives) at the cost of recall (1,057 false negatives). The 0.5 threshold offers the best
balance, achieving high accuracy with 13,658 true positives and 11,184 true negatives. These
results demonstrate how threshold selection tailors the model to specific priorities, such as recall
or precision.

The observed Precision-Recall (PR) curve illustrates the behavior of a well-performing classifier
and highlights key trade-offs between precision and recall. Initially, precision is perfect (1) at
recall 0, as the model selectively predicts only the most confident positive cases, avoiding false
positives. Yet as recall increases, precision declines gradually because the model includes less
confident cases, introducing false positives. When recall reaches 1, a sharp drop in precision
occurs, reflecting the model’s difficulty in maintaining precision while identifying all positive
cases. Overall, the initial flat region of the curve indicates high confidence and precision for a
subset of cases, while the gradual decline demonstrates the model’s capacity to balance false
positives and false negatives. Practical threshold selection, such as using 0.3 or 0.5 where the
trade-off is optimal, can help balance these metrics effectively. The sharp precision drop at full
recall highlights the trade-offs inherent in classification, emphasizing the need for careful
optimization based on the application’s priorities.

<img width="462" alt="Screenshot 2025-06-16 at 22 33 08" src="https://github.com/user-attachments/assets/54d52aeb-1e22-4b46-88c5-43170a045654" />


The ROC curve demonstrates strong performance of the model, with AUC scores exceeding 0.9
across all thresholds, signifying excellent discrimination between positive and negative classes.
The plotted curves consistently trend upward and to the right, reflecting the model’s behavior as
the threshold decreases: the true positive rate (TPR) increases as more true positives are
identified, while the false positive rate (FPR) also rises due to more negatives being misclassified
as positives. An effective classifier generates a curve that quickly ascends toward the top-left
corner, representing high TPR with low FPR before leveling off. This desirable pattern is evident
in the model's ROC curves, which closely hug the top-left corner, reinforcing its ability to
achieve a strong trade-off between sensitivity and specificity across thresholds.

<img width="551" alt="Screenshot 2025-06-16 at 22 32 56" src="https://github.com/user-attachments/assets/9fefde36-14e1-4056-b7bd-ab5511128405" />


Finally, with feature importance for the XGBoost model, it highlights the relative contribution of
each variable based on their F scores, indicating how often they are used to split the data in the
ensemble's decision trees. The most significant feature, Flight Distance (2067.0), underscores its
dominant role in influencing the target variable, suggesting that longer or shorter flight distances
are critical to the model's predictions. Age (1732.0) follows closely, reflecting the importance of
customer demographics. Seat Comfort (879.0) and Leg Room Service (867.0) rank highly,
emphasizing the significance of passenger experience during the flight. Logistic factors such as
Gate Location (859.0) and Departure/Arrival Time Convenience (838.0) also play a key role,
pointing to the influence of accessibility and timing. Features like Baggage Handling (794.0),
Cleanliness (767.0), Check-in Service (710.0), and Inflight Entertainment (702.0) further
contribute to the model's predictions, showcasing the comprehensive impact of customer service
elements. Together, these top features provide the bulk of the model’s predictive power, offering
valuable insights into the factors driving the outcomes and highlighting areas for optimization or
focus.

**K-Fold Cross-Validation:**

Both the XGBoost and Random Forest models show strong performance, with high accuracy
across 5-fold cross-validation. XGBoost's accuracy ranges from 94.55% to 95.04%, with an
average of 94.84%. The Random Forest model performs similarly, with accuracies between
95.06% and 95.32%, and an average of 95.21%. While both models demonstrate reliability,
Random Forest slightly outperforms XGBoost in average accuracy.

Offered Value & Key Insights for Business Stakeholders

Our analysis offers valuable insights for business stakeholders, helping airlines improve
operations and create a better travel experience for passengers. By understanding what truly
drives customer satisfaction, stakeholders can prioritize investments to deliver more personalized
experiences that not only meet but exceed expectations. For example, inflight entertainment, seat
comfort, and legroom emerged as top factors influencing satisfaction, highlighting the need to
invest in high-quality systems and ergonomic seating. Seamless online booking, accessible
customer support, cleanliness, and efficient baggage handling also stood out as areas where
improvements can make a big difference in the customer journey. Operational factors like flight
distance and convenient schedules play a crucial role too, offering opportunities to optimize
travel plans and enhance convenience. Additionally, by identifying differences in what drives
satisfaction for loyal versus disloyal customers, airlines can create more effective strategies to
build and maintain loyalty. These insights empower stakeholders to make informed decisions,
ensuring a smoother travel experience for passengers, strengthening customer loyalty, and
helping airlines stand out in a competitive industry.

**Conclusion**

This project highlights the critical drivers of customer satisfaction in the airline industry and
provides actionable insights that can help stakeholders enhance passenger experiences. By
leveraging advanced machine learning models, airlines are equipped to make data-driven
decisions that prioritize improvements, optimize resource allocation, and gain a competitive
advantage in the market. Key drivers such as inflight entertainment, seat comfort, legroom, and
digital touchpoints like online booking and support emerged as the most influential factors
impacting customer satisfaction. Airlines can target improvements by focusing on operational
aspects, such as cleanliness, baggage handling, and flight scheduling, which are essential to
service quality. Additionally, insights into the behavior of loyal versus disloyal customers offer
strategic opportunities to design tailored retention strategies and allocate resources more
effectively. The positive impact on customer satisfaction not only boosts retention but also
strengthens brand reputation and market positioning. Furthermore, the model framework
developed in this analysis can be expanded to incorporate more customer experience data or
adapted to other service industries, offering broader insights for business optimization.


**References**

Park, E., Jang, Y., Kim, J., Jeong, N. J., Bae, K., & del Pobil, A. P. (2019). Determinants of
customer satisfaction with airline services: An analysis of customer feedback big
data. Journal of Retailing and Consumer Services, 51, 186–190.
https://doi.org/10.1016/j.jretconser.2019.06.009

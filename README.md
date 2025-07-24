# JM Financial Fraud Detection

This project analyzes **1 million anonymized bank transactions** to detect fraudulent activity.  
The work is structured across multiple Jupyter notebooks that prepare, explore, and model the dataset.

## Project Structure
- **`eda.ipynb`** – Exploratory Data Analysis (EDA)  
  - Examines the distributions of transaction amounts, balances, and fraud occurrences.  
  - Visualizes correlations, identifies outliers, and highlights patterns for feature engineering.

  **Univariate Analysis**
* Take a closer look at the numeric features in your dataset. How are these values distributed and what might this tell you about how most transactions behave compared to a few **rare** ones?

**Bivariate Analysis**
* When comparing different numerical features against one another, do any interesting patterns emerge for transactions marked as fraudulent? Are there particular regions or ranges where these transactions seem to concentrate?  
* How do types of transaction relate to the typical amounts involved? Are some types of transactions consistently larger or smaller than others?
* Do transaction amounts vary when you compare fraudulent and non-fraudulent transactions across different transaction types? What patterns emerge when you look at both fraud status and transaction type together?  
* Consider how well the system's built-in fraud flag (`isFlaggedFraud`) aligns with actual fraudulent activity. Are there mismatches? What does this tell you about the system's current performance?  

- **`transform.ipynb`** – Data Transformation  
  - Cleans and preprocesses the raw dataset.  
  - Handles skewed distributions, encodes categorical variables, balances the classes (Random Oversampling & SMOTE), and engineers additional features for fraud detection.

  **Data Transformation Questions**
* Does your model contain any missing values or "non-predictive" columns? If so, which adjustments should you take to ensure that your model has good predictive capabilities?
* Do certain transaction types consistently differ in amount or fraud likelihood? If so, how might you transform the type column to make this pattern usable by a machine learning model?
* After exploring your data, you may have noticed that fraudulent transactions are rare compared to non-fraudulent ones. What challenges might this pose when training a machine learning model? What strategies could you use to ensure your model learns meaningful patterns from the minority class?
* Are there interaction effects between variables (e.g., fraud and high amount and transaction type) that aren't captured directly in the dataset? Would it be helpful to manually engineer any new features that reflect these interactions?
* (Bonus/Optional) Are there interaction effects between variables (e.g., fraud and high amount and transaction type) that aren't captured directly in the dataset? Would it be helpful to manually engineer any new features that reflect these interactions? 

- **`model_train.ipynb`** – Model Training  
  - Implements machine learning models (Logistic Regression, Random Forest, Gradient Boosting) to classify fraudulent vs. non-fraudulent transactions.  
  - Includes hyperparameter tuning, evaluation (accuracy, precision, recall, F1), and confusion matrix visualizations.

  * Is this a classification or regression task?  
* Are you predicting for multiple classes or binary classes?  
* Given these observations, which 2 (or possibly 3) machine learning models will you choose?  

After selecting your models, you will follow the steps listed in the notebook. Use your classroom notes, recordings, and labs to answer questions and create ML models.

- **`data/`** – Contains the raw dataset (`bank_transactions.csv`) and processed balanced datasets for modeling (`processed_data/`).

## Data Dictionary

This dataset contains a mix of categorical and numerical variables. The data-dictionary below describes what each column represents:

* Type: The type of transaction   
* Amount: The amount of money transferred   
* NameOrig: The origin account name  
* OldBalanceOrg: The origin accounts balance before the transaction 
* NewBalanceOrig: The origin accounts balance after the transaction   
* NameDest: The destination account name   
* OldbalanceDest: The destination accounts balance before the transaction 
* NewbalanceDest: The destination accounts balance after the transaction 
* IsFlaggedFraud: A “naive” model that simply flags a transaction as fraudulent if it is greater than 200,000 (note that this currency is not USD)   
* IsFraud: Was this simulated transaction actually fraudulent? In this case, we consider “fraud” to be a malicious transaction that aimed to transfer funds out of a victim’s bank account before the account owner could secure their information. (This will be your target variable)   

Note that not all variables are important for this prediction task. Namely, all bank accounts are susceptible to being targets of fraudulent activity, so we most likely want to remove all columns that identify bank account information, and solely observe numerical predictors.

## Key Findings
- The dataset is **highly imbalanced**: only ~1,300 fraudulent transactions (0.13%) out of 1 million.  
  Balancing (Random Oversampling and SMOTE) was required before modeling.

- **Transaction behavior patterns**:
  - Fraudulent cases were mostly tied to **high-value `TRANSFER` and `CASH_OUT` transactions**, while `PAYMENT`, `DEBIT`, and `CASH_IN` rarely involved fraud.
  - Fraudulent activity tended to occur at **extreme transaction sizes and account balances**, far from the dense clusters of legitimate activity.
  - The system’s built-in `isFlaggedFraud` flag was **almost useless**, catching just 1 fraud case out of 1,297.

- **Model performance (preliminary)**:
  - Logistic Regression, Random Forest, and Gradient Boosting all achieved **high recall (98–99%) and F1-scores above 0.97** on the balanced datasets.
  - Gradient Boosting and Random Forest slightly outperformed Logistic Regression in recall while maintaining strong precision.

## Liscense 

I do not claim any data used, it was solely for educational use to learn how to build machine learning algorithm(s).


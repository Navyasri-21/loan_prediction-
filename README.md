# Loan Prediction using Machine Learning
# Project Overview

Loan approval is an important task for financial institutions. Banks must evaluate whether an applicant is likely to repay a loan based on several factors such as income, education, employment status, credit history, and property details.

This project builds a Machine Learning model to predict loan approval status using applicant information. The model learns patterns from historical data and predicts whether a loan should be approved or rejected.

The project demonstrates the complete Machine Learning pipeline, including:

>Data preprocessing

>Data visualization

>Feature encoding

>Model training

>Model evaluation

# Dataset Description

The dataset contains loan application details of applicants.

>Input                   
>Feature	                
>Gender	                
>Married	              
>Dependents	            
>Education	              
>Self_Employed	          
>ApplicantIncome	       
>CoapplicantIncome	      
>LoanAmount	            
>Loan_Amount_Term	      
>Credit_History	        
>Property_Area	        
>Target Variable

>Variable
>Loan_Status	Y = Loan Approved
>Loan_Status	N = Loan Rejected

# Technologies Used

• Python

• Pandas

• NumPy

• Matplotlib

• Seaborn

• Scikit-learn

# Project Workflow
# 1 Data Collection

The dataset is loaded using Pandas.

data = pd.read_csv("train.csv")
# 2 Data Preprocessing

Steps performed:

Handling missing values

Dropping null rows

Encoding categorical variables

Converting text values to numeric values

Example:

data.replace({'Loan_Status':{'N':0,'Y':1}}, inplace=True)
# 3 Data Visualization

Visualization helps understand patterns in the data.

Examples:

Education vs Loan Approval

Marital Status vs Loan Approval

Example visualization code:

sns.countplot(x='Education', hue='Loan_Status', data=data)

# Sample Visualizations
visualizations 
education_vs_loan.png 
marital_status_vs_loan.png
<img width="571" height="432" alt="image" src="https://github.com/user-attachments/assets/01e108f3-c034-48e1-81cb-898bc8336c23" />
<img width="571" height="432" alt="image" src="https://github.com/user-attachments/assets/ac68bbdc-ad84-469a-a199-44fa32d7a239" />



# Feature and Target Separation

Input features:

X = data.drop(columns=['Loan_ID','Loan_Status'], axis=1)

Target variable:

Y = data['Loan_Status']

# Train Test Split

The dataset is divided into training and testing sets.

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=0.1,
    stratify=Y,
    random_state=2
)
# Model Training

The model used in this project is Support Vector Machine (SVM).

classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

# Model Evaluation

Accuracy is used to evaluate model performance.

Dataset	Accuracy
Training Data	    ~0.79
Test Data	        ~0.83

# Project Structure
Loan-Prediction-ML
│
├── loan_prediction.ipynb
├── train.csv
├── README.md
│
└── visualizations
      education_vs_loan.png
      marital_status_vs_loan.png

# Key Insights

From the data analysis:

• Applicants with good credit history have higher loan approval chances.

• Graduates tend to receive approvals more frequently.

• Semi-urban areas show higher approval rates.

• Income and credit history are strong indicators for loan approval.

• These insights help understand the factors influencing bank decisions.

# Author

Navya Sri , 
B.Tech Artificial intelligence and data science Student

Interested in:

• Data Science

• Machine Learning

• Data Analysis

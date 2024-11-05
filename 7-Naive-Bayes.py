''' Implement linear regression using python 

=================================
Explanation:
=================================

===> To run this program you need to install the pandas Module

---> pandas Module is used to read csv files

===> To install, Open Command propmt and then execute the following command

---> pip install pandas


And, then you need to install the sklearn Module

===> Open Command propmt and then execute the following command to install sklearn Module

---> pip install scikit-learn

Finally, you need to create dataset called "Statements_data.csv" file.

===============================
Source Code :
===============================

'''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

msglbl_data = pd.read_csv('Statements_data.csv', names=['Message', 'Label'])
print("The Total instances in the Dataset: ", msglbl_data.shape[0])
msglbl_data['labelnum'] = msglbl_data.Label.map({'pos': 1, 'neg': 0})
# place the data in X and Y Vectors
X = msglbl_data["Message"]
Y = msglbl_data.labelnum
# to split the data into train se and test set
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y)

count_vect = CountVectorizer()
Xtrain_dims = count_vect.fit_transform(Xtrain)
Xtest_dims = count_vect.transform(Xtest)
df = pd.DataFrame(Xtrain_dims.toarray(),columns=count_vect.get_feature_names_out())
clf = MultinomialNB()
# to fit the train data into model
clf.fit(Xtrain_dims, Ytrain)
# to predict the test data
prediction = clf.predict(Xtest_dims)
print('******** Accuracy Metrics *********')
print('Accuracy : ', accuracy_score(Ytest, prediction)) 
print('Recall : ', recall_score(Ytest, prediction)) 
print('Precision : ',precision_score(Ytest, prediction))
print('Confusion Matrix : \n', confusion_matrix(Ytest, prediction))
print(10*"-")
# to predict the input statement
test_stmt = [input("Enter any statement to predict :")]
test_dims = count_vect.transform(test_stmt)
pred = clf.predict(test_dims)
for stmt,lbl in zip(test_stmt,pred):
	if lbl == 1:
		print("Statement is Positive")
	else:
		print("Statement is Negative")


Statements_data.csv
This is very good place,pos
I like this biryani,pos
I feel very happy,pos
This is my best work,pos
I do not like this restaurant,neg
I am tired of this stuff,neg
I can't deal with this,neg
What an idea it is,pos
My place is horrible,neg
This is an awesome place,pos
I do not like the taste of this juice,neg
I love to sing,pos
I am sick and tired,neg
I love to dance,pos
What a great holiday,pos
That is a bad locality to stay,neg
We will have good fun tomorrow,pos
I hate this food,neg

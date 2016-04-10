import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot #for plotting.
from sklearn import datasets,neighbors,linear_model,cross_validation,metrics

# Input data files are available in the "../input/" directory
digits_train = pd.DataFrame.from_csv("train.csv")
digits_test = pd.DataFrame.from_csv("test.csv")
#print digits_train.head()

#reset the index so as to slice out the labels for each digit
new_dataset = digits_train.reset_index()
test_reset = digits_test.reset_index()
train_target = new_dataset['label']#.ix[:999]
train_data = new_dataset.ix[:,1:]

n_samples = len(train_data) 

#X_train,X_test,y_train,y_test = cross_validation.train_test_split(train_data, train_target, test_size=0.4, random_state=0)
model  = linear_model.LogisticRegression()

try:
	model.fit(train_data,train_target)
	#scores = cross_validation.cross_val_score(model,train_data,train_target,cv=5)
	#expected = train_target[n_samples/2:]
	
	try:
		predicted = model.predict(test_reset)
	except Exception as e:
		print "Exception during prediction ",str(e)
except Exception as e:
	print str(e)

'''
#print "score: ",model.score(X_test,y_test)

print("Classification report for classifier %s:\n%s\n"
      % (model, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
'''

ImageId = test_reset.index +1
submission = pd.DataFrame({r'ImageId': ImageId ,
						   r'Label': predicted})

submission.to_csv("kaggle.csv", index = False)




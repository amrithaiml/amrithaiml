- 👋 Hi, I’m @amrithaiml
- 👀 I’m interested in ...
- 🌱 I’m currently learning ...
- 💞️ I’m looking to collaborate on ...
- 📫 How to reach me ...

<!---
amrithaiml/amrithaiml is a ✨ special ✨ repository because its `README.md` (this file) appears on your GitHub profile.
You can click the Preview link to take a look at your changes.
--->
!pip install h2o
import h2o as h2
from h2o.estimators import H2ORandomForestEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
import matplotlib.pyplot as plt

h2.init()
h1 = h2.create_frame(rows=10000,cols=3,seed=123)

h1.describe
mfolds = 2
mseed = 3
train, test = h1.split_frame(ratios = [0.7])

x = ['C1','C2']
C3 = ['C1' + 'C2']
print(C3)
y = 'C3'
xlist = []
test_ylist = []
train_ylist = []

#Buid RandomForest
index = 0
while index <= 1000:
  rb = H2ORandomForestEstimator(model_id="myrf",ntrees=index)
  rb.train(x=x, y=y, training_frame= train)
  xlist.append(index)
  temp_test = rb.model_performance(test)
  temp_train = rb.model_performance(train)
  test_ylist.append(temp_test['MSE'])
  train_ylist.append(temp_train['MSE'])
  rb.shap_explain_row_plot
  index += 99

plt.plot(xlist,test_ylist,label = "test")
plt.plot(xlist,train_ylist, label = "train")
plt.title('Test - Train V/S Tree depth and MSE')
plt.xlabel('Tree Depth')
plt.ylabel('MSE')
plt.show()

# Trying with Gradient Boosting Algorith with default parameters.
gbm = H2OGradientBoostingEstimator()
gbm.train(x=x, y=y,  training_frame=train)
perf = gbm.model_performance(test)
print ("GBM Same Data Set  MSE : ", perf['MSE'])

def classify_horizontal (x , y ) :
    if y > 2:
       return 1
    else:
        return 0
def classify_vertical (x , y ) :
    if x > 3:
       return 1
    else:
        return 0
    
def classify_diagonal (x , y ) :
    if y > x:
       return 1
    else:
        return 0
    
def classify_diagonal2 (x , y ) :
    if y > (-x +5):
       return 1
    else:
        return 0
    
def classify_circle (x , y ) :
    if y**2 + x**2 > 4:
       return 1
    else:
        return 0
    
def classify_any_line (w1, w2, x , y, b ) :
    if w1*x + w2*y +b> 0:
       return 1
    else:
        return 0
 # Hints : # 1. Review Python ’s if - else statements to implement simplem,→ conditional logic .
 # 2. You can visualize the decision boundary using matplotlib to,→ understand how the classification works .

 # Example for Problem 2
import numpy as np
from sklearn.linear_model import LogisticRegression

def find_optimal_boundary (points, labels ) :
    model = LogisticRegression ()
    model.fit( points , labels )
    w = model.coef_ [0]
    b = model . intercept_ [0]
    return w , b

# Hints :
# 1. Use sklearn ’s Lo gi st ic Re gr es si on model to fit your training data .

# 2. The fit method will learn the optimal parameters for the decision, boundary .
# 3. coef_ and intercept_ attributes of the trained model will give, you the weights and bias respectively .

 # Example usage
points = np . array ([[3.9 , 525] , [3.1 , 495] , [3.7 , 520] , [1.7 , 527] , [4.0 , 510] , [1.2 , 430]])
labels = np . array ([1 , -1 , 1 , -1 , 1 , -1])

w , b = find_optimal_boundary ( points , labels )

print(f" Optimal weights: { w } , Bias : { b } ")

new_point = np.array([[3.4,510]])
model = LogisticRegression()
model.fit(points, labels)
prediction = model.predict(new_point)

print(f"Prediction for GPA 3.4 and MCAT 510: {1 if prediction == 1 else -1}")
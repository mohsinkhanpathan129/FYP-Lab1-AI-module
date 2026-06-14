import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

print("Student Final Marks Predictor")
print("Lab 1 - FYP Module")
print("------------------------------")

# data manually
data = {
    'Attendance': [85, 90, 75, 60, 95, 70, 80, 65, 88, 72, 78, 92],
    'Assignment': [80, 88, 70, 55, 92, 68, 75, 60, 85, 70, 73, 89],
    'Quiz': [78, 85, 72, 50, 90, 65, 78, 58, 82, 68, 70, 87],
    'Final': [82, 87, 71, 54, 91, 66, 76, 59, 84, 69, 72, 88]
}

df = pd.DataFrame(data)
print("Data loaded, total rows:", len(df))

# x and y sepprete 
x = df[['Attendance', 'Assignment', 'Quiz']]
y = df['Final']


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# model
model = LinearRegression()
model.fit(x_train, y_train)

# prediction 
y_pred = model.predict(x_test)
acc = r2_score(y_test, y_pred)

print("Model trained")
print("Accuracy:", round(acc, 2))
print("------------------------------")

#user input
att = float(input("Attendance %: "))
assign = float(input("Assignment marks : "))
quiz = float(input("Quiz marks : "))

pred = model.predict([[att, assign, quiz]])[0]


if pred > 100:
    pred = 100
if pred < 0:
    pred = 0

print("\nPredicted Final Marks:", round(pred, 2))

# grade nikalna
if pred >= 80:
    grade = "A"
elif pred >= 70:
    grade = "B"
elif pred >= 60:
    grade = "C"
else:
    grade = "D"

print("Grade:", grade)
print("------------------------------")

input("Exir")
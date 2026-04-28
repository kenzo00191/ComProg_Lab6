import numpy as np
import pandas as pd

STUDENT_NAME = "Yzaac Fernandez"
STUDENT_ID = 930

print(f"--- {STUDENT_NAME}'s Lab Report (ID: {STUDENT_ID}) ---\n")

np.random.seed(STUDENT_ID)
data = np.random.randint(1, STUDENT_ID % 100 + 50, size=10)
print(f"1. Array: {data}")
print(f"   Mean: {np.mean(data):.2f}, Std Dev: {np.std(data):.2f}\n")

matrix = np.random.randint(1, STUDENT_ID % 50 + 20, size=(3, 4))
print(f"2. Matrix:\n{matrix}")
print(f"   Row sums: {np.sum(matrix, axis=1)}, Column sums: {np.sum(matrix, axis=0)}\n")

arr3d = np.random.randint(0, STUDENT_ID % 20 + 10, size=(3, 3, 3))
slice_mid = arr3d[:, 1, :]
print(f"3. 3D Slice [:, 1, :]:\n{slice_mid}\n")

x = np.random.randint(1, STUDENT_ID % 100 + 50, size=10)
filtered = x[(x > STUDENT_ID % 50) & (x < STUDENT_ID % 100)]
print(f"4. Original: {x}")
print(f"   Filtered: {filtered}\n")

A = np.random.randint(1, STUDENT_ID % 50 + 20, size=(2, 2))
B = np.random.randint(1, STUDENT_ID % 50 + 20, size=(2, 2))
product = A @ B
print(f"5. Matrix A @ B:\n{product}")
print(f"   Det(A): {np.linalg.det(A):.2f}\n")

angles = np.linspace(0, 2 * np.pi, 8)
print(f"6. Angles: {angles}")
print(f"   Sine values: {np.sin(angles)}\n")

arr_4x4 = np.random.randint(1, STUDENT_ID % 50 + 20, size=(4, 4))
arr_4x4[::2, ::2] = 0
print(f"7. Modified 4x4 Array:\n{arr_4x4}\n")

scores = np.random.randint(0, STUDENT_ID % 100 + 50, 10)
grades = np.where(scores >= STUDENT_ID % 70, 'Pass', 'Fail')
print(f"8. Scores: {scores}\n   Grades: {grades}\n")

arr_3x4 = np.random.randint(1, STUDENT_ID % 50 + 20, size=(3, 4))
print(f"9. Flattened: {arr_3x4.flatten()}")
print(f"   Transposed:\n{arr_3x4.T}\n")

arr_3x3 = np.random.randint(1, STUDENT_ID % 50 + 20, (3, 3))
arr_3x3[arr_3x3 % 2 == 0] = -1
print(f"10. Even replaced with -1:\n{arr_3x3}\n")

df_devs = pd.DataFrame({
    'Name': [STUDENT_NAME] * 5,
    'Score': np.random.randint(STUDENT_ID % 50, STUDENT_ID % 100 + 50, 5),
    'YearsCodePro': np.random.randint(0, STUDENT_ID % 20 + 10, 5)
})
high_exp = df_devs[df_devs['YearsCodePro'] > STUDENT_ID % 10]
print(f"11/14. High Experience Devs:\n{high_exp}\n")

df_edu = pd.DataFrame({
    'EdLevel': ['Bachelor', 'Master', 'PhD', 'Bachelor', 'Master'],
    'ConvertedComp': np.random.randint(40000, 150000, 5)
})
summary = df_edu.groupby('EdLevel')['ConvertedComp'].agg(['mean', 'median', 'std'])
print(f"12/18. Education Stats:\n{summary}\n")

df_geo = pd.DataFrame({
    'Country': ['Philippines', 'USA', 'UK', 'Canada', 'Germany'],
    'ConvertedComp': np.random.randint(40000, 150000, 5)
})
print(f"13. Top Salaries:\n{df_geo.sort_values(by='ConvertedComp', ascending=False)}\n")
print(f"20. Country Aggregation:\n{df_geo.groupby('Country')['ConvertedComp'].agg(['mean', 'max'])}\n")

bins = [0, 50000, 100000, 150000, 200000]
labels = ['0-50k', '50-100k', '100-150k', '>150k']
df_geo['Bracket'] = pd.cut(df_geo['ConvertedComp'], bins=bins, labels=labels)
print(f"15. Salary Brackets:\n{df_geo[['ConvertedComp', 'Bracket']]}\n")

df_geo['LogComp'] = np.log(df_geo['ConvertedComp'])
print(f"16. Log Transformed Data:\n{df_geo[['ConvertedComp', 'LogComp']].head()}\n")

df_geo['YearsExp'] = np.random.randint(1, 20, 5)
print(f"17. Correlation Matrix:\n{df_geo[['ConvertedComp', 'YearsExp']].corr()}\n")

threshold = 110000
df_geo['HighPay'] = np.where(df_geo['ConvertedComp'] > threshold, 'Yes', 'No')
print(f"19. High Pay Threshold (>{threshold}):\n{df_geo[['ConvertedComp', 'HighPay']]}")
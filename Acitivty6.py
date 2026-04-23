import numpy as np

years_exp = np.array([1, 3, 5, 7, 10, 12, 15])
print("Years of Experience:", years_exp)

salaries = np.array([
    [50, 60, 70, 80, 90], 
    [80, 90, 100, 110, 120]
])
print("Salary Matrix:\n", salaries)

zeros_array = np.zeros((2, 2))  # 2x2 zeros
ones_array = np.ones((2, 3))   # 2x3 ones
print("Zeros Array:\n", zeros_array)
print("Ones Array:\n", ones_array)

identity_matrix = np.eye(3)
print("3x3 Identity Matrix:\n", identity_matrix)

exp_plus_5 = years_exp + 5
print("Years + 5:", exp_plus_5)

exp_times_2 = years_exp * 2
print("Years * 2:", exp_times_2)

sample1 = np.array([1, 2, 3])
sample2 = np.array([4, 5, 6])
dot_result = np.dot(sample1, sample2)
print("Dot Product:", dot_result)

exp_minus_1 = years_exp - 1
exp_div_2 = years_exp / 2
print("Years - 1:", exp_minus_1)
print("Years / 2:", exp_div_2)

exp_values = np.exp(years_exp)  # Calculates e^x for each element
log_values = np.log(years_exp)  # Calculates natural log (ln) for each element
print("Exponential values:\n", exp_values)
print("Logarithmic values:\n", log_values)

print("First year of experience:", years_exp[0])

print("First two salaries:", salaries[0, :2])

print("Second column salaries:", salaries[:, 1])

print("Last year of experience:", years_exp[-1])

reversed_years = years_exp[::-1]
print("Years of experience reversed:", reversed_years)

salary_subgroup = salaries[0:2, 1:3]
print("Subgroup of salaries (2x2 middle section):\n", salary_subgroup)

reshaped_exp = np.reshape(np.arange(1, 7), (2, 3))
print("Reshaped Experience Array:\n", reshaped_exp)

flattened_exp = reshaped_exp.flatten()
print("Flattened Array:", flattened_exp)

print("Transposed Array:\n", reshaped_exp.T)

reshaped_3x2 = np.reshape(np.arange(1, 7), (3, 2))
print("Reshaped into 3x2 Matrix:\n", reshaped_3x2)

bonus = np.array([5, 10, 15, 20, 25]) 

salaries_with_bonus = salaries + bonus
print("Salaries after bonus:\n", salaries_with_bonus)

scaling_factor = 1.10
scaled_salaries = salaries * scaling_factor
print("Scaled Salaries (10% raise):\n", scaled_salaries)

print("Mean experience:", np.mean(years_exp))

print("Std deviation of experience:", np.std(years_exp))

print("Max salary:", np.max(salaries), "Min salary:", np.min(salaries))

print("Sum of all salaries:", np.sum(salaries))

print("Median salary:", np.median(salaries))
print("75th percentile of experience:", np.percentile(years_exp, 75))

angles = np.array([0, np.pi/4, np.pi/2])
print("Sine of angles:", np.sin(angles))
print("Cosine of angles:", np.cos(angles))

salary_sums = np.apply_along_axis(np.sum, 1, salaries)
print("Sum of Salaries per person:", salary_sums)

print("Square root of experience:", np.sqrt(years_exp))

def calculate_tax(row):
    return np.sum(row) * 0.05

tax_per_person = np.apply_along_axis(calculate_tax, 1, salaries)
print("5% Tax deduction per person:", tax_per_person)

import pandas as pd

data = np.random.randint(1, 50, size=(5, 3))

df_data = pd.DataFrame(data, columns=['X', 'Y', 'Z'])
print("Generated Data:\n", df_data)

df_data['Log_X'] = np.log(df_data['X'])
df_data['Sqrt_Y'] = np.sqrt(df_data['Y'])
print("DataFrame with NumPy Transformations:\n", df_data)

print("Correlation Matrix:\n", df_data.corr())

print("Mean of each column:\n", df_data.mean())
print("Median of each column:\n", df_data.median())

df_data['Square_Z'] = np.square(df_data['Z'])
print("DataFrame with Squared Z:\n", df_data)

df_data.to_csv('sample_data.csv', index=False)
print("Data saved to 'sample_data.csv'.")

df_imported = pd.read_csv('sample_data.csv')
print("Imported DataFrame:\n", df_imported)

summary_stats = df_imported.describe()
print("Summary Statistics:\n", summary_stats)

print("Column Means:\n", df_imported.mean())
print("Column Standard Deviations:\n", df_imported.std())

df_imported['Sum_XY'] = df_imported['X'] + df_imported['Y']
df_imported.to_csv('modified_data.csv', index=False)
print("Modified DataFrame with Sum_XY saved to 'modified_data.csv'.")
print(df_imported.head())

df_clean = df_imported.copy()

df_clean['ExperienceLevel'] = np.where(df_clean['X'] >= 25, 'Senior', 'Junior')

grouped_data = df_clean.groupby(['ExperienceLevel'])['Z'].mean()
print("Grouped Average Value of Z:\n", grouped_data)

formatted_data = grouped_data.reset_index()
print("Formatted Grouped Data:\n", formatted_data)

top_values = df_clean.sort_values(by='Z', ascending=False).head(3)
print("Top 3 highest Z values:\n", top_values)
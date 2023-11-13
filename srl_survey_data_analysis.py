import pandas as pd
import numpy as np
#from scipy import stats
from scipy.stats import wilcoxon
import seaborn as sns
import matplotlib.pyplot as plt
from fpdf import FPDF


# Load the data
pre_data_wide = pd.read_excel('pre_survey.xlsx')
#print(pre_data_wide.dtypes)
#print(pre_data_wide.info())

post_data_wide = pd.read_excel('post_survey.xlsx')
#print(post_data_wide.info())
#print(post_data_wide.dtypes)

# print(list(pre_data_wide))
# print(pre_data_wide.shape)
# print(pre_data_wide.head())



# # Converting the data from wide format to long format

pre_data_long = pd.melt(pre_data_wide, id_vars=['ID'], var_name='Question', value_name='Response')
#print(pre_data_long.info())
post_data_long = pd.melt(post_data_wide, id_vars=['ID'], var_name='Question', value_name='Response')
#print(post_data_long.info())

# #print(pre_data_long.head())
# #print(post_data_long.head())


# Descriptive Statistics
# Mean, Median for each question in both surveys
pre_descriptive = pre_data_long.groupby('Question')['Response'].agg(['mean', 'median'])
#print(pre_descriptive)
post_descriptive = post_data_long.groupby('Question')['Response'].agg(['mean', 'median'])
#print(post_descriptive)

# #Reliability Analysis
# #Compute Cronbach's alpha for internal consistency
# def cronbach_alpha(df):
#     df_corr = df.corr()
#     N = df.shape[1]  # Number of items in the scale
#     rs = np.array([])
#     for i, col in enumerate(df_corr.columns):
#         sum_ = df_corr[col][i+1:].values
#         rs = np.append(rs, sum_)
#     mean_corr = np.mean(rs)
#     cronbach_alpha = (N * mean_corr) / (1 + (N - 1) * mean_corr)
#     return cronbach_alpha

# alpha_pre = cronbach_alpha(pre_data_wide.iloc[:, 1:])  # Excluding the participant ID column
# alpha_post = cronbach_alpha(post_data_wide.iloc[:, 1:])

# print(f'Cronbach alpha for pre survey: {alpha_pre}')
# print(f'Cronbach alpha for post survey: {alpha_post}')

# # Comparative Analysis using Wilcoxon Signed-Rank Test for each question
# # This assumes the same participants answered both surveys and are in the same order
# p_values = []
# for question in range(1, 41):  # Assuming questions are labeled as Q1, Q2, ..., Q40
#     stat, p = stats.wilcoxon(pre_data_wide[f'Q{question}'], post_data_wide[f'Q{question}'])
#     p_values.append(p)

# # Correction for multiple testing, e.g., Bonferroni
# adjusted_p_values = [p * 40 for p in p_values]  # Adjusting for 40 tests

# Assuming pre_data and post_data are your DataFrame with 30 participants and 42 questions

# Step 1: Wilcoxon Signed-Rank Test
p_values = []
effect_sizes = []
questions = pre_data_wide.columns[1:]  # Assuming same questions in pre and post

for question in questions:  # Exclude participant ID column
#     stat, p_value = wilcoxon(pre_data_wide[question], post_data_wide[question])
#     wilcoxon_results.append((question, stat, p_value))
#for question in questions:
    stat, p = wilcoxon(pre_data_wide[question], post_data_wide[question])
    p_values.append(p)
    # Step 2: Calculate Effect Size
    effect_size = stat / len(pre_data_wide[question])**0.5
    effect_sizes.append(effect_size)


# 95% Confidence Interval
def bootstrap_ci(data1, data2, n_bootstrap=1000, ci=95):
    bootstrap_diffs = []
    for _ in range(n_bootstrap):
        sample1 = np.random.choice(data1, size=len(data1), replace=True)
        sample2 = np.random.choice(data2, size=len(data2), replace=True)
        bootstrap_diffs.append(np.median(sample2) - np.median(sample1))  # or mean, if more appropriate
    
    lower_bound = np.percentile(bootstrap_diffs, (100-ci)/2)
    upper_bound = np.percentile(bootstrap_diffs, 100-(100-ci)/2)
    return lower_bound, upper_bound

# Example usage with your data
# Assuming pre_data and post_data are your pre and post survey dataframes
conf_intervals = {}
for question in pre_data_wide.columns[1:]:
    lb, ub = bootstrap_ci(pre_data_wide[question], post_data_wide[question])
    conf_intervals[question] = (lb, ub)

print(conf_intervals)

# Step 3: Reporting Results
report_df = pd.DataFrame({
    'Question': questions,
    'P-Value': p_values,
    'Effect Size': effect_sizes
    #'95% CI': conf_intervals
})

#print(report_df)

# Step 4: Save Results to PDF
pdf = FPDF()
pdf.add_page()
pdf.set_font('Arial', 'B', 16)
pdf.cell(0, 10, 'Survey Analysis Report', 0, 1, 'C')

# Column Headers
column_headers = ['Question', 'P-Value', 'Effect Size', '95% CI']
pdf.set_font('Arial', 'B', 12)
pdf.cell(40, 10, column_headers[0], 1)
pdf.cell(40, 10, column_headers[1], 1)
pdf.cell(40, 10, column_headers[2], 1)
#pdf.cell(40, 10, column_headers[3], 1)
pdf.ln()

for i in range(len(report_df)):
    pdf.cell(40, 10, report_df['Question'][i], 1)
    pdf.cell(40, 10, str(round(report_df['P-Value'][i], 5)), 1)  # Rounded to 5 decimal places
    pdf.cell(40, 10, str(round(report_df['Effect Size'][i], 5)), 1)  # Rounded to 5 decimal places
    #pdf.cell(40, 10, str(round(report_df['95% CI'][i], 5)), 1)  # Rounded to 5 decimal places
    pdf.ln()

# Table Content
# pdf.set_font('Arial', '', 12)
# for i in range(len(report_df)):
#     pdf.cell(40, 10, report_df['Question'][i], 1)
#     pdf.cell(40, 10, str(report_df['P-Value'][i]), 1)
#     pdf.cell(40, 10, str(report_df['Effect Size'][i]), 1)
#     pdf.ln()

# Save PDF
pdf.output('survey_analysis_report.pdf')


# # Creating a report DataFrame
# report_df = pd.DataFrame({
#     'Question': questions,
#     'P-Value': p_values,
#     'Effect Size': effect_sizes
# })

# # Step 3: Reporting Results
# print(report_df)

# # Step 4: Save Results to PDF
# pdf = FPDF()
# pdf.add_page()
# pdf.set_font('Arial', 'B', 12)

# pdf.cell(0, 10, 'Survey Analysis Report', 0, 1, 'C')

# for i in range(len(report_df)):
#     line = report_df.iloc[i].to_string()
#     pdf.cell(0, 10, line, 0, 1)

# # Save the PDF with your results
# pdf.output('survey_analysis_report.pdf')

# Note: This is a basic PDF report. You can use more advanced libraries like ReportLab for more complex formatting.



# # Initialize a list to store the test results
# wilcoxon_results = []

# # Iterate through each question to perform the Wilcoxon signed-rank test
# for question in pre_data_wide.columns[1:]:  # Exclude participant ID column
#     stat, p_value = wilcoxon(pre_data_wide[question], post_data_wide[question])
#     wilcoxon_results.append((question, stat, p_value))

# # Convert the results to a DataFrame
# wilcoxon_df = pd.DataFrame(wilcoxon_results, columns=['Question', 'Statistic', 'P-Value'])

# # Apply a correction for multiple comparisons if needed
# # For example, Bonferroni correction
# alpha = 0.05
# bonferroni_correction = alpha / len(pre_data_wide.columns[1:])
# wilcoxon_df['Significant After Correction'] = wilcoxon_df['P-Value'] < bonferroni_correction

# # Display the results
# #print(wilcoxon_df)

## Combined visualization

# Melt the dataframes to long format for easier plotting with seaborn
# pre_data_long = pre_data.melt(id_vars=['ParticipantID'], var_name='Question', value_name='Response')
pre_data_long['Survey'] = 'Pre'

#post_data_long = post_data.melt(id_vars=['ParticipantID'], var_name='Question', value_name='Response')
post_data_long['Survey'] = 'Post'

# Combine the data
combined_data = pd.concat([pre_data_long, post_data_long])

# Get a list of unique questions
questions = combined_data['Question'].unique()

# Divide the questions into chunks of 5
chunk_size = 5
question_chunks = [questions[i:i + chunk_size] for i in range(0, len(questions), chunk_size)]

# Plot each chunk of questions
for chunk in question_chunks:
    subset_data = combined_data[combined_data['Question'].isin(chunk)]
    plt.figure(figsize=(15, 6))  # Adjust the size according to your needs
    sns.violinplot(x='Question', y='Response', hue='Survey', data=subset_data, split=True)

    # plt.figure(figsize=(20, 6))  # Adjust the size according to your needs
    # sns.violinplot(x='Question', y='Response', hue='Survey', data=subset_data, split=True, inner=None)  # inner=None removes the inner bars inside the violins
    #sns.swarmplot(x='Question', y='Response', hue='Survey', data=subset_data, color='k', alpha=0.7)  # alpha sets the transparency

    # plt.xticks(rotation=90)  # Rotates the question labels to avoid overlap
    # plt.title('Violin and Swarm Plot Distribution for Pre and Post Survey Data')
    # plt.ylabel('Response')
    # plt.xlabel('Question')
    # plt.legend(loc='upper right')  # Adjust the legend location as needed
    
    plt.xticks(rotation=45)  # Rotate the question labels to avoid overlap
    plt.title('Boxplot Distribution for Questions')
    plt.ylabel('Response')
    plt.xlabel('Question')
    plt.show()

# # Plotting
# plt.figure(figsize=(20, 10))  # Adjust the size according to your needs
# sns.boxplot(x='Question', y='Response', hue='Survey', data=combined_data)
# plt.xticks(rotation=90)  # Rotates the question labels to avoid overlap
# plt.title('Boxplot Distribution for Pre and Post Survey Data')
# plt.ylabel('Response')
# plt.xlabel('Question')

# Display the plot
#plt.show()


# Visualization
# Boxplot for a question's distribution in pre and post survey
# sns.boxplot(data=[pre_data_wide['Q1'], post_data_wide['Q1']])
# plt.xlabel('Survey')
# plt.ylabel('Response')
# plt.title('Boxplot of Responses for Question 1')
# plt.xticks([0, 1], ['Pre', 'Post'])
# plt.show()

# # Interpretation and Reporting
# # You will need to interpret the p-values and effect sizes in the context of your research
# # For example:
# for i, p in enumerate(adjusted_p_values):
#     if p < 0.05:
#         print(f'Question {i+1} shows a statistically significant difference.')
#     else:
#         print(f'Question {i+1} does not show a statistically significant difference.')

# # Make sure to interpret the results with caution and in context.
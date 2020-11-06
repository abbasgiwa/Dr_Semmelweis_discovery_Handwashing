#importing modules
import pandas as pd

# Read datasets/yearly_deaths_by_clinic.csv into yearly
yearly = pd.read_csv("yearly_deaths_by_clinic.csv")

# Print out yearly
print(yearly)

# Calculate proportion of deaths per no. births
yearly['proportion_deaths']= yearly['deaths']/yearly['births']

# Extract clinic 1 data into yearly1 and clinic 2 data into yearly2
yearly1 = yearly[yearly['clinic']=='clinic 1']
yearly2 =yearly[yearly['clinic']=='clinic 2']

# Print out yearly1
print(yearly1)

# This makes plots appear in the notebook
# %matplotlib inline
import matplotlib.pyplot as plt

# Plot yearly proportion of deaths at the two clinics

ax= yearly1.plot("year","proportion_deaths", label='yeary1')
yearly2.plot("year","proportion_deaths", label='yeary2',ax=ax)

ax.set_ylabel('Proportion deaths')
plt.show()

# Read datasets/monthly_deaths.csv into monthly
monthly = pd.read_csv('monthly_deaths.csv', parse_dates=['date'])

# Calculate proportion of deaths per no. births.
monthly["proportion_deaths"]= monthly['deaths']/ monthly['births']
# Print out the first rows in monthly
monthly.head()

# Plot monthly proportion of deaths

ax=monthly.plot("date","proportion_deaths")

ax.set_ylabel('Proportion deaths')




plt.show()

# Date when handwashing was made mandatory
import pandas as pd
handwashing_start = pd.to_datetime('1847-06-01')

# Split monthly into before and after handwashing_start
before_washing = monthly[monthly['date']< handwashing_start]
after_washing = monthly[monthly['date']>= handwashing_start]

# Plot monthly proportion of deaths before and after handwashing

ax= before_washing.plot('date', 'proportion_deaths')
after_washing.plot('date', 'proportion_deaths', ax=ax)
ax.set_ylabel('Proportion deaths')


# Difference in mean monthly proportion of deaths due to handwashing
import numpy as np
before_proportion = before_washing['proportion_deaths']
after_proportion = after_washing['proportion_deaths']
mean_diff = np.mean(after_proportion) - np.mean(before_proportion)
mean_diff

# A bootstrap analysis of the reduction of deaths due to handwashing
boot_mean_diff = []
for i in range(3000):
    boot_before = before_proportion.sample(frac=1, replace=True)
    boot_after = after_proportion.sample(frac=1, replace=True)
    boot_mean_diff.append( boot_after.mean() - boot_before.mean() )

# Calculating a 95% confidence interval from boot_mean_diff
confidence_interval = pd.Series(boot_mean_diff).quantile([0.025, 0.975])
confidence_interval

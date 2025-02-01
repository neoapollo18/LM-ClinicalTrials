import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('src/TrialData/2012_to_2016.csv')

print('\n')
# Status
status_0 = (df['Status'] == 0).sum()
status_1 = (df['Status'] == 1).sum()

print(f"Status")
print(f"Terminated: {status_0}")
print(f"Completed: {status_1}")

#Phases
phase_0 = (df['Phase'] == 0).sum()
phase_1 = (df['Phase'] == 1).sum()
phase_2 = (df['Phase'] == 2).sum()
phase_3 = (df['Phase'] == 3).sum()
phase_4 = (df['Phase'] == 4).sum()

print('\n')
print(f'Phases')
print(f"Early Phase 1: {phase_0}")
print(f"Phase 1: {phase_1}")
print(f"Phase 2: {phase_2}")
print(f"Phase 3: {phase_3}")
print(f"Phase 4: {phase_4}")

#Gender
gender_0 = (df['GenderRestrictions'] == 0).sum()
gender_1 = (df['GenderRestrictions'] == 1).sum()

print('\n')
print(f'Gender Restrictions')
print(f'Gender Non-Restrictive: {gender_0}')
print(f'Gender Restrictive: {gender_1}')

#DesignMasking
df['DesignMasking'] = df['DesignMasking'].str.strip().str.lower()

# Count the occurrences of each value
design_none = pd.isna(df['DesignMasking']).sum()
design_single = (df['DesignMasking'] == "single").sum()
design_double = (df['DesignMasking'] == "double").sum()
design_triple = (df['DesignMasking'] == "triple").sum()
design_quadruple = (df['DesignMasking'] == "quadruple").sum()

print('\n')
print(f'DesignMasking')
print(f'None: {design_none}')
print(f'Single: {design_single}')
print(f'Double: {design_double}')
print(f'Triple: {design_triple}')
print(f'Quadruple: {design_quadruple}')











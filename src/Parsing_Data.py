import xml.etree.ElementTree as ET
import re
import os
import pandas as pd
import string
import random
from datetime import datetime
cleaned_trails = []

#MainPreprocessing


from nltk.tokenize import word_tokenize

def preprocess_summary(text):
    text = re.sub('<[^<]+?>', '', text)
    text = text.strip()

    # Tokenization
    tokens = word_tokenize(text)
    text = ' '.join(tokens)
    return text


cleaned_trials = []

def process_xml_file(filepath):
    
    try:
        #Parsing
        tree = ET.parse(filepath)
        root = tree.getroot()

        #Factors to Extract from Trial Data
        if root.find(".//Field[@Name='BriefSummary']") is not None:
            brief_summary = root.find(".//Field[@Name='BriefSummary']").text
        else:
            return False
        if root.find(".//Field[@Name='DetailedDescription']") is not None:
            detailed_description = root.find(".//Field[@Name='DetailedDescription']").text
        else:
            return False    
        if root.find(".//Field[@Name='NCTId']") is not None:
            trial_id = root.find(".//Field[@Name='NCTId']").text
        else:
            return False
        if root.find(".//Field[@Name='Phase']") is not None:
            if root.find(".//Field[@Name='Phase']").text != "Not Applicable":
                if root.find(".//Field[@Name='Phase']").text == 'Phase 1':
                    phase = 1
                elif root.find(".//Field[@Name='Phase']").text == 'Phase 2':
                    phase = 2
                elif root.find(".//Field[@Name='Phase']").text == 'Phase 3':
                    phase = 3
                elif root.find(".//Field[@Name='Phase']").text == 'Phase 4':
                    phase = 4
                else:
                    phase = 0
            else:
                return False
        else:
            return False
        if root.find(".//Field[@Name='OrgFullName']") is not None:
            org_full_name = root.find(".//Field[@Name='OrgFullName']").text
        else:
            return False
        if root.find(".//Field[@Name='BriefTitle']") is not None:
            brief_title = root.find(".//Field[@Name='BriefTitle']").text
        else:
            return False
       # Your existing code
        if root.find(".//Field[@Name='StudyFirstSubmitDate']") is not None:
            study_first_submit_date = root.find(".//Field[@Name='StudyFirstSubmitDate']").text
        else:
            return False
        try:
            study_first_submit_date_dt = datetime.strptime(study_first_submit_date, '%B %d, %Y')
        except ValueError as e:
            print(f"Unexpected error with date format: {e}")
            return False

        # cutoff_date = datetime(2002, 1, 1)
        start_date = datetime(2002, 1, 1)
        end_date = datetime(2011, 12, 31)
        # Check if the study was submitted after the cutoff date
        if start_date <= study_first_submit_date_dt <= end_date:
            return False
        if root.find(".//Field[@Name='LastUpdateSubmitDate']") is not None:
            last_update_submit_date = root.find(".//Field[@Name='LastUpdateSubmitDate']").text
        else:
            return False
        if root.find(".//Field[@Name='LeadSponsorName']") is not None:
            lead_sponsor_name = root.find(".//Field[@Name='LeadSponsorName']").text
        else:
            return False
        if root.find(".//Field[@Name='Condition']") is not None:
            condition = root.find(".//Field[@Name='Condition']").text
        else:
            return False
        if root.find(".//Field[@Name='DesignPrimaryPurpose']") is not None:
            design_primary_purpose = root.find(".//Field[@Name='DesignPrimaryPurpose']").text
        else:
            return False
        if root.find(".//Field[@Name='EnrollmentCount']") is not None:
            enrollment_count = root.find(".//Field[@Name='EnrollmentCount']").text
        else:
            return False
        if root.find(".//Field[@Name='InterventionType']") is not None:
            intervention_type = root.find(".//Field[@Name='InterventionType']").text
        else:
            return False
        if root.find(".//Field[@Name='InterventionName']") is not None:
            intervention_name = root.find(".//Field[@Name='InterventionName']").text
        else:
            return False
        if root.find(".//Field[@Name='Gender']") is not None:
            if root.find(".//Field[@Name='Gender']").text == "All":
                gender = 0
            else:
                gender = 1
        else:
            return False
        if root.find(".//Field[@Name='EligibilityCriteria']") is not None:
            eligibility_criteria = root.find(".//Field[@Name='EligibilityCriteria']").text
        else:
            return False
        if root.find(".//Field[@Name='MinimumAge']") is not None:
            minimum_age = root.find(".//Field[@Name='MinimumAge']").text
        else:
            return False
        if root.find(".//Field[@Name='MaximumAge']") is not None:
            maximum_age = root.find(".//Field[@Name='MaximumAge']").text
        else:
            return False
        if root.find(".//Field[@Name='OverallStatus']") is not None:
            if (root.find(".//Field[@Name='OverallStatus']").text == "Terminated"):
                status = 0
            elif (root.find(".//Field[@Name='OverallStatus']").text == "Completed"):
                rand_num = random.random()
                if rand_num <= 0.15:
                    status = 1
                else:
                    return False
            else:
                return False
            
        else:
            return False
        if root.find(".//Field[@Name='DesignMasking']") is not None:
            if root.find(".//Field[@Name='DesignMasking']").text == "None (Open Label)":
                design_masking = "None"
            else:
                design_masking = root.find(".//Field[@Name='DesignMasking']").text
        else:
            return False

        #Preprocessing Data
        preprocessed_brief_summary = preprocess_summary(brief_summary)
        preprocessed_eligibility_criteria = preprocess_summary(eligibility_criteria)
        preprocessed_detailed_description = preprocess_summary(detailed_description)
        #Array
        return [org_full_name, trial_id, brief_title, lead_sponsor_name, phase, status, condition,intervention_type,intervention_name,design_primary_purpose,design_masking, preprocessed_brief_summary, preprocessed_detailed_description, study_first_submit_date, last_update_submit_date, preprocessed_eligibility_criteria, enrollment_count, gender, minimum_age, maximum_age]

        

    #Exceptions    
    except ET.ParseError:
            print(f"Error parsing file: {filepath}")
            return False
    except PermissionError:
            print(f"Permission denied: {filepath}")
            return False
    except Exception as e:
            print(f"Unexpected error with file {filepath}: {e}")
            return False
    

dct = "/Users/charles/Documents/Python_Files/AI-ClinicalTrialsProject/AllAPIXML"

# dir = "/Users/charles/Documents/Python_Files/AI-ClinicalTrialsProject/AllAPIXML/NCT0000xxxx"


# for file in os.listdir(dir):
#     if file.endswith(".xml"):
#         filepath = os.path.join(dir, file)
#         preprocessed_summary = process_xml_file(filepath)
#         if preprocessed_summary == False:
#             continue
#         cleaned_trials.append(preprocessed_summary)


# Iter over files in dir
import os

for dirName, subdirList, fileList in os.walk(dct):
    print(f'Found directory: {dirName}')
    for file in fileList:
        if file.endswith(".xml"):
            filepath = os.path.join(dirName, file)
            preprocessed_summary = process_xml_file(filepath)
            if preprocessed_summary == False:
                continue
            cleaned_trials.append(preprocessed_summary)

    

df = pd.DataFrame(cleaned_trials, columns=['OrgFullName','TrialID', 'BriefTitle','LeadSponsorName','Phase', 'Status', 'Condition','InterventionType','InterventionName','DesignPrimaryPurpose','DesignMasking','BriefSummary', 'DetailedDescription','StudyFirstSubmitDate','LastUpdateSubmitDate','EligibilityCriteria', 'StudyEnrollmentCount','GenderRestrictions','MinAge','MaxAge'])
df.to_csv('src/TrialData/trial_data1.csv', index=False)

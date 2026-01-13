import os
import pandas as pd
import pydicom
from tqdm import tqdm

df = pd.read_csv('./dataset.csv')

def find_first_dicom(dce_root):
    for root, _, files in os.walk(dce_root):
        for f in files:
            if f.lower().endswith('.dcm'):
                return os.path.join(root, f)
    return None  


results = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing patients"):
    patient_path = row['patient_dir']
    dce_folder = None

    try:
        all_fold = []
        for folder_name in os.listdir(patient_path):
            if folder_name.startswith('DCE') and os.path.isdir(os.path.join(patient_path, folder_name)):
                dce_folder = os.path.join(patient_path, folder_name)
                break
            elif os.path.isdir(os.path.join(patient_path, folder_name)):
                all_fold.append(folder_name)
        if dce_folder is None:
            dce_folder = os.path.join(patient_path, all_fold[0])

        dicom_file = find_first_dicom(dce_folder)
        if dicom_file is None:
            print('No DICOM file found',patient_path,dce_folder)
            raise FileNotFoundError("No DICOM file found")

        # read DICOM info
        ds = pydicom.dcmread(dicom_file, stop_before_pixels=True)
        manufacturer = ds.get((0x0008, 0x0070), 'N/A')  # Manufacturer
        manufacturer = manufacturer.value if  manufacturer!='N/A' else 'N/A'

        model_name = ds.get((0x0008, 0x1090), 'N/A')    # Model
        model_name = model_name.value if  model_name!='N/A' else 'N/A'
        
        field_strength = ds.get((0x0018, 0x0087), 'N/A')  # Magnetic Field Strength
        field_strength = field_strength.value if  field_strength!='N/A' else 'N/A'

        results.append({
            # 'patient_dir': patient_path,
            'patient_id':row['Unique_ID'],
            'manufacturer': str(manufacturer),
            'model_name': str(model_name),
            'field_strength_T': field_strength if isinstance(field_strength, (int, float)) else 'N/A'
        })

    except Exception as e:
        print(e,patient_path,dce_folder,dicom_file)
        raise e
        results.append({
            'patient_dir': patient_path,
            'manufacturer': 'Error',
            'model_name': 'Error',
            'field_strength_T': 'Error'
        })

# save as CSV
result_df = pd.DataFrame(results)
result_df.to_csv('./mri_machine_info.csv', index=False)
print("finished: ./mri_machine_info.csv")

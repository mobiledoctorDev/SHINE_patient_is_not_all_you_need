
nohup echo "start" && \
python benchmarks.py --bucket_name shine_v3_11_kt --split manual --using_features patient2 --fill_null > logs/v22_patient_fillnull_20230828.log && \
python benchmarks.py --bucket_name shine_v3_11_kt --split manual --using_features patient2+gps --fill_null > logs/v22_patient_gps_fillnull_20230828.log && \
python benchmarks.py --bucket_name shine_v3_11_kt --split manual --using_features patient2+si10 --fill_null > logs/v22_patient_si10_fillnull_20230828.log && \
python benchmarks.py --bucket_name shine_v3_11_kt --split manual --using_features patient2+owid --fill_null > logs/v22_patient_owid_fillnull_20230828.log && \
python benchmarks.py --bucket_name shine_v3_11_kt --split manual --using_features patient2+gps+si10+owid --fill_null > logs/v22_patient_gps_si10_owid_fillnull_20230828.log && \
echo "Finished" &

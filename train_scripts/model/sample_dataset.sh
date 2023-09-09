#python automl_csv.py --bucket_name shine_v3_7_kt --using_features patient+gps+bts+si10+hw7+owid
#python automl_dataset.py --bucket_name shine_v3_7_kt --using_features patient+gps+bts+si10+hw7+owid
#
#python automl_csv.py --bucket_name shine_v3_7_kt --using_features patient+gps+bts+si10+hw7+owid --fill_null
#python automl_dataset.py --bucket_name shine_v3_7_kt --using_features patient+gps+bts+si10+hw7+owid --fill_null
#
#python automl_csv.py --bucket_name shine_v3_7_kt --using_features patient+screason+gps+bts+si10+hw7+owid --onlyuse_selfcheck_first
#python automl_dataset.py --bucket_name shine_v3_7_kt --using_features patient+screason+gps+bts+si10+hw7+owid --onlyuse_selfcheck_first
#
#python automl_csv.py --bucket_name shine_v3_7_kt --using_features patient+screason+gps+bts+si10+hw7+owid --onlyuse_selfcheck_first --fill_null
#python automl_dataset.py --bucket_name shine_v3_7_kt --using_features patient+screason+gps+bts+si10+hw7+owid --onlyuse_selfcheck_first --fill_null

nohup echo "start" && \
python automl_csv.py --bucket_name shine_v3_7_kt --using_features patient && \
python automl_dataset.py --bucket_name shine_v3_7_kt --using_features patient && \
echo "finished" &

nohup echo "start" && \
python automl_csv.py --bucket_name shine_v3_7_kt --using_features patient --fill_null && \
python automl_dataset.py --bucket_name shine_v3_7_kt --using_features patient --fill_null && \
echo "finished" &

nohup echo "start" && \
python automl_csv.py --bucket_name shine_v3_7_kt --using_features patient --onlyuse_selfcheck_first && \
python automl_dataset.py --bucket_name shine_v3_7_kt --using_features patient --onlyuse_selfcheck_first && \
echo "finished" &

nohup echo "start" && \
python automl_csv.py --bucket_name shine_v3_7_kt --using_features patient --onlyuse_selfcheck_first --fill_null && \
python automl_dataset.py --bucket_name shine_v3_7_kt --using_features patient --onlyuse_selfcheck_first --fill_null && \
echo "finished" &


#nohup echo "start" && \
#python automl_run.py --bucket_name shine_v3_7_kt --using_features patient 1>> logs/automl_p.log 2>> logs/automl_p.log && \
#echo "finished" &

#nohup echo "start" && \
#python automl_run.py --bucket_name shine_v3_7_kt --using_features patient+gps 1>> logs/automl_pg.log 2>> logs/automl_pg.log && \
#echo "finished" &

#nohup echo "start" && \
#python automl_run.py --bucket_name shine_v3_7_kt --using_features patient+gps+si10 1>> logs/automl_pgs.log 2>> logs/automl_pgs.log && \
#echo "finished" &

#nohup echo "start" && \
#python automl_run.py --bucket_name shine_v3_7_kt --using_features patient+si10 1>> logs/automl_ps.log 2>> logs/automl_ps.log && \
#echo "finished" &

#nohup echo "start" && \
#python automl_run.py --bucket_name shine_v3_7_kt --using_features patient+hw7 1>> logs/automl_ph.log 2>> logs/automl_ph.log && \
#echo "finished" &
#
#nohup echo "start" && \
#python automl_run.py --bucket_name shine_v3_7_kt --using_features patient+owid 1>> logs/automl_po.log 2>> logs/automl_po.log && \
#echo "finished" &

#nohup echo "start" && \
#python automl_run.py --bucket_name shine_v3_7_kt --using_features patient+gps+si10+hw7+owid 1>> logs/automl_pgsho.log 2>> logs/automl_pgsho.log && \
#echo "finished" &

#nohup echo "start" && \
#python automl_run.py --bucket_name shine_v3_7_kt --using_features patient+screason --onlyuse_selfcheck_first 1>> logs/automl_p_sf.log 2>> logs/automl_p_sf.log && \
#echo "finished" &

#nohup python automl_train.py --bucket_name shine_v3_7_kt --using_features patient+si10 --training_num 4 1>> logs/automl_ps4.log 2>> logs/automl_ps4.log &
#nohup python automl_train.py --bucket_name shine_v3_7_kt --using_features patient+si10 --training_num 5 1>> logs/automl_ps5.log 2>> logs/automl_ps5.log &
#
#nohup python automl_train.py --bucket_name shine_v3_7_kt --using_features patient+hw7 --training_num 2 1>> logs/automl_ps2.log 2>> logs/automl_ps2.log &
#nohup python automl_train.py --bucket_name shine_v3_7_kt --using_features patient+hw7 --training_num 3 1>> logs/automl_ps3.log 2>> logs/automl_ps3.log &
#nohup python automl_train.py --bucket_name shine_v3_7_kt --using_features patient+hw7 --training_num 4 1>> logs/automl_ps4.log 2>> logs/automl_ps4.log &

#nohup echo "start" && \
#python automl_run.py --bucket_name shine_v3_7_kt --using_features patient+screason+gps --onlyuse_selfcheck_first 1>> logs/automl_pg_sf.log 2>> logs/automl_pg_sf.log && \
#echo "finished" &

#nohup echo "start" && \
#python automl_run.py --bucket_name shine_v3_7_kt --using_features patient+screason+si10 --onlyuse_selfcheck_first 1>> logs/automl_ps_sf.log 2>> logs/automl_ps_sf.log && \
#echo "finished" &

#nohup python automl_train.py --bucket_name shine_v3_7_kt --using_features patient+screason+si10 --onlyuse_selfcheck_first --training_num 4 1>> logs/automl_ps_sf4.log 2>> logs/automl_ps_sf4.log &
#nohup python automl_train.py --bucket_name shine_v3_7_kt --using_features patient+screason+si10 --onlyuse_selfcheck_first --training_num 5 1>> logs/automl_ps_sf5.log 2>> logs/automl_ps_sf5.log &

#nohup echo "start" && \
#python automl_run.py --bucket_name shine_v3_7_kt --using_features patient+screason+hw7 --onlyuse_selfcheck_first 1>> logs/automl_ph_sf.log 2>> logs/automl_ph_sf.log && \
#echo "finished" &

#nohup echo "start" && \
#python automl_run.py --bucket_name shine_v3_7_kt --using_features patient+screason+gps+si10+hw7+owid --onlyuse_selfcheck_first 1>> logs/automl_pgsho_sf.log 2>> logs/automl_pgsho_sf.log && \
#echo "finished" &

#nohup echo 'sleep 8000' && sleep 8000 && echo "start" && \
#python automl_run.py --bucket_name shine_v3_7_kt --using_features patient+screason+owid --onlyuse_selfcheck_first 1>> logs/automl_po_sf.log 2>> logs/automl_po_sf.log && \
#echo "finished" &

nohup python automl_train.py --bucket_name shine_v3_7_kt --using_features patient+hw7 --training_num 5 1>> logs/automl_ph5.log 2>> logs/automl_ph5.log &

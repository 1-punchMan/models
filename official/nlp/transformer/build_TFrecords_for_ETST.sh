set -e

export PYTHONPATH="$PYTHONPATH:/home/zchen/encyclopedia-text-style-transfer/models/"

WIKIPATH="/home/zchen/encyclopedia-text-style-transfer/data/ETST/wiki/processed_data_cleaned/filtered/"
BAIDUPATH="/home/zchen/encyclopedia-text-style-transfer/data/ETST/baidu/processed_data_cleaned/filtered/"
VOC_PATH="/home/zchen/encyclopedia-text-style-transfer/data/vocab"

OUT_PATH="/home/zchen/encyclopedia-text-style-transfer/data/ETST/TFrecords/wiki/cleaned/"
python build_TFrecords_for_ETST.py --out_path=$OUT_PATH --data_path=$WIKIPATH --voc_path=$VOC_PATH

OUT_PATH="/home/zchen/encyclopedia-text-style-transfer/data/ETST/TFrecords/baidu/cleaned/"
python build_TFrecords_for_ETST.py --out_path=$OUT_PATH --data_path=$BAIDUPATH --voc_path=$VOC_PATH
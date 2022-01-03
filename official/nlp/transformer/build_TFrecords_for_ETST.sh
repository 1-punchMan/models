set -e

export PYTHONPATH="$PYTHONPATH:/home/zchen/encyclopedia-text-style-transfer/models/"

WIKIPATH="/home/zchen/encyclopedia-text-style-transfer/data/wiki/processed_data/"
BAIDUPATH="/home/zchen/encyclopedia-text-style-transfer/data/baidu/processed_data/tokenized/"
VOC_PATH="/home/zchen/encyclopedia-text-style-transfer/data/vocab"

OUT_PATH=./data/ETST/wiki
python build_TFrecords_for_ETST.py --out_path=$OUT_PATH --data_path=$WIKIPATH --voc_path=$VOC_PATH

OUT_PATH=./data/ETST/baidu
python build_TFrecords_for_ETST.py --out_path=$OUT_PATH --data_path=$BAIDUPATH --voc_path=$VOC_PATH
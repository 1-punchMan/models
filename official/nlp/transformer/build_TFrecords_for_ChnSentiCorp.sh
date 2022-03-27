set -e

export PYTHONPATH="$PYTHONPATH:/home/zchen/encyclopedia-text-style-transfer/models/"

NEG_PATH="/home/zchen/encyclopedia-text-style-transfer/data/ChnSentiCorp/processed/negative/"
POS_PATH="/home/zchen/encyclopedia-text-style-transfer/data/ChnSentiCorp/processed/positive/"
VOC_PATH="/home/zchen/encyclopedia-text-style-transfer/data/vocab"

OUT_PATH="/home/zchen/encyclopedia-text-style-transfer/data/ChnSentiCorp/TFrecords/negative/"
python build_TFrecords_for_ETST.py --out_path=$OUT_PATH --data_path=$NEG_PATH --voc_path=$VOC_PATH

OUT_PATH="/home/zchen/encyclopedia-text-style-transfer/data/ChnSentiCorp/TFrecords/positive/"
python build_TFrecords_for_ETST.py --out_path=$OUT_PATH --data_path=$POS_PATH --voc_path=$VOC_PATH
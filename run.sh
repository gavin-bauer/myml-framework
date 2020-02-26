export TRAIN_SET_PATH=data/interim/train_set.csv
export TEST_SET_PATH=data/interim/test_set.csv
export FITTED_PIPELINE_PATH=data/processed/fitted_pipeline.joblib
export MODEL=$1

python -m src.models.1-select_model
#python -m src.models.2-train_model
#python -m src.models.3-finetune_model
#python -m src.models.4-predict_model
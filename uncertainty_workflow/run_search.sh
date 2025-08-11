# run test script for hyperparameter search
# please set the model_path, metadata_path, logits_file_path, and output_base_path
# if you want to run on the val_dataset, change the ground_truth_path to the "val_dataset_split_GT.json"
python ./hyper_para_search.py \
    --ground_truth_path ./GT_data/test_lab-manual-combine-test-5768_GT_entity+path.json \
    --model_path your path \
    --metadata_path ./metadata_path \
    --logits_file_path ./logits_data_path \
    --output_base_path ./hyper_search_output \
    --seed 42 \
    
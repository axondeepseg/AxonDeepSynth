image_dir=$1
labels_dir=$2
output_dir=$3
model=$4
NNUNET_PATH=$5

model_name=${model#Dataset???_}

python scripts/run_inference.py --path-dataset $image_dir \
                                --path-out ${output_dir}/segmentations/ensemble/${model_name}_model_best_checkpoints_inference \
                                --path-model ${NNUNET_PATH}/nnUNet_results/${model}/nnUNetTrainer__nnUNetPlans__2d/ \
                                --use-gpu \
                                --use-best-checkpoint


python scripts/segmentation_evaluation.py   --pred_path ${output_dir}/segmentations/ensemble/${model_name}_model_best_checkpoints_inference \
                                            --gt_path ${labels_dir} \
                                            --output_fname ${output_dir}/segmentations/ensemble/scores/${model_name}_model_best_checkpoints_scores \
                                            --pred_suffix _0000


# Run inference and evaluation for each fold
for fold in {0..4}; do
    echo -e "\nProcessing fold $fold\n"
    python scripts/run_inference.py  --path-dataset $image_dir \
                                     --path-out ${output_dir}/segmentations/fold_${fold}/${model_name}_model_best_checkpoints_inference \
                                     --path-model ${NNUNET_PATH}/nnUNet_results/${model}/nnUNetTrainer__nnUNetPlans__2d/ \
                                     --use-gpu \
                                     --folds $fold \
                                     --use-best-checkpoint

    
    python scripts/segmentation_evaluation.py   --pred_path ${output_dir}/segmentations/fold_${fold}/${model_name}_model_best_checkpoints_inference \
                                                --gt_path ${labels_dir} \
                                                --output_fname ${output_dir}/segmentations/fold_${fold}/scores/${model_name}_model_best_checkpoints_scores \
                                                --pred_suffix _0000

done
# FIXED MASK RATIO
# etth2 - pretrain
python patchtst_pretrain.py --dset etth2 --mask_ratio 0.4 --n_epochs_pretrain 20 --batch_size 32 --n_heads 4 --d_model 16 --d_ff 128 >> run.logs
# etth2 - finetune
python patchtst_finetune.py --is_finetune 1 --dset_finetune etth2 --n_epochs_finetune 20 --batch_size 32 --n_heads 4 --d_model 16 --d_ff 128 --pretrained_model saved_models/etth2/masked_patchtst/based_model/patchtst_pretrained_cw512_patch12_stride12_epochs-pretrain20_mask0.4_model1.pth >> run.logs


# etth1 - pretrain
python patchtst_pretrain.py --dset etth1 --mask_ratio 0.4 --n_epochs_pretrain 20 --batch_size 32 --n_heads 4 --d_model 16 --d_ff 128 >> run.logs
# etth1 - finetune
python patchtst_finetune.py --is_finetune 1 --dset_finetune etth1 --n_epochs_finetune 20 --batch_size 32 --n_heads 4 --d_model 16 --d_ff 128 --pretrained_model saved_models/etth1/masked_patchtst/based_model/patchtst_pretrained_cw512_patch12_stride12_epochs-pretrain20_mask0.4_model1.pth >> run.logs


# RANDOM MASK RATIO
# etth2 - pretrain
python patchtst_pretrain.py --dset etth2 --n_epochs_pretrain 20 --batch_size 32 --n_heads 4 --d_model 16 --d_ff 128 --pretrained_model_id 2 >> run.logs
# etth2 - finetune
python patchtst_finetune.py --is_finetune 1 --dset_finetune etth2 --n_epochs_finetune 20 --batch_size 32 --n_heads 4 --d_model 16 --d_ff 128 --finetuned_model_id 2 --pretrained_model saved_models/etth2/masked_patchtst/based_model/patchtst_pretrained_cw512_patch12_stride12_epochs-pretrain20_maskNone_model2.pth >> run.logs

# etth1 - pretrain
python patchtst_pretrain.py --dset etth1 --n_epochs_pretrain 20 --batch_size 32 --n_heads 4 --d_model 16 --d_ff 128 --pretrained_model_id 2 >> run.logs
# etth1 - finetune
python patchtst_finetune.py --is_finetune 1 --dset_finetune etth1 --n_epochs_finetune 20 --batch_size 32 --n_heads 4 --d_model 16 --d_ff 128 --finetuned_model_id 2 --pretrained_model saved_models/etth1/masked_patchtst/based_model/patchtst_pretrained_cw512_patch12_stride12_epochs-pretrain20_maskNone_model2.pth >> run.logs


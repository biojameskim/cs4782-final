# python patchtst_pretrain.py --dset etth2 --mask_ratio 0.4 --n_epochs_pretrain 20 --batch_size 32 --n_heads 4 --d_model 16 --d_ff 128 >> run.logs
# python patchtst_pretrain.py --dset etth2 --n_epochs_pretrain 20 --batch_size 32 --n_heads 4 --d_model 16 --d_ff 128 >> run.logs
# python patchtst_pretrain.py --dset etth1 --mask_ratio 0.4 --n_epochs_pretrain 20 --batch_size 32 --n_heads 4 --d_model 16 --d_ff 128 >> run.logs
# python patchtst_pretrain.py --dset etth1 --n_epochs_pretrain 20 --batch_size 32 --n_heads 4 --d_model 16 --d_ff 128 >> run.logs

python patchtst_finetune.py --is_finetune 1 --dset_finetune etth1 --n_epochs_finetune 20 --batch_size 32 --n_heads 4 --d_model 16 --d_ff 128 --pretrained_model saved_models/etth1/masked_patchtst/based_model/patchtst_pretrained_cw512_patch12_stride12_epochs-pretrain20_maskNone_model1.pth >> run.logs

python patchtst_finetune.py --is_finetune 1 --dset_finetune etth2 --n_epochs_finetune 20 --batch_size 32 --n_heads 4 --d_model 16 --d_ff 128 --pretrained_model saved_models/etth2/masked_patchtst/based_model/patchtst_pretrained_cw512_patch12_stride12_epochs-pretrain20_maskNone_model1.pth >> run.logs

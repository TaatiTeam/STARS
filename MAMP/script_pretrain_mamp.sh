python3 -m torch.distributed.launch --nproc_per_node=4 --master_port 11234 main_pretrain.py \
--config ./config/pkuv2_xsub_joint/pretrain_mamp_t120_layer8+5_mask90.yaml \
--output_dir ./output_dir/pkuv2_xsub_joint/pretrain_mamp_t120_layer8+5_mask90_tau0.80_ep400_noamp \
--log_dir ./output_dir/pkuv2_xsub_joint/pretrain_mamp_t120_layer8+5_mask90_tau0.80_ep400_noamp

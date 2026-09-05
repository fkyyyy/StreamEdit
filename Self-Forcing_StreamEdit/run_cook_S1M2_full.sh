#!/bin/bash
# S1+M2 Full System: Soft Region Modulation + Role-Aware Immutable Bank
# S1: hand_posterior_threshold=0.3 for soft object posterior modulation
# M1: immutable_delta_v_bank with topk=16, strength=0.8
# M2: role_object_posterior as owner_gate (automatic when S1 enabled)

python inference_edit_streamedit.py \
    --pretrained_model_name_or_path "genmo/mochi-1-preview" \
    --validation_prompt_file "configs/prompts_wallet.txt" \
    --num_frames 163 \
    --width 848 \
    --height 480 \
    --num_inference_steps 64 \
    --guidance_scale 4.5 \
    --rollout_chunk_size 20 \
    --num_frame_per_block 20 \
    --seed 967 \
    --output_dir "outputs/cook_S1M2_full" \
    --routing_mode "dynamic_sog" \
    --blend_power 0.0 \
    --inject_step 30 \
    --inject_step_end 32 \
    --mask_layers 0 0 0 0 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 \
    --enhance_layers 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 \
    --hand_tracking_checkpoint "/mnt/data/checkpoints/hand_tracking/sapiens_1b_goliath_best_goliath_AP_573.pth" \
    --hand_model "sapiens_1b" \
    --hand_mask_mode "goliath" \
    --hand_only_mask_path "data/hand_masks/wallet/hand_only" \
    --hand_occupancy_mask_path "data/hand_masks/wallet/occupancy" \
    --hand_persistent_mask_path "data/hand_masks/wallet/persistent" \
    --soft_region_modulation \
    --hand_posterior_threshold 0.3 \
    --hand_role_enabled \
    --immutable_delta_v_bank \
    --immutable_delta_v_topk 16 \
    --immutable_delta_v_strength 0.8 \
    --immutable_delta_v_min_similarity 0.3 \
    --immutable_delta_v_max_rms_ratio 2.0 \
    --closed_loop_delta_v_max_error_ratio 1.5 \
    --immutable_delta_v_layers 0 0 0 0 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1

echo "S1+M2 full system inference complete. Check outputs/cook_S1M2_full/"

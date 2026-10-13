# python train.py     -s /data/albedo/skyanchor/stella_vslam_dense/output/staff_full/colmap_split     -m output/360/staff_full      --data_device cpu     --images images     --ground_source_path /data/albedo/skyanchor/stella_vslam_dense/output/staff_full/colmap_split     --ground_images images           --eval --aerial_stage_iterations 30000   --ground_stage_iterations 30000   --joint_finetune_iterations 40000 --dense_init_ply_name dense_v2.ply


# python train.py     -s /data/albedo/SkyAlbedo/gaussian_large_scale/data/road/aerial     -m output/road/aerial_ground1      --data_device cpu     --images images     --ground_source_path /data/albedo/SkyAlbedo/gaussian_large_scale/data/road/aeiral     --ground_images images           --eval --aerial_stage_iterations 30000   --ground_stage_iterations 30000   --joint_finetune_iterations 40000 --dense_init_ply_name dense_v2.ply


# python render.py     -m output/road/aerial_ground1 --skip_train

# python metrics.py -m output/road/aerial_ground1



# python train.py     -s /data/albedo/SkyAlbedo/gaussian_large_scale/data/uni100k/ground     -m output/uni100k/aerial_ground1      --data_device cpu     --images images     --ground_source_path /data/albedo/SkyAlbedo/gaussian_large_scale/data/uni100k/ground/     --ground_images images           --eval --aerial_stage_iterations 30000   --ground_stage_iterations 30000   --joint_finetune_iterations 40000 --dense_init_ply_name dense_v2.ply


# python render.py     -m output/uni100k/aerial_ground1 --skip_train

# python metrics.py -m output/park/aerial_ground1





python train.py     -s /data/albedo/SkyAlbedo/gaussian_large_scale/data/park/aerial/     -m output/park/aerial_ground1      --data_device cpu     --images images     --ground_source_path /data/albedo/SkyAlbedo/gaussian_large_scale/data/park/ground/     --ground_images images_masked     --pair_file data/park/aerial/pair.txt     --ground_pair_file data/park/ground/pair.txt --eval --aerial_stage_iterations 30000   --ground_stage_iterations 30000   --joint_finetune_iterations 40000 --dense_init_ply_name dense_v2.ply


python render.py     -m output/park/aerial_ground1 --skip_train

python metrics.py -m output/park/aerial_ground1
    
# python render.py \
#     -m output/park/aerial_ground1 \
#     --render_video \
#     --trajectory_source /data/albedo/SkyAlbedo/gaussian_large_scale/data/park/ground/ \
#     --trajectory_images images \
#     --video_duration 10 \
#     --video_fps 30 \
#     --video_name ground_flythrough \
#     --rot_smooth_window 5



# python train.py     -s /data/albedo/SkyAlbedo/gaussian_large_scale/data/rataus/aerial/     -m output/rataus/aerial_ground1      --data_device cpu     --images images_masked     --ground_source_path /data/albedo/SkyAlbedo/gaussian_large_scale/data/rataus/ground/     --ground_images images_masked     --pair_file data/rataus/aerial/pair.txt     --ground_pair_file data/rataus/ground/pair.txt --eval --aerial_stage_iterations 30000   --ground_stage_iterations 30000   --joint_finetune_iterations 40000 --dense_init_ply_name dense_v2.ply


# python render.py     -m output/rataus/aerial_ground1 --skip_train

# python metrics.py -m output/rataus/aerial_ground1
    
# # python render.py \
# #     -m output/rataus/aerial_ground1 \
# #     --render_video \
# #     --trajectory_source /data/albedo/SkyAlbedo/gaussian_large_scale/data/rataus/ground/ \
# #     --trajectory_images images \
# #     --video_duration 10 \
# #     --video_fps 30 \
# #     --video_name ground_flythrough \
# #     --rot_smooth_window 5

# python train.py     -s /data/albedo/SkyAlbedo/gaussian_large_scale/data/staff/aerial/     -m output/staff/aerial_ground1      --data_device cpu     --images images     --ground_source_path /data/albedo/SkyAlbedo/gaussian_large_scale/data/staff/ground/     --ground_images images     --pair_file data/staff/aerial/pair.txt     --ground_pair_file data/staff/ground/pair.txt --eval --aerial_stage_iterations 30000   --ground_stage_iterations 30000   --joint_finetune_iterations 40000 --dense_init_ply_name dense_v2.ply


# python render.py     -m output/staff/aerial_ground1 --skip_train

# python metrics.py -m output/staff/aerial_ground1
    
# # python render.py \
# #     -m output/staff/aerial_ground1 \
# #     --render_video \
# #     --trajectory_source /data/albedo/SkyAlbedo/gaussian_large_scale/data/staff/ground/ \
# #     --trajectory_images images \
# #     --video_duration 10 \
# #     --video_fps 30 \
# #     --video_name ground_flythrough \
# #     --rot_smooth_window 5




# python train.py     -s /data/albedo/SkyAlbedo/gaussian_large_scale/data/zeche/aerial/     -m output/zeche/aerial_ground1      --data_device cpu     --images images_masked     --ground_source_path /data/albedo/SkyAlbedo/gaussian_large_scale/data/zeche/ground/     --ground_images images_masked     --pair_file data/zeche/aerial/pair.txt     --ground_pair_file data/zeche/ground/pair.txt --eval --aerial_stage_iterations 30000   --ground_stage_iterations 30000   --joint_finetune_iterations 40000 --dense_init_ply_name dense_v2.ply


# python render.py     -m output/zeche/aerial_ground1 --skip_train

# python metrics.py -m output/zeche/aerial_ground1
    
# # python render.py \
# #     -m output/zeche/aerial_ground1   \
# #     --render_video \
# #     --trajectory_source /data/albedo/SkyAlbedo/gaussian_large_scale/data/zeche/ground/ \
# #     --trajectory_images images \
# #     --video_duration 10 \
# #     --video_fps 30 \
# #     --video_name ground_flythrough \
# #     --rot_smooth_window 5



# !/bin/bash

BASE_DIR="/root/pingpang-video-analyzer/server-api/app/round_detect/star_dir"

# Run the round detection server
python round_cut.py \
  --video_path "/root/pingpang-video-analyzer/server-api/app/round_detect/output_video_球星挑战赛.mp4" \
  --output_path "${BASE_DIR}/res.json" \
  --preview_dir "${BASE_DIR}/preview" \
  --query_path "${BASE_DIR}" \
  --progress_path "${BASE_DIR}/progress.json" \
  --player1_num 9 \
  --player2_num 9 \
  --table_loc_path "/root/pingpang-video-analyzer/server-api/app/round_detect/table_loc.json" \


# # Run cut video
# python cut_video.py \
#   --video_path "/root/pingpang-video-analyzer/server-api/app/round_detect/temp/output_video.mp4" \
#   --output_path "/root/pingpang-video-analyzer/server-api/app/round_detect/temp/round_cut/videos" \
#   --clips_path "/root/pingpang-video-analyzer/server-api/app/round_detect/temp/round_cut/res.json" \
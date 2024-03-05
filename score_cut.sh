# !/bin/bash

BASE_DIR="/root/pingpang-video-analyzer/server-api/app/round_detect/star_dir/score_cut"

# python score_cut.py \
#   --video_path "/root/pingpang-video-analyzer/server-api/app/round_detect/output_video_球星挑战赛.mp4" \
#   --output_path "${BASE_DIR}/res.json" \
#   --preview_dir "${BASE_DIR}/preview" \
#   --query_path "${BASE_DIR}/../" \
#   --progress_path "${BASE_DIR}/progress.json" \
#   --round_clips "/root/pingpang-video-analyzer/server-api/app/round_detect/star_dir/res.json"

python cut_video.py \
  --video_path="/root/pingpang-video-analyzer/server-api/app/round_detect/output_video_球星挑战赛.mp4" \
  --output_path "${BASE_DIR}/videos" \
  --clips_path "/root/pingpang-video-analyzer/server-api/app/round_detect/star_dir/score_cut/res.json" \
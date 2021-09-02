python \
detect_letter.py \
--brand Chanel \
--part sign \
--weights /media/D_4TB/YL_4TB/BaoDetection/data/Chanel/LetterDetection/data/yolov5_rundata/train20/weights/ckp_aug_all_6c_epoch1760.pt \
--source /media/D_4TB/YL_4TB/BaoDetection/data/Chanel/LetterDetection/data/sign/sign_2_need_已鉴定标记 \
--batch-size 10 \
--iou-thres 0.2 \
--conf-thres 0.5

python \
train_letter.py \
--img 640 \
--brand Chanel \
--part sign \
--hyp data/hyp.baoletter.yaml \
--batch 100 \
--epochs 10000 \
--weights yolov5s.pt \
--device 0,1,2,3 \
-train aug_all.txt \
-test all.txt \
--test-inter 20 \
--save-inter 20 \

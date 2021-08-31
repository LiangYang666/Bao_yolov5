python \
train_letter.py \
--img 640 \
--brand Chanel \
--part sign \
--batch 4 \
--epochs 60000 \
--weights yolov5s.pt \
--device 0 \
-train all.txt \
-test all.txt

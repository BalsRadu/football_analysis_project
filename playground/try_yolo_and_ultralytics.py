from ultralytics import YOLO

model = YOLO('models/best.pt')

results = model.predict('samples\input_video.mp4', save=True, device='cuda')

print(results[0])
print("="*150)
for box in results[0].boxes:
    print(box)
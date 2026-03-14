from ultralytics import YOLO
import os

def main():
    # 1. Load a pretrained YOLOv8 segmentation model
    # 'n' stands for Nano - it's the fastest and best for smaller datasets
    model = YOLO("yolov8n-seg.pt")

    # 2. Train the model
    # This will automatically look for your data.yaml
    results = model.train(
        data="data.yaml",
        epochs=100,         # High enough to learn the 5 parts
        imgsz=640,          # Standard resolution for clarity
        device="cpu",       # Change to 0 if you have an NVIDIA GPU
        plots=True,         # Generates charts so you can see accuracy
        name="TANCAM_Final_Train"
    )

    print("--- Training Complete ---")
    print("Model saved to: runs/segment/TANCAM_Final_Train/weights/best.pt")

if __name__ == "__main__":
    main()
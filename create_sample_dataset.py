"""
Create a minimal sample dataset for testing the FruitGuard pipeline.
This creates placeholder images to allow testing the training/evaluation workflow.
"""

from pathlib import Path
import json
from PIL import Image, ImageDraw, ImageFont
import random

# Load the taxonomy
taxonomy_path = Path("ml/taxonomy.json")
with open(taxonomy_path, encoding="utf-8") as f:
    taxonomy = json.load(f)

# Define the target directory
data_dir = Path("ml/data/raw")
data_dir.mkdir(parents=True, exist_ok=True)

print("Creating sample dataset for testing...")
print("This will create minimal placeholder images to test the pipeline.")
print()

# Create sample images for each class
total_images = 0
for cls in taxonomy["classes"]:
    class_name = cls["plantvillage_folder"]  # Use plantvillage_folder for dataset compatibility
    class_dir = data_dir / class_name
    class_dir.mkdir(parents=True, exist_ok=True)
    
    # Create 10 sample images per class
    for i in range(10):
        # Create a random colored image
        img = Image.new('RGB', (224, 224), color=(
            random.randint(50, 200),
            random.randint(50, 200), 
            random.randint(50, 200)
        ))
        
        # Add some random noise/patterns
        draw = ImageDraw.Draw(img)
        for _ in range(20):
            x1 = random.randint(0, 224)
            y1 = random.randint(0, 224)
            x2 = random.randint(x1, 224)
            y2 = random.randint(y1, 224)
            draw.ellipse([x1, y1, x2, y2], fill=(
                random.randint(100, 255),
                random.randint(100, 255),
                random.randint(100, 255)
            ))
        
        # Save the image
        img_path = class_dir / f"sample_{i:03d}.jpg"
        img.save(img_path, quality=85)
    
    total_images += 10
    print(f"Created 10 sample images for {class_name}")

print(f"\nTotal sample images created: {total_images}")
print(f"Dataset location: {data_dir}")
print("\nNOTE: These are placeholder images for testing only.")
print("For real results, download the actual PlantVillage dataset using the instructions shown earlier.")
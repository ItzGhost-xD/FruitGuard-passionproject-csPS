"""
Create OOD (Out-of-Distribution) phone-photo dataset for testing.
This simulates real-world phone photos with different lighting/conditions.
"""

from pathlib import Path
import json
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import random

# Load the taxonomy
taxonomy_path = Path("ml/taxonomy.json")
with open(taxonomy_path, encoding="utf-8") as f:
    taxonomy = json.load(f)

# Define the target directory
ood_dir = Path("ml/data/ood_phone")
ood_dir.mkdir(parents=True, exist_ok=True)

print("Creating OOD phone-photo dataset for testing...")
print("This simulates real-world phone photos with varying conditions.")
print()

# Create sample images for each class with "phone photo" characteristics
total_images = 0
for cls in taxonomy["classes"]:
    class_name = cls["plantvillage_folder"]  # Use plantvillage_folder for dataset compatibility
    class_dir = ood_dir / class_name
    class_dir.mkdir(parents=True, exist_ok=True)
    
    # Create 5 sample images per class for OOD testing
    for i in range(5):
        # Create a base image with more realistic colors
        base_color = (
            random.randint(30, 150),
            random.randint(50, 180), 
            random.randint(20, 120)
        )
        img = Image.new('RGB', (224, 224), color=base_color)
        
        # Add more complex patterns to simulate real leaves
        draw = ImageDraw.Draw(img)
        
        # Add leaf-like shapes
        for _ in range(10):
            x1 = random.randint(10, 200)
            y1 = random.randint(10, 200)
            x2 = random.randint(x1, 214)
            y2 = random.randint(y1, 214)
            draw.ellipse([x1, y1, x2, y2], fill=(
                random.randint(80, 200),
                random.randint(100, 220),
                random.randint(50, 150)
            ))
        
        # Add some "noise" to simulate phone camera artifacts
        for _ in range(50):
            x = random.randint(0, 223)
            y = random.randint(0, 223)
            img.putpixel((x, y), (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255)
            ))
        
        # Apply some blur to simulate out-of-focus areas
        if random.random() > 0.5:
            img = img.filter(ImageFilter.BLUR)
        
        # Adjust brightness to simulate different lighting conditions
        if random.random() > 0.5:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(random.uniform(0.7, 1.3))
        
        # Save the image
        img_path = class_dir / f"phone_photo_{i:03d}.jpg"
        img.save(img_path, quality=80)
    
    total_images += 5
    print(f"Created 5 OOD phone-style images for {class_name}")

print(f"\nTotal OOD images created: {total_images}")
print(f"OOD dataset location: {ood_dir}")
print("\nNOTE: These are placeholder images simulating phone photos.")
print("For real research results, collect actual phone photos in this directory.")
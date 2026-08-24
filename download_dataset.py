"""
Dataset download script for FruitGuard project.
Supports multiple methods for obtaining the PlantVillage dataset.
"""

from pathlib import Path
import json
import zipfile
import shutil
import sys
from dotenv import load_dotenv
import os

load_dotenv()


kaggle_username = os.getenv("KAGGLE_USERNAME")
kaggle_key = os.getenv("KAGGLE_KEY")

# Define the target directory
data_dir = Path("ml/data/raw")
data_dir.mkdir(parents=True, exist_ok=True)

# Load the taxonomy to get the classes we need
taxonomy_path = Path("ml/taxonomy.json")
with open(taxonomy_path, encoding="utf-8") as f:
    taxonomy = json.load(f)

# Get the PlantVillage folder names for our target classes
target_classes = {cls["plantvillage_folder"] for cls in taxonomy["classes"]}

print("=" * 60)
print("FruitGuard Dataset Setup")
print("=" * 60)
print(f"Target classes ({len(target_classes)}):")
for cls in sorted(target_classes):
    print(f"  - {cls}")
print()

def try_kaggle_download():
    """Try to download using Kaggle API"""
    if not kaggle_username or not kaggle_key:
    raise RuntimeError(
        "KAGGLE_USERNAME and KAGGLE_KEY must be set as environment variables."
    )

    try:
        from kaggle import KaggleApi
        
        print("Attempting Kaggle API download...")
        
        
        api = KaggleApi()
        
        try:
            api.authenticate()
        except Exception as auth_error:
            print(f"Authentication error: {auth_error}")
            print("Note: Kaggle API authentication has changed. Please use manual download.")
            return False
        
        # Download the dataset
        print("Downloading dataset from Kaggle...")
        dataset_name = "emmarex/plantdisease"
        api.dataset_download_files(dataset_name, path="ml/data", unzip=False)
        
        # The downloaded file will be something like plantdisease.zip
        zip_files = list(Path("ml/data").glob("*.zip"))
        if not zip_files:
            raise FileNotFoundError("No zip file found after download")
        
        zip_file = zip_files[0]
        print(f"Extracting {zip_file}...")
        
        # Extract to a temporary directory first
        temp_dir = Path("ml/data/temp_extract")
        temp_dir.mkdir(exist_ok=True)
        
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Find the extracted directory structure
        extracted_dirs = [d for d in temp_dir.iterdir() if d.is_dir()]
        
        if not extracted_dirs:
            raise FileNotFoundError("No directories found in extracted archive")
        
        # Move only the target classes to our data directory
        print("Filtering and moving target classes...")
        total_moved = 0
        
        for extracted_dir in extracted_dirs:
            # Check if this directory name matches any of our target classes
            if extracted_dir.name in target_classes:
                target_dir = data_dir / extracted_dir.name
                if target_dir.exists():
                    shutil.rmtree(target_dir)
                shutil.move(str(extracted_dir), str(target_dir))
                
                # Count images
                jpg_count = len(list(target_dir.glob("*.JPG")))
                jpg_lower_count = len(list(target_dir.glob("*.jpg")))
                jpeg_count = len(list(target_dir.glob("*.jpeg")))
                png_count = len(list(target_dir.glob("*.png")))
                img_count = jpg_count + jpg_lower_count + jpeg_count + png_count
                print(f"  - {extracted_dir.name}/ ({img_count} images)")
                total_moved += img_count
        
        # Clean up
        shutil.rmtree(temp_dir)
        zip_file.unlink()
        
        print(f"\nSuccessfully downloaded and organized {total_moved} images!")
        print(f"Dataset ready in {data_dir}")
        return True
        
    except ImportError:
        print("Kaggle library not installed. Install with: pip install kaggle")
        return False
    except Exception as e:
        print(f"Kaggle download failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def try_manual_zip():
    """Try to find and extract a manually downloaded zip file"""
    print("\nLooking for manually downloaded zip files...")
    
    # Common locations where user might have downloaded the file
    search_paths = [
        Path("."),
        Path("ml/data"),
        Path("Downloads"),
        Path.home() / "Downloads",
    ]
    
    zip_files = []
    for search_path in search_paths:
        if search_path.exists():
            zip_files.extend(search_path.glob("*.zip"))
            zip_files.extend(search_path.glob("plant*.zip"))
            zip_files.extend(search_path.glob("PlantVillage*.zip"))
    
    if not zip_files:
        print("No zip files found in common locations.")
        return False
    
    print(f"Found {len(zip_files)} zip file(s):")
    for i, zip_file in enumerate(zip_files, 1):
        print(f"  {i}. {zip_file}")
    
    # Use the first zip file found
    zip_file = zip_files[0]
    print(f"\nUsing: {zip_file}")
    
    try:
        print("Extracting...")
        temp_dir = Path("ml/data/temp_extract")
        temp_dir.mkdir(exist_ok=True)
        
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Find the extracted directory structure
        extracted_dirs = [d for d in temp_dir.rglob("*") if d.is_dir()]
        
        # Look for directories that match our target classes
        print("Filtering and moving target classes...")
        total_moved = 0
        
        for extracted_dir in extracted_dirs:
            if extracted_dir.name in target_classes:
                target_dir = data_dir / extracted_dir.name
                if target_dir.exists():
                    shutil.rmtree(target_dir)
                shutil.move(str(extracted_dir), str(target_dir))
                
                # Count images
                jpg_count = len(list(target_dir.glob("*.JPG")))
                jpg_lower_count = len(list(target_dir.glob("*.jpg")))
                jpeg_count = len(list(target_dir.glob("*.jpeg")))
                png_count = len(list(target_dir.glob("*.png")))
                img_count = jpg_count + jpg_lower_count + jpeg_count + png_count
                print(f"  - {extracted_dir.name}/ ({img_count} images)")
                total_moved += img_count
        
        # Clean up
        shutil.rmtree(temp_dir)
        
        if total_moved > 0:
            print(f"\nSuccessfully organized {total_moved} images!")
            print(f"Dataset ready in {data_dir}")
            return True
        else:
            print("No matching class directories found in the zip file.")
            return False
            
    except Exception as e:
        print(f"Error extracting zip file: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_manual_instructions():
    """Show instructions for manual download"""
    print("\n" + "=" * 60)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("=" * 60)
    print("\nOption 1: Kaggle")
    print("  1. Visit: https://www.kaggle.com/datasets/emmarex/plantdisease")
    print("  2. Download the dataset")
    print("  3. Extract and place class folders in ml/data/raw/")
    print("  Required folders:")
    for cls in sorted(target_classes):
        print(f"    - ml/data/raw/{cls}/")
    
    print("\nOption 2: Original PlantVillage")
    print("  1. Visit: https://github.com/spMohanty/PlantVillage-Dataset")
    print("  2. Download the dataset")
    print("  3. Extract and place class folders in ml/data/raw/")
    
    print("\nOption 3: Create minimal sample dataset")
    print("  Run: python create_sample_dataset.py")
    print("  (This creates a small dataset for testing)")
    
    print("\nExpected folder structure:")
    print("  ml/data/raw/")
    for cls in sorted(target_classes):
        print(f"    {cls}/")
        print(f"      image1.jpg")
        print(f"      image2.jpg")
        print(f"      ...")

# Try different methods
success = False

# Method 1: Try Kaggle API
if not success:
    success = try_kaggle_download()

# Method 2: Try manual zip file
if not success:
    success = try_manual_zip()

# If all methods failed, show instructions
if not success:
    show_manual_instructions()
    sys.exit(1)

print("\nDataset setup complete!")
"""
setup_microsoft.py
==================
Microsoft ke kagglecatsanddogs zip ke liye
Automatically PetImages/Cat aur PetImages/Dog se
dataset/cats/ aur dataset/dogs/ mein copy karta hai

Usage:
    python setup_microsoft.py
"""

import os
import shutil
from PIL import Image

SRC_CAT  = os.path.join("PetImages", "Cat")
SRC_DOG  = os.path.join("PetImages", "Dog")
DEST_CAT = os.path.join("dataset", "cats")
DEST_DOG = os.path.join("dataset", "dogs")
MAX      = 500   # kitni images chahiye per class

def copy_images(src, dest, label, max_count):
    os.makedirs(dest, exist_ok=True)
    files = [f for f in os.listdir(src)
             if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    count = 0
    skipped = 0
    for fname in files:
        if count >= max_count:
            break
        src_path  = os.path.join(src, fname)
        dest_path = os.path.join(dest, fname)
        try:
            # Corrupt images skip karo
            img = Image.open(src_path)
            img.verify()
            shutil.copy2(src_path, dest_path)
            count += 1
        except Exception:
            skipped += 1
    print(f"  {label}: {count} images copied, {skipped} skipped (corrupt)")
    return count

def main():
    print("\n🐾 Microsoft Pet Dataset Setup")
    print("=" * 40)

    # PetImages folder dhundo
    if not os.path.exists("PetImages"):
        # Check karo zip extract hua ya nahi
        zip_name = "kagglecatsanddogs_5340.zip"
        alt_name = "kagglecatsanddogs_5340 (1).zip"
        found = None
        for z in [zip_name, alt_name]:
            if os.path.exists(z):
                found = z
                break
        if found:
            print(f"  Zip mila: {found}")
            print("  Extract kar raha hoon...")
            import zipfile
            with zipfile.ZipFile(found, 'r') as zf:
                zf.extractall(".")
            print("  Extract complete!")
        else:
            print("❌  PetImages folder ya zip nahi mila!")
            print("   Zip file ko project folder mein rakh ke dubara run karo.")
            return

    print(f"\n  Copying up to {MAX} images per class...")
    cats = copy_images(SRC_CAT, DEST_CAT, "🐱 Cats", MAX)
    dogs = copy_images(SRC_DOG, DEST_DOG, "🐶 Dogs", MAX)

    print(f"\n✅  Done!")
    print(f"   dataset/cats/ → {cats} images")
    print(f"   dataset/dogs/ → {dogs} images")
    print(f"\n   Ab run karo:")
    print(f"   python train_model.py")
    print(f"   streamlit run app.py")

if __name__ == "__main__":
    main()

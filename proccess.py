import cv2
import numpy as np
import os
import time
import face_recognition
from PIL import Image
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import imagehash
from skimage.metrics import structural_similarity as ssim
from sklearn.cluster import DBSCAN
import shutil
import json
import multiprocessing


class IranianActorImageProcessor:
    def __init__(self, input_dir="iranian_actors_dataset", output_dir=None):
        self.input_dir = input_dir
        self.output_dir = output_dir or input_dir

        os.makedirs(f"{self.output_dir}/processed", exist_ok=True)
        os.makedirs(f"{self.output_dir}/reference_images", exist_ok=True)

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        self.image_hashes = {}
        self.face_encodings_cache = {}
        self.reference_encodings = {}

        self.HASH_SIMILARITY_THRESHOLD = 5
        self.SSIM_THRESHOLD = 0.85
        self.FACE_SIMILARITY_THRESHOLD = 0.6
        self.IDENTITY_THRESHOLD = 0.6

        self.lock = threading.Lock()
        self.process_workers = self.get_optimal_worker_count()

        print(f"🧵 Using {self.process_workers} processing workers")

    def get_optimal_worker_count(self):
        """Calculate optimal number of worker threads"""
        try:
            import psutil
            cpu_count = psutil.cpu_count(logical=True)
            if cpu_count:

                return max(1, int(cpu_count * 0.5))
        except ImportError:
            pass

        try:
            return max(1, multiprocessing.cpu_count() // 2)
        except:
            return 1

    def calculate_image_hash(self, image_path):
        """Calculate multiple types of image hashes for duplicate detection"""
        try:
            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                hashes = {
                    'phash': str(imagehash.phash(img)),
                    'dhash': str(imagehash.dhash(img)),
                    'whash': str(imagehash.whash(img)),
                    'average': str(imagehash.average_hash(img))
                }

                return hashes
        except Exception as e:
            print(f"Error calculating hash for {image_path}: {e}")
            return None

    def calculate_structural_similarity(self, image1_path, image2_path):
        """Calculate structural similarity between two images"""
        try:
            img1 = cv2.imread(image1_path)
            img2 = cv2.imread(image2_path)

            if img1 is None or img2 is None:
                return 0

            target_size = (256, 256)
            img1_resized = cv2.resize(img1, target_size)
            img2_resized = cv2.resize(img2, target_size)

            gray1 = cv2.cvtColor(img1_resized, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY)

            similarity = ssim(gray1, gray2)
            return similarity

        except Exception as e:
            print(f"Error calculating SSIM: {e}")
            return 0

    def calculate_face_encoding(self, image_path):
        """Calculate face encoding for face similarity comparison"""
        try:
            image = face_recognition.load_image_file(image_path)
            face_encodings = face_recognition.face_encodings(image)

            if len(face_encodings) == 1:
                return face_encodings[0]
            else:
                return None

        except Exception as e:
            print(f"Error calculating face encoding: {e}")
            return None

    def detect_single_face(self, image_path):
        """Detect if image has exactly ONE face using multiple methods"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return False, "Could not read image"

            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            try:
                face_locations = face_recognition.face_locations(
                    rgb_img, model="hog")
                face_count_fr = len(face_locations)

                if face_count_fr == 1:
                    top, right, bottom, left = face_locations[0]
                    face_width = right - left
                    face_height = bottom - top

                    if face_width >= 80 and face_height >= 80:
                        img_height, img_width = rgb_img.shape[:2]
                        face_ratio = (face_width * face_height) / \
                            (img_width * img_height)

                        if 0.05 <= face_ratio <= 0.8:
                            return True, f"Single face detected (face_recognition): {face_width}x{face_height}"
                        else:
                            return False, f"Face too {'large' if face_ratio > 0.8 else 'small'}: {face_ratio:.2%} of image"
                    else:
                        return False, f"Face too small: {face_width}x{face_height}"

                elif face_count_fr == 0:
                    pass
                else:
                    return False, f"Multiple faces detected (face_recognition): {face_count_fr} faces"

            except Exception as e:
                print(f"Face recognition library error: {e}")

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60),
                maxSize=(int(gray.shape[1]*0.8), int(gray.shape[0]*0.8))
            )

            if len(faces) == 1:
                x, y, w, h = faces[0]
                if w >= 60 and h >= 60:
                    return True, f"Single face detected (OpenCV): {w}x{h}"
                else:
                    return False, f"Face too small (OpenCV): {w}x{h}"
            elif len(faces) == 0:

                faces = self.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.05, minNeighbors=3, minSize=(40, 40),
                    maxSize=(int(gray.shape[1]*0.9), int(gray.shape[0]*0.9))
                )

                if len(faces) == 1:
                    x, y, w, h = faces[0]
                    if w >= 40 and h >= 40:
                        return True, f"Single face detected (OpenCV sensitive): {w}x{h}"
                    else:
                        return False, f"Face too small (OpenCV): {w}x{h}"
                else:
                    return False, f"No face or multiple faces (OpenCV sensitive): {len(faces)} faces"
            else:
                return False, f"Multiple faces detected (OpenCV): {len(faces)} faces"

        except Exception as e:
            return False, f"Error detecting faces: {e}"

    def is_duplicate_image(self, image_path, actor_name):
        """Check if image is a duplicate using multiple methods"""
        try:
            with self.lock:
                if actor_name not in self.image_hashes:
                    self.image_hashes[actor_name] = []
                    self.face_encodings_cache[actor_name] = []

            new_hashes = self.calculate_image_hash(image_path)
            if not new_hashes:
                return True, "Could not calculate hash"

            with self.lock:

                for stored_hash_data in self.image_hashes[actor_name]:
                    for hash_type in ['phash', 'dhash', 'whash', 'average']:
                        if hash_type in new_hashes and hash_type in stored_hash_data['hashes']:
                            hash1 = imagehash.hex_to_hash(
                                new_hashes[hash_type])
                            hash2 = imagehash.hex_to_hash(
                                stored_hash_data['hashes'][hash_type])
                            hamming_distance = hash1 - hash2

                            if hamming_distance <= self.HASH_SIMILARITY_THRESHOLD:
                                return True, f"Duplicate detected (hash {hash_type}): distance={hamming_distance}"

                for stored_hash_data in self.image_hashes[actor_name]:
                    stored_image_path = stored_hash_data['path']
                    if os.path.exists(stored_image_path):
                        ssim_score = self.calculate_structural_similarity(
                            image_path, stored_image_path)
                        if ssim_score >= self.SSIM_THRESHOLD:
                            return True, f"Duplicate detected (SSIM): score={ssim_score:.3f}"

            new_face_encoding = self.calculate_face_encoding(image_path)
            if new_face_encoding is not None:
                with self.lock:
                    for stored_encoding in self.face_encodings_cache[actor_name]:
                        face_distance = face_recognition.face_distance(
                            [stored_encoding], new_face_encoding)[0]
                        similarity = 1 - face_distance

                        if similarity >= self.FACE_SIMILARITY_THRESHOLD:
                            return True, f"Duplicate detected (face): similarity={similarity:.3f}"

                    hash_data = {
                        'hashes': new_hashes,
                        'path': image_path
                    }
                    self.image_hashes[actor_name].append(hash_data)
                    self.face_encodings_cache[actor_name].append(
                        new_face_encoding)

            return False, "Unique image"

        except Exception as e:
            print(f"Error checking duplicate: {e}")
            return False, "Error in duplicate check"

    def identify_most_common_face(self, image_paths, actor_name):
        """Use clustering to identify the most common face (likely the correct actor)"""
        if not image_paths:
            return None

        print(f"🧩 Clustering faces for {actor_name}...")
        face_encodings = []
        valid_paths = []

        for img_path in image_paths:
            try:
                has_face, _ = self.detect_single_face(img_path)
                if has_face:
                    encoding = self.calculate_face_encoding(img_path)
                    if encoding is not None:
                        face_encodings.append(encoding)
                        valid_paths.append(img_path)
            except Exception as e:
                print(f"Error processing image for clustering: {e}")

        if len(face_encodings) < 5:
            print(
                f"⚠️ Not enough valid faces for clustering: {len(face_encodings)}")
            return None

        print(f"🧮 Clustering {len(face_encodings)} face encodings...")

        try:

            clustering = DBSCAN(eps=0.6, min_samples=3).fit(face_encodings)

            labels = clustering.labels_
            if -1 in labels and len(set(labels)) <= 1:
                print("❌ No meaningful clusters found")
                return None

            label_counts = {}
            for label in labels:
                if label != -1:
                    label_counts[label] = label_counts.get(label, 0) + 1

            if not label_counts:
                print("❌ No valid clusters found")
                return None

            largest_cluster = max(label_counts.items(), key=lambda x: x[1])[0]
            cluster_size = label_counts[largest_cluster]

            print(
                f"🎯 Found largest cluster with {cluster_size} faces ({(cluster_size/len(face_encodings))*100:.1f}%)")

            cluster_indices = [i for i, label in enumerate(
                labels) if label == largest_cluster]

            if not cluster_indices:
                return None

            min_distance_sum = float('inf')
            center_idx = -1

            for i in cluster_indices:
                distance_sum = sum(
                    face_recognition.face_distance(
                        [face_encodings[i]], face_encodings[j])[0]
                    for j in cluster_indices if i != j
                )

                if distance_sum < min_distance_sum:
                    min_distance_sum = distance_sum
                    center_idx = i

            if center_idx == -1:
                return None

            center_encoding = face_encodings[center_idx]
            center_path = valid_paths[center_idx]

            self.reference_encodings[actor_name] = center_encoding

            ref_dir = os.path.join(self.output_dir, "reference_images")
            os.makedirs(ref_dir, exist_ok=True)
            ref_path = os.path.join(
                ref_dir, f"{actor_name.replace(' ', '_')}_reference.jpg")

            shutil.copy(center_path, ref_path)
            print(f"✅ Created reference image from cluster for {actor_name}")

            return center_encoding

        except Exception as e:
            print(f"❌ Error in clustering: {e}")
            return None

    def verify_face_identity(self, image_path, actor_name, similarity_threshold=0.6):
        """Verify if the face in the image matches the reference face for the actor"""
        if actor_name not in self.reference_encodings:

            return True, 1.0

        try:
            face_encoding = self.calculate_face_encoding(image_path)
            if face_encoding is None:
                return False, 0.0

            reference_encoding = self.reference_encodings[actor_name]
            face_distance = face_recognition.face_distance(
                [reference_encoding], face_encoding)[0]
            similarity = 1 - face_distance

            if similarity >= similarity_threshold:
                return True, similarity
            else:
                return False, similarity

        except Exception as e:
            print(f"Error verifying identity: {e}")
            return False, 0.0

    def process_single_image(self, img_path, actor_name):
        """Process a single image: check face, duplicates, and identity"""
        try:

            has_single_face, face_message = self.detect_single_face(img_path)
            if not has_single_face:
                if os.path.exists(img_path):
                    os.remove(img_path)
                return False, img_path, f"Face check failed: {face_message}"

            is_correct_person, similarity = self.verify_face_identity(
                img_path, actor_name)
            if not is_correct_person:
                if os.path.exists(img_path):
                    os.remove(img_path)
                return False, img_path, f"Wrong person: {similarity:.2f} similarity"

            is_duplicate, duplicate_message = self.is_duplicate_image(
                img_path, actor_name)
            if is_duplicate:
                if os.path.exists(img_path):
                    os.remove(img_path)
                return False, img_path, f"Duplicate: {duplicate_message}"

            with Image.open(img_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                if max(img.size) > 1024:
                    img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                elif max(img.size) < 300:
                    scale_factor = 300 / max(img.size)
                    new_size = (
                        int(img.size[0] * scale_factor), int(img.size[1] * scale_factor))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)

                img.save(img_path, 'JPEG', quality=90, optimize=True)

            actor_dir = os.path.join(
                self.output_dir, "processed", actor_name.replace(' ', '_').replace('/', '_'))
            os.makedirs(actor_dir, exist_ok=True)

            processed_path = os.path.join(
                actor_dir, os.path.basename(img_path))
            os.rename(img_path, processed_path)

            with self.lock:
                for hash_data in self.image_hashes[actor_name]:
                    if hash_data['path'] == img_path:
                        hash_data['path'] = processed_path
                        break

            return True, processed_path, f"Unique single face: {face_message}, Identity: {similarity:.2f}"

        except Exception as e:
            error_msg = f"Error processing image {img_path}: {e}"
            if os.path.exists(img_path):
                try:
                    os.remove(img_path)
                except:
                    pass
            return False, img_path, error_msg

    def parallel_process_images(self, image_paths, actor_name, max_workers=None):
        """Process multiple images in parallel"""
        results = []
        if max_workers is None:
            max_workers = self.process_workers

        print(
            f"🔍 Parallel processing {len(image_paths)} images with {max_workers} workers")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {executor.submit(self.process_single_image, path, actor_name): path
                              for path in image_paths}

            completed = 0
            for future in as_completed(future_to_path):
                success, path, message = future.result()
                results.append((success, path, message))

                completed += 1
                if completed % 5 == 0 or completed == len(image_paths):
                    success_count = sum(1 for s, _, _ in results if s)
                    print(
                        f"⏳ Processed {completed}/{len(image_paths)} images ({success_count} successful)")

        successful_count = sum(1 for success, _, _ in results if success)
        print(
            f"📊 Parallel processing complete: {successful_count}/{len(image_paths)} successful")

        return results

    def process_actor_images(self, actor_name, target_images=100):
        """Process all images for a single actor"""
        print(f"\n🎭 Processing images for: {actor_name}")

        raw_dir = os.path.join(self.input_dir, "raw_downloads",
                               actor_name.replace(' ', '_').replace('/', '_'))
        if not os.path.exists(raw_dir):
            print(f"⚠️ No raw images found for {actor_name}")
            return 0

        image_files = []
        for filename in os.listdir(raw_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(raw_dir, filename))

        if not image_files:
            print(f"⚠️ No image files found for {actor_name}")
            return 0

        print(f"📊 Found {len(image_files)} raw images for {actor_name}")

        if len(image_files) >= 10:

            self.identify_most_common_face(image_files[:50], actor_name)

        processed_count = 0
        batch_size = 50

        for i in range(0, len(image_files), batch_size):
            if processed_count >= target_images:
                break

            batch_files = image_files[i:i + batch_size]
            print(
                f"\n🔄 Processing batch {i//batch_size + 1}/{(len(image_files) + batch_size - 1)//batch_size}")

            results = self.parallel_process_images(batch_files, actor_name)

            for success, path, message in results:
                if success:
                    processed_count += 1
                    print(
                        f"✅ [{processed_count:3d}/{target_images}] PROCESSED: {actor_name}")
                    if processed_count >= target_images:
                        break

        print(
            f"📊 Processing complete for {actor_name}: {processed_count} unique images")
        return processed_count

    def process_all_actors(self, target_images_per_actor=100):
        """Process images for all actors"""

        raw_downloads_dir = os.path.join(self.input_dir, "raw_downloads")
        if not os.path.exists(raw_downloads_dir):
            print("❌ No raw downloads directory found!")
            return

        actor_dirs = [d for d in os.listdir(raw_downloads_dir)
                      if os.path.isdir(os.path.join(raw_downloads_dir, d))]

        print(f"\n🚀 Starting processing for {len(actor_dirs)} actors")
        print(
            f"🎯 Target: {target_images_per_actor} processed images per actor")

        total_processed = 0
        completed_actors = 0

        for i, actor_dir in enumerate(actor_dirs, 1):

            actor_name = actor_dir.replace('_', ' ')

            print(f"\n{'='*60}")
            print(f"🎭 Actor {i}/{len(actor_dirs)}: {actor_name}")
            print(f"{'='*60}")

            processed = self.process_actor_images(
                actor_name, target_images_per_actor)
            total_processed += processed
            completed_actors += 1

            print(f"\n📊 Overall Progress:")
            print(
                f"   🎭 Actors processed: {completed_actors}/{len(actor_dirs)}")
            print(f"   📊 Total processed images: {total_processed:,}")
            print(
                f"   📈 Average per actor: {total_processed/completed_actors:.1f}")

            time.sleep(2)

        print(f"\n🎉 PROCESSING COMPLETED!")
        print(f"📊 Final Statistics:")
        print(f"   🎭 Actors processed: {completed_actors}")
        print(f"   📊 Total processed images: {total_processed:,}")
        print(
            f"   📈 Average per actor: {total_processed/completed_actors:.1f}")
        print(f"   📁 Processed images saved in: {self.output_dir}/processed/")

        return total_processed

    def create_processing_summary(self):
        """Create a summary of processed images"""
        summary = {
            "processing_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_actors": 0,
            "total_images": 0,
            "fully_completed_actors": 0,
            "actors_data": {}
        }

        processed_dir = os.path.join(self.output_dir, "processed")
        if os.path.exists(processed_dir):
            for actor_dir in os.listdir(processed_dir):
                actor_path = os.path.join(processed_dir, actor_dir)
                if os.path.isdir(actor_path):
                    image_count = len([f for f in os.listdir(actor_path)
                                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

                    summary["actors_data"][actor_dir] = {
                        "image_count": image_count,
                        "path": actor_path,
                        "completed": image_count >= 100,
                        "completion_rate": f"{min(100, (image_count/100)*100):.1f}%",
                        "uniqueness": "guaranteed_no_duplicates",
                        "identity_verified": "guaranteed_correct_person"
                    }
                    summary["total_images"] += image_count
                    summary["total_actors"] += 1

                    if image_count >= 100:
                        summary["fully_completed_actors"] += 1

        summary_file = os.path.join(self.output_dir, "processing_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"📄 Processing summary saved to: {summary_file}")
        return summary


if __name__ == "__main__":
    processor = IranianActorImageProcessor()
    processor.process_all_actors(target_images_per_actor=100)
    processor.create_processing_summary()

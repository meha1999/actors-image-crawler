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
from collections import Counter


class IranianActorImageProcessor:
    def __init__(self, input_dir="iranian_actors_dataset", output_dir=None):
        self.input_dir = input_dir
        self.output_dir = output_dir or input_dir

        # Create main directories
        os.makedirs(f"{self.output_dir}/stage1_face_detection", exist_ok=True)
        os.makedirs(f"{self.output_dir}/stage2_face_separation", exist_ok=True)
        os.makedirs(f"{self.output_dir}/stage3_actor_identification", exist_ok=True)
        os.makedirs(f"{self.output_dir}/final_processed", exist_ok=True)
        
        # Create error directories for each stage
        os.makedirs(f"{self.output_dir}/errors/stage1_no_face", exist_ok=True)
        os.makedirs(f"{self.output_dir}/errors/stage1_detection_error", exist_ok=True)
        os.makedirs(f"{self.output_dir}/errors/stage2_separation_error", exist_ok=True)
        os.makedirs(f"{self.output_dir}/errors/stage3_clustering_error", exist_ok=True)
        os.makedirs(f"{self.output_dir}/errors/stage3_wrong_actor", exist_ok=True)
        os.makedirs(f"{self.output_dir}/errors/stage3_duplicates", exist_ok=True)

        # Initialize face detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Initialize caches for duplicate detection
        self.image_hashes = {}
        self.face_encodings_cache = {}

        # Thresholds
        self.HASH_SIMILARITY_THRESHOLD = 5
        self.SSIM_THRESHOLD = 0.85
        self.FACE_SIMILARITY_THRESHOLD = 0.6
        self.IDENTITY_THRESHOLD = 0.5
        self.MIN_FACE_SIZE = 60
        self.MAX_FACE_RATIO = 0.8

        # Threading
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

    def is_duplicate_image(self, image_path, actor_name, stored_hashes, stored_encodings):
        """Check if image is a duplicate using multiple methods"""
        try:
            # Calculate hashes for new image
            new_hashes = self.calculate_image_hash(image_path)
            if not new_hashes:
                return True, "Could not calculate hash"

            # Check hash similarity
            for stored_hash_data in stored_hashes:
                for hash_type in ['phash', 'dhash', 'whash', 'average']:
                    if hash_type in new_hashes and hash_type in stored_hash_data['hashes']:
                        hash1 = imagehash.hex_to_hash(new_hashes[hash_type])
                        hash2 = imagehash.hex_to_hash(stored_hash_data['hashes'][hash_type])
                        hamming_distance = hash1 - hash2

                        if hamming_distance <= self.HASH_SIMILARITY_THRESHOLD:
                            return True, f"Duplicate detected (hash {hash_type}): distance={hamming_distance}"

            # Check structural similarity
            for stored_hash_data in stored_hashes:
                stored_image_path = stored_hash_data['path']
                if os.path.exists(stored_image_path):
                    ssim_score = self.calculate_structural_similarity(image_path, stored_image_path)
                    if ssim_score >= self.SSIM_THRESHOLD:
                        return True, f"Duplicate detected (SSIM): score={ssim_score:.3f}"

            # Check face similarity
            new_face_encoding = self.calculate_face_encoding(image_path)
            if new_face_encoding is not None:
                for stored_encoding in stored_encodings:
                    face_distance = face_recognition.face_distance([stored_encoding], new_face_encoding)[0]
                    similarity = 1 - face_distance

                    if similarity >= self.FACE_SIMILARITY_THRESHOLD:
                        return True, f"Duplicate detected (face): similarity={similarity:.3f}"

            return False, "Unique image"

        except Exception as e:
            print(f"Error checking duplicate: {e}")
            return False, "Error in duplicate check"

    def detect_faces_in_image(self, image_path):
        """Detect all faces in an image using multiple methods"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None, "Could not read image"

            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces_info = []

            # Method 1: face_recognition library
            try:
                face_locations = face_recognition.face_locations(rgb_img, model="hog")
                for i, (top, right, bottom, left) in enumerate(face_locations):
                    face_width = right - left
                    face_height = bottom - top
                    
                    if face_width >= self.MIN_FACE_SIZE and face_height >= self.MIN_FACE_SIZE:
                        faces_info.append({
                            'method': 'face_recognition',
                            'bbox': (left, top, right, bottom),
                            'size': (face_width, face_height),
                            'confidence': 0.8
                        })
            except Exception as e:
                print(f"Face recognition library error: {e}")

            # Method 2: OpenCV Haar Cascade (as backup/additional detection)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces_cv = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, 
                minSize=(self.MIN_FACE_SIZE, self.MIN_FACE_SIZE),
                maxSize=(int(gray.shape[1]*self.MAX_FACE_RATIO), int(gray.shape[0]*self.MAX_FACE_RATIO))
            )

            # Add OpenCV detections if no face_recognition detections
            if len(faces_info) == 0:
                for (x, y, w, h) in faces_cv:
                    faces_info.append({
                        'method': 'opencv',
                        'bbox': (x, y, x+w, y+h),
                        'size': (w, h),
                        'confidence': 0.6
                    })

            return faces_info, f"Detected {len(faces_info)} faces"

        except Exception as e:
            return None, f"Error detecting faces: {e}"

    def extract_face_from_image(self, image_path, bbox, face_id, output_dir):
        """Extract a single face from an image"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None, "Could not read image"

            left, top, right, bottom = bbox
            
            # Add padding around face
            padding = 20
            height, width = img.shape[:2]
            
            left = max(0, left - padding)
            top = max(0, top - padding)
            right = min(width, right + padding)
            bottom = min(height, bottom + padding)

            # Extract face region
            face_img = img[top:bottom, left:right]
            
            if face_img.size == 0:
                return None, "Empty face region"

            # Generate output filename
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            face_filename = f"{base_name}_face_{face_id}.jpg"
            face_path = os.path.join(output_dir, face_filename)

            # Save face image
            cv2.imwrite(face_path, face_img)
            
            return face_path, f"Face extracted: {right-left}x{bottom-top}"

        except Exception as e:
            return None, f"Error extracting face: {e}"

    def stage1_face_detection(self, actor_name):
        """Stage 1: Detect images with faces and separate them"""
        print(f"\n🔍 STAGE 1: Face Detection for {actor_name}")
        
        # Input directory
        raw_dir = os.path.join(self.input_dir, "raw_downloads", 
                              actor_name.replace(' ', '_').replace('/', '_'))
        if not os.path.exists(raw_dir):
            print(f"⚠️ No raw images found for {actor_name}")
            return 0, 0, 0

        # Output directories
        stage1_dir = os.path.join(self.output_dir, "stage1_face_detection", 
                                 actor_name.replace(' ', '_').replace('/', '_'))
        no_face_dir = os.path.join(self.output_dir, "errors/stage1_no_face", 
                                  actor_name.replace(' ', '_').replace('/', '_'))
        error_dir = os.path.join(self.output_dir, "errors/stage1_detection_error", 
                                actor_name.replace(' ', '_').replace('/', '_'))
        
        os.makedirs(stage1_dir, exist_ok=True)
        os.makedirs(no_face_dir, exist_ok=True)
        os.makedirs(error_dir, exist_ok=True)

        # Get all image files
        image_files = []
        for filename in os.listdir(raw_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(raw_dir, filename))

        print(f"📊 Processing {len(image_files)} images for face detection")

        with_faces = 0
        no_faces = 0
        errors = 0

        for i, img_path in enumerate(image_files):
            try:
                faces_info, message = self.detect_faces_in_image(img_path)
                
                if faces_info is None:
                    # Detection error
                    shutil.copy(img_path, os.path.join(error_dir, os.path.basename(img_path)))
                    errors += 1
                    if i % 50 == 0:
                        print(f"❌ [{i+1}/{len(image_files)}] ERROR: {os.path.basename(img_path)}")
                
                elif len(faces_info) == 0:
                    # No faces detected
                    shutil.copy(img_path, os.path.join(no_face_dir, os.path.basename(img_path)))
                    no_faces += 1
                    if i % 50 == 0:
                        print(f"👤 [{i+1}/{len(image_files)}] NO FACE: {os.path.basename(img_path)}")
                
                else:
                    # Faces detected - copy to stage1
                    shutil.copy(img_path, os.path.join(stage1_dir, os.path.basename(img_path)))
                    with_faces += 1
                    if i % 50 == 0:
                        print(f"✅ [{i+1}/{len(image_files)}] FACES FOUND: {len(faces_info)} faces")

            except Exception as e:
                shutil.copy(img_path, os.path.join(error_dir, os.path.basename(img_path)))
                errors += 1
                print(f"❌ Error processing {img_path}: {e}")

        print(f"📊 Stage 1 Results for {actor_name}:")
        print(f"   ✅ Images with faces: {with_faces}")
        print(f"   👤 Images without faces: {no_faces}")
        print(f"   ❌ Processing errors: {errors}")

        return with_faces, no_faces, errors

    def stage2_face_separation(self, actor_name):
        """Stage 2: Separate multi-face images into individual face images"""
        print(f"\n✂️ STAGE 2: Face Separation for {actor_name}")
        
        # Input directory (from stage 1)
        stage1_dir = os.path.join(self.output_dir, "stage1_face_detection", 
                                 actor_name.replace(' ', '_').replace('/', '_'))
        if not os.path.exists(stage1_dir):
            print(f"⚠️ No stage 1 results found for {actor_name}")
            return 0, 0

        # Output directories
        stage2_dir = os.path.join(self.output_dir, "stage2_face_separation", 
                                 actor_name.replace(' ', '_').replace('/', '_'))
        error_dir = os.path.join(self.output_dir, "errors/stage2_separation_error", 
                                actor_name.replace(' ', '_').replace('/', '_'))
        
        os.makedirs(stage2_dir, exist_ok=True)
        os.makedirs(error_dir, exist_ok=True)

        # Get all image files from stage 1
        image_files = []
        for filename in os.listdir(stage1_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(stage1_dir, filename))

        print(f"📊 Processing {len(image_files)} images for face separation")

        total_faces_extracted = 0
        single_face_images = 0
        multi_face_images = 0
        errors = 0

        for i, img_path in enumerate(image_files):
            try:
                faces_info, message = self.detect_faces_in_image(img_path)
                
                if faces_info is None or len(faces_info) == 0:
                    # No faces or error - move to error folder
                    shutil.copy(img_path, os.path.join(error_dir, os.path.basename(img_path)))
                    errors += 1
                    continue

                if len(faces_info) == 1:
                    # Single face - just copy the original image
                    shutil.copy(img_path, os.path.join(stage2_dir, os.path.basename(img_path)))
                    single_face_images += 1
                    total_faces_extracted += 1
                    
                    if i % 50 == 0:
                        print(f"👤 [{i+1}/{len(image_files)}] SINGLE FACE: {os.path.basename(img_path)}")

                else:
                    # Multiple faces - extract each face separately
                    multi_face_images += 1
                    faces_extracted = 0
                    
                    for face_id, face_info in enumerate(faces_info):
                        face_path, extract_message = self.extract_face_from_image(
                            img_path, face_info['bbox'], face_id, stage2_dir)
                        
                        if face_path:
                            faces_extracted += 1
                            total_faces_extracted += 1
                    
                    if i % 20 == 0:
                        print(f"👥 [{i+1}/{len(image_files)}] MULTI FACE: {len(faces_info)} faces, {faces_extracted} extracted")

            except Exception as e:
                shutil.copy(img_path, os.path.join(error_dir, os.path.basename(img_path)))
                errors += 1
                print(f"❌ Error processing {img_path}: {e}")

        print(f"📊 Stage 2 Results for {actor_name}:")
        print(f"   👤 Single face images: {single_face_images}")
        print(f"   👥 Multi face images: {multi_face_images}")
        print(f"   ✂️ Total faces extracted: {total_faces_extracted}")
        print(f"   ❌ Processing errors: {errors}")

        return total_faces_extracted, errors

    def calculate_face_encoding(self, image_path):
        """Calculate face encoding for face similarity comparison"""
        try:
            image = face_recognition.load_image_file(image_path)
            face_encodings = face_recognition.face_encodings(image)

            if len(face_encodings) >= 1:
                return face_encodings[0]  # Return first face encoding
            else:
                return None

        except Exception as e:
            print(f"Error calculating face encoding for {image_path}: {e}")
            return None

    def stage3_actor_identification(self, actor_name):
        """Stage 3: Use clustering to identify the correct actor and keep ALL their images"""
        print(f"\n🎭 STAGE 3: Actor Identification for {actor_name}")
        print(f"🎯 Target: Save ALL good images (no limit)")
        
        # Input directory (from stage 2)
        stage2_dir = os.path.join(self.output_dir, "stage2_face_separation", 
                                 actor_name.replace(' ', '_').replace('/', '_'))
        if not os.path.exists(stage2_dir):
            print(f"⚠️ No stage 2 results found for {actor_name}")
            return 0

        # Output directories
        final_dir = os.path.join(self.output_dir, "final_processed", 
                                actor_name.replace(' ', '_').replace('/', '_'))
        clustering_error_dir = os.path.join(self.output_dir, "errors/stage3_clustering_error", 
                                           actor_name.replace(' ', '_').replace('/', '_'))
        wrong_actor_dir = os.path.join(self.output_dir, "errors/stage3_wrong_actor", 
                                      actor_name.replace(' ', '_').replace('/', '_'))
        duplicates_dir = os.path.join(self.output_dir, "errors/stage3_duplicates", 
                                     actor_name.replace(' ', '_').replace('/', '_'))
        
        os.makedirs(final_dir, exist_ok=True)
        os.makedirs(clustering_error_dir, exist_ok=True)
        os.makedirs(wrong_actor_dir, exist_ok=True)
        os.makedirs(duplicates_dir, exist_ok=True)

        # Get all face images from stage 2
        image_files = []
        for filename in os.listdir(stage2_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(stage2_dir, filename))

        print(f"📊 Processing {len(image_files)} face images for actor identification")

        if len(image_files) < 10:
            print(f"⚠️ Not enough images for clustering ({len(image_files)} < 10)")
            # Move all to clustering error
            for img_path in image_files:
                shutil.copy(img_path, os.path.join(clustering_error_dir, os.path.basename(img_path)))
            return 0

        # Step 1: Calculate face encodings for all images
        print("🧮 Calculating face encodings...")
        face_encodings = []
        valid_paths = []
        encoding_errors = 0

        for i, img_path in enumerate(image_files):
            try:
                encoding = self.calculate_face_encoding(img_path)
                if encoding is not None:
                    face_encodings.append(encoding)
                    valid_paths.append(img_path)
                else:
                    shutil.copy(img_path, os.path.join(clustering_error_dir, os.path.basename(img_path)))
                    encoding_errors += 1
                
                if i % 100 == 0:
                    print(f"   📊 Processed {i+1}/{len(image_files)} images")
            except Exception as e:
                shutil.copy(img_path, os.path.join(clustering_error_dir, os.path.basename(img_path)))
                encoding_errors += 1
                print(f"❌ Error encoding {img_path}: {e}")

        print(f"✅ Successfully encoded {len(face_encodings)} faces ({encoding_errors} errors)")

        if len(face_encodings) < 5:
            print(f"⚠️ Not enough valid encodings for clustering ({len(face_encodings)} < 5)")
            return 0

        # Step 2: Perform clustering
        print("🧩 Performing face clustering...")
        try:
            clustering = DBSCAN(eps=0.5, min_samples=3).fit(face_encodings)
            labels = clustering.labels_

            # Count cluster sizes
            label_counts = Counter(labels)
            if -1 in label_counts:
                del label_counts[-1]  # Remove noise cluster

            if not label_counts:
                print("❌ No valid clusters found")
                for img_path in valid_paths:
                    shutil.copy(img_path, os.path.join(clustering_error_dir, os.path.basename(img_path)))
                return 0

            # Find the largest cluster (assumed to be the main actor)
            largest_cluster_label = max(label_counts.items(), key=lambda x: x[1])[0]
            largest_cluster_size = label_counts[largest_cluster_label]

            print(f"🎯 Found largest cluster: {largest_cluster_size} images ({(largest_cluster_size/len(face_encodings))*100:.1f}%)")
            print(f"📊 Total clusters found: {len(label_counts)}")

            # Step 3: Calculate cluster center (reference face)
            cluster_indices = [i for i, label in enumerate(labels) if label == largest_cluster_label]
            
            if not cluster_indices:
                print("❌ No images in largest cluster")
                return 0

            # Find the face closest to cluster center
            min_distance_sum = float('inf')
            center_idx = -1

            for i in cluster_indices:
                distance_sum = sum(
                    face_recognition.face_distance([face_encodings[i]], face_encodings[j])[0]
                    for j in cluster_indices if i != j
                )
                if distance_sum < min_distance_sum:
                    min_distance_sum = distance_sum
                    center_idx = i

            if center_idx == -1:
                print("❌ Could not find cluster center")
                return 0

            reference_encoding = face_encodings[center_idx]
            reference_path = valid_paths[center_idx]

            print(f"✅ Reference face selected: {os.path.basename(reference_path)}")

            # Step 4: Filter ALL images based on similarity to reference
            print("🔍 Processing ALL images (no limit)...")
            
            correct_actor_count = 0
            wrong_actor_count = 0
            duplicate_count = 0
            
            # Initialize duplicate detection storage
            stored_hashes = []
            stored_encodings = []

            for i, (img_path, encoding) in enumerate(zip(valid_paths, face_encodings)):
                try:
                    # Calculate similarity to reference
                    face_distance = face_recognition.face_distance([reference_encoding], encoding)[0]
                    similarity = 1 - face_distance

                    if similarity >= self.IDENTITY_THRESHOLD:
                        # Check for duplicates
                        is_duplicate, duplicate_message = self.is_duplicate_image(
                            img_path, actor_name, stored_hashes, stored_encodings)
                        
                        if is_duplicate:
                            # This is a duplicate
                            shutil.copy(img_path, os.path.join(duplicates_dir, os.path.basename(img_path)))
                            duplicate_count += 1
                            if duplicate_count % 20 == 0:
                                print(f"🔄 Found {duplicate_count} duplicates")
                        else:
                            # This is the correct actor and unique
                            output_filename = f"{actor_name.replace(' ', '_')}_{correct_actor_count:04d}.jpg"
                            output_path = os.path.join(final_dir, output_filename)
                            
                            # Optimize and save image
                            self.optimize_and_save_image(img_path, output_path)
                            correct_actor_count += 1
                            
                            # Store hash and encoding for duplicate detection
                            new_hashes = self.calculate_image_hash(img_path)
                            if new_hashes:
                                stored_hashes.append({
                                    'hashes': new_hashes,
                                    'path': output_path
                                })
                            stored_encodings.append(encoding)
                            
                            if correct_actor_count % 50 == 0:
                                print(f"✅ Saved {correct_actor_count} unique images of {actor_name}")
                    else:
                        # Wrong actor or poor quality
                        shutil.copy(img_path, os.path.join(wrong_actor_dir, os.path.basename(img_path)))
                        wrong_actor_count += 1

                except Exception as e:
                    shutil.copy(img_path, os.path.join(clustering_error_dir, os.path.basename(img_path)))
                    print(f"❌ Error processing {img_path}: {e}")

            print(f"📊 Stage 3 Results for {actor_name}:")
            print(f"   ✅ Correct actor images (unique): {correct_actor_count}")
            print(f"   🔄 Duplicate images removed: {duplicate_count}")
            print(f"   ❌ Wrong actor images: {wrong_actor_count}")
            print(f"   🔧 Encoding errors: {encoding_errors}")
            print(f"   📈 Success rate: {(correct_actor_count/(len(valid_paths)-encoding_errors))*100:.1f}%")

            return correct_actor_count

        except Exception as e:
            print(f"❌ Clustering error: {e}")
            for img_path in valid_paths:
                shutil.copy(img_path, os.path.join(clustering_error_dir, os.path.basename(img_path)))
            return 0

    def optimize_and_save_image(self, input_path, output_path):
        """Optimize image size and quality before saving"""
        try:
            with Image.open(input_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # Resize if needed
                if max(img.size) > 1024:
                    img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                elif max(img.size) < 300:
                    scale_factor = 300 / max(img.size)
                    new_size = (int(img.size[0] * scale_factor), int(img.size[1] * scale_factor))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)

                img.save(output_path, 'JPEG', quality=90, optimize=True)
                
        except Exception as e:
            # If optimization fails, just copy the original
            shutil.copy(input_path, output_path)
            print(f"⚠️ Optimization failed, copied original: {e}")

    def process_actor_complete(self, actor_name):
        """Complete 3-stage processing for a single actor - saves ALL good images"""
        print(f"\n{'='*80}")
        print(f"🎬 PROCESSING ACTOR: {actor_name}")
        print(f"🎯 Target: Save ALL good images (no limit)")
        print(f"{'='*80}")

        start_time = time.time()

        # Stage 1: Face Detection
        with_faces, no_faces, stage1_errors = self.stage1_face_detection(actor_name)
        
        if with_faces == 0:
            print(f"❌ No faces found for {actor_name}, skipping remaining stages")
            return 0

        # Stage 2: Face Separation
        total_faces, stage2_errors = self.stage2_face_separation(actor_name)
        
        if total_faces == 0:
            print(f"❌ No faces extracted for {actor_name}, skipping stage 3")
            return 0

        # Stage 3: Actor Identification (saves ALL good images)
        final_count = self.stage3_actor_identification(actor_name)

        processing_time = time.time() - start_time
        
        print(f"\n📊 COMPLETE PROCESSING SUMMARY for {actor_name}:")
        print(f"   🕒 Processing time: {processing_time:.1f} seconds")
        print(f"   🔍 Stage 1 - Images with faces: {with_faces}")
        print(f"   ✂️ Stage 2 - Faces extracted: {total_faces}")
        print(f"   🎭 Stage 3 - Final unique actor images: {final_count}")
        print(f"   📈 Overall success rate: {(final_count/max(1, with_faces))*100:.1f}%")

        return final_count

    def process_all_actors(self):
        """Process all actors through the complete 3-stage pipeline - saves ALL good images"""
        # Get list of actors from raw downloads
        raw_downloads_dir = os.path.join(self.input_dir, "raw_downloads")
        if not os.path.exists(raw_downloads_dir):
            print("❌ No raw downloads directory found!")
            return

        actor_dirs = [d for d in os.listdir(raw_downloads_dir)
                      if os.path.isdir(os.path.join(raw_downloads_dir, d))]

        print(f"\n🚀 STARTING 3-STAGE PROCESSING PIPELINE")
        print(f"🎭 Actors to process: {len(actor_dirs)}")
        print(f"🎯 Target: Save ALL good images per actor (no limits)")
        print(f"\n📋 PROCESSING STAGES:")
        print(f"   1️⃣ Face Detection: Find images with faces")
        print(f"   2️⃣ Face Separation: Extract individual faces from multi-face images")
        print(f"   3️⃣ Actor Identification: Use clustering to keep ALL correct actor images")

        total_processed = 0
        completed_actors = 0
        pipeline_start_time = time.time()

        for i, actor_dir in enumerate(actor_dirs, 1):
            # Convert directory name back to actor name
            actor_name = actor_dir.replace('_', ' ')

            print(f"\n🎬 ACTOR {i}/{len(actor_dirs)}: {actor_name}")
            
            processed = self.process_actor_complete(actor_name)
            total_processed += processed
            completed_actors += 1

            # Progress update
            elapsed_time = time.time() - pipeline_start_time
            avg_time_per_actor = elapsed_time / completed_actors
            estimated_remaining = avg_time_per_actor * (len(actor_dirs) - completed_actors)

            print(f"\n📊 PIPELINE PROGRESS:")
            print(f"   🎭 Actors completed: {completed_actors}/{len(actor_dirs)} ({(completed_actors/len(actor_dirs))*100:.1f}%)")
            print(f"   📊 Total images processed: {total_processed:,}")
            print(f"   📈 Average per actor: {total_processed/completed_actors:.1f}")
            print(f"   ⏱️ Time elapsed: {elapsed_time/60:.1f} minutes")
            print(f"   ⏳ Estimated remaining: {estimated_remaining/60:.1f} minutes")

            time.sleep(1)  # Brief pause between actors

        total_time = time.time() - pipeline_start_time

        print(f"\n🎉 3-STAGE PROCESSING PIPELINE COMPLETED!")
        print(f"{'='*80}")
        print(f"📊 FINAL STATISTICS:")
        print(f"   🎭 Total actors processed: {completed_actors}")
        print(f"   📊 Total final images: {total_processed:,}")
        print(f"   📈 Average images per actor: {total_processed/completed_actors:.1f}")
        print(f"   ⏱️ Total processing time: {total_time/60:.1f} minutes")
        print(f"   📁 Final images location: {self.output_dir}/final_processed/")
        print(f"   🗂️ Error logs location: {self.output_dir}/errors/")

        # Create processing summary
        self.create_processing_summary()

        return total_processed

    def create_processing_summary(self):
        """Create a comprehensive summary of the 3-stage processing"""
        summary = {
            "processing_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "processing_mode": "save_all_good_images",
            "processing_stages": {
                "stage1": "Face Detection",
                "stage2": "Face Separation", 
                "stage3": "Actor Identification + Duplicate Removal"
            },
            "total_actors": 0,
            "total_final_images": 0,
            "actors_data": {},
            "error_summary": {
                "stage1_no_face": 0,
                "stage1_detection_error": 0,
                "stage2_separation_error": 0,
                "stage3_clustering_error": 0,
                "stage3_wrong_actor": 0,
                "stage3_duplicates": 0
            }
        }

        # Count final processed images
        final_dir = os.path.join(self.output_dir, "final_processed")
        if os.path.exists(final_dir):
            for actor_dir in os.listdir(final_dir):
                actor_path = os.path.join(final_dir, actor_dir)
                if os.path.isdir(actor_path):
                    image_count = len([f for f in os.listdir(actor_path)
                                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

                    summary["actors_data"][actor_dir] = {
                        "final_images": image_count,
                        "path": actor_path,
                        "processing_complete": True,
                        "quality": "high_quality_unique_single_actor_faces",
                        "duplicates_removed": True,
                        "identity_verified": True
                    }
                    summary["total_final_images"] += image_count
                    summary["total_actors"] += 1

        # Count errors in each category
        error_base_dir = os.path.join(self.output_dir, "errors")
        if os.path.exists(error_base_dir):
            for error_type in summary["error_summary"].keys():
                error_dir = os.path.join(error_base_dir, error_type)
                if os.path.exists(error_dir):
                    for actor_dir in os.listdir(error_dir):
                        actor_error_path = os.path.join(error_dir, actor_dir)
                        if os.path.isdir(actor_error_path):
                            error_count = len([f for f in os.listdir(actor_error_path)
                                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                            summary["error_summary"][error_type] += error_count

        summary_file = os.path.join(self.output_dir, "processing_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"📄 Processing summary saved to: {summary_file}")
        return summary


if __name__ == "__main__":
    processor = IranianActorImageProcessor()
    processor.process_all_actors()  # No limit parameter - saves ALL good images
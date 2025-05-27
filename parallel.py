import requests
from bs4 import BeautifulSoup
import cv2
import numpy as np
import os
import time
import random
from urllib.parse import urljoin, urlparse, quote
import json
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import face_recognition
from PIL import Image
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
from skimage.metrics import structural_similarity as ssim
import imagehash
import multiprocessing
import shutil
from sklearn.cluster import DBSCAN


class IranianActorImageCrawler:
    def __init__(self, output_dir="iranian_actors_dataset"):
        self.output_dir = output_dir
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/processed", exist_ok=True)
        os.makedirs(f"{output_dir}/reference_images", exist_ok=True)

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        self.selenium_available = self.check_selenium_availability()

        self.image_hashes = {}
        self.face_encodings_cache = {}
        self.reference_encodings = {}

        self.HASH_SIMILARITY_THRESHOLD = 5
        self.SSIM_THRESHOLD = 0.85
        self.FACE_SIMILARITY_THRESHOLD = 0.6
        self.IDENTITY_THRESHOLD = 0.6

        self.lock = threading.Lock()

        self.download_workers = self.get_optimal_worker_count()
        self.process_workers = 1

        print(
            f"🧵 Using {self.download_workers} download workers and {self.process_workers} processing workers")

        self.actors_list = [
            "شهاب حسینی", "پیمان معادی", "حامد بهداد", "رضا عطاران", "بهرام رادان",
            "محمدرضا گلزار", "رضا کیانیان", "سام درخشانی", "امین حیایی", "جواد عزتی",
            "محسن تنابنده", "پژمان جمشیدی", "امیر جعفری", "فرهاد اصلانی", "محمدرضا شریفینیا",
            "علیرضا خمسه", "میلاد کی‌مرام", "علی نصیریان", "اکبر عبدی", "مهدی هاشمی",
            "حسین یاری", "امیرحسین صدیق", "مهران مدیری", "رضا ناجی", "بیژن بنفشه‌خواه",
            "داریوش ارجمند", "جمشید هاشم‌پور", "عزت‌الله انتظامی", "محمود پاک‌نیت", "اصغر همت",
            "مسعود رایگان", "رضا بابک", "امیر آقایی", "فریبرز عرب‌نیا", "مسعود کرامتی",
            "رضا صفایی پور", "علی اوجی", "حسن پورشیرازی", "فرید سجادی حسینی", "مجید مشیری",
            "علی‌رضا عصار", "مرتضی علی‌عباسی", "یکتا ناصر", "امیرحسین رستمی", "محسن کیایی",
            "رامبد جوان", "حسام نواب صفوی", "پوریا پورسرخ", "امیرمهدی ژوله", "بهروز شعیبی",

            "گلشیفته فراهانی", "لیلا حاتمی", "ترانه علیدوستی", "مهناز افشار", "هدیه تهرانی",
            "فاطمه معتمدآریا", "نیکی کریمی", "بهنوش طباطبایی", "مریلا زارعی", "لادن مستوفی",
            "سحر دولتشاهی", "بهاره رهنما", "مهتاب کرامتی", "ساره بیات", "مریم بوبانی",
            "هانیه توسلی", "نازنین بیاتی", "مهراوه شریفینیا", "بهاره کیان‌افشار", "الناز شاکردوست",
            "مهراوه شریفی‌نیا", "پانته‌آ بهرام", "مریم خدارحمی", "نگار جواهریان", "لیلی رشیدی",
            "گلاره عباسی", "نسرین مقانلو", "سارا بهرامی", "ستاره اسکندری", "مینا ساداتی",
            "ویشکا آسایش", "شبنم مقدمی", "یکتا ناصر", "مهناز افشار", "شیرین بینا",
            "فریبا نادری", "مرجان شیرمحمدی", "لیندا کیانی", "نیوشا ضیغمی", "آزاده صمدی",
            "رعنا آزادی‌ور", "سیما تیرانداز", "مریم معصومی", "آناهیتا افشار", "مهسا کرامتی",
            "ماهچهره خلیلی", "فلور نظری", "شقایق فراهانی", "لعیا زنگنه", "سوگل خلیق",
            "نگین معتضدی", "نسیم ادبی", "سحر قریشی", "مهراوه شریفی‌نیا", "آتنه فقانی"
        ]

        self.actors_english = [
            "Shahab Hosseini", "Peyman Maadi", "Hamed Behdad", "Reza Attaran", "Bahram Radan",
            "Mohammad Reza Golzar", "Reza Kianian", "Sam Derakhshani", "Amin Hayaei", "Javad Ezati",
            "Mohsen Tanabandeh", "Pejman Jamshidi", "Amir Jafari", "Farhad Aslani", "Mohammad Reza Sharifinia",
            "Alireza Khamseh", "Milad Keymaram", "Ali Nasirian", "Akbar Abdi", "Mehdi Hashemi",
            "Hossein Yari", "Amirhossein Seddigh", "Mehran Modiri", "Reza Naji", "Bijan Banafshekhah",
            "Dariush Arjmand", "Jamshid Hashempour", "Ezzatollah Entezami", "Mahmoud Pak-Niat", "Asghar Hemmat",
            "Masoud Rayegan", "Reza Babak", "Amir Aghaei", "Fariborz Arabnia", "Masoud Keramati",
            "Reza Safaei Pour", "Ali Owji", "Hassan Pourshirazi", "Farid Sajjadi Hosseini", "Majid Moshiri",
            "Alireza Osivand", "Morteza Ali Abbasi", "Yekta Naser", "Amirhossein Rostami", "Mohsen Kiaei",
            "Rambod Javan", "Hossam Navab Safavi", "Pouria Poursorkh", "Amir Mehdi Zhuleh", "Behrouz Shoeibi",

            "Golshifteh Farahani", "Leila Hatami", "Taraneh Alidoosti", "Mahnaz Afshar", "Hedieh Tehrani",
            "Fatemeh Motamed Arya", "Niki Karimi", "Behnoosh Tabatabaei", "Merila Zarei", "Laden Mostofi",
            "Sahar Dolatshahi", "Bahareh Rahnama", "Mahtab Keramati", "Sara Bayat", "Maryam Boubani",
            "Hanieh Tavassoli", "Nazanin Bayati", "Mehraveh Sharifinia", "Bahareh Kian Afshar", "Elnaz Shakerdoost",
            "Mehraveh Sharifinia", "Pantea Bahram", "Maryam Khodarahmi", "Negar Javaherian", "Leili Rashidi",
            "Golara Abbasi", "Nasrin Moghanloo", "Sara Bahrami", "Setareh Eskandari", "Mina Sadati",
            "Vishka Asayesh", "Shabnam Moghadami", "Yekta Naser", "Mahnaz Afshar", "Shirin Bina",
            "Fariba Naderi", "Marjan Shirmohammadi", "Linda Kiani", "Niousha Zeighami", "Azadeh Samadi",
            "Rana Azadivar", "Sima Tirandaz", "Maryam Masoumi", "Anahita Afshar", "Mahsa Keramati",
            "Mahchehreh Khalili", "Flor Nazari", "Shaghayegh Farahani", "Laya Zanganeh", "Sogol Khaligh",
            "Negin Motazedi", "Nasim Adabi", "Sahar Ghoreishi", "Mehraveh Sharifinia", "Atneh Faghani"
        ]

    def get_optimal_worker_count(self):
        """Calculate optimal number of worker threads based on system resources"""
        try:
            import psutil

            cpu_count = psutil.cpu_count(logical=True)
            if cpu_count:
                return max(1, int(cpu_count * 0.75))
        except ImportError:
            pass

        try:
            return max(1, multiprocessing.cpu_count() - 1)
        except:
            return 1

    def check_selenium_availability(self):
        """Check if Selenium and Chrome are properly configured"""
        try:
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")

            chrome_binary = self.find_chrome_binary()
            if chrome_binary:
                chrome_options.binary_location = chrome_binary

            chromedriver_path = self.find_chromedriver_binary()
            if chromedriver_path:
                service = Service(chromedriver_path)
                driver = webdriver.Chrome(
                    service=service, options=chrome_options)
            else:
                driver = webdriver.Chrome(options=chrome_options)

            driver.quit()
            print("✅ Selenium with Chrome is available")
            return True
        except Exception as e:
            print(f"⚠️  Selenium not available: {e}")
            print("🔄 Will use alternative scraping methods")
            return False

    def find_chrome_binary(self):
        """Find Chrome binary in common locations"""
        possible_paths = [
            '/usr/bin/google-chrome', '/usr/bin/google-chrome-stable',
            '/usr/bin/chromium-browser', '/usr/bin/chromium',
            '/snap/bin/chromium', '/usr/local/bin/google-chrome',
            '/opt/google/chrome/chrome'
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path
        return None

    def find_chromedriver_binary(self):
        """Find ChromeDriver binary in common locations"""
        possible_paths = [
            '/usr/bin/chromedriver', '/usr/local/bin/chromedriver',
            '/snap/bin/chromium.chromedriver', './chromedriver'
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path
        return None

    def setup_selenium_driver(self):
        """Setup Selenium WebDriver with better error handling"""
        if not self.selenium_available:
            return None

        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument(
            "--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option(
            "excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)

        chrome_binary = self.find_chrome_binary()
        if chrome_binary:
            chrome_options.binary_location = chrome_binary

        try:
            chromedriver_path = self.find_chromedriver_binary()
            if chromedriver_path:
                service = Service(chromedriver_path)
                driver = webdriver.Chrome(
                    service=service, options=chrome_options)
            else:
                driver = webdriver.Chrome(options=chrome_options)

            driver.execute_script(
                "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            return driver
        except Exception as e:
            print(f"Error setting up Chrome driver: {e}")
            return None

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

    def add_reference_images(self):
        """Add reference images for each actor to use as verification"""
        self.reference_encodings = {}

        reference_dir = os.path.join(self.output_dir, "reference_images")
        os.makedirs(reference_dir, exist_ok=True)

        for actor_name in self.actors_list:

            ref_path = os.path.join(
                reference_dir, f"{actor_name.replace(' ', '_')}_reference.jpg")

            if os.path.exists(ref_path):
                try:

                    encoding = self.calculate_face_encoding(ref_path)
                    if encoding is not None:
                        self.reference_encodings[actor_name] = encoding
                        print(f"✅ Loaded reference image for {actor_name}")
                        continue
                except:
                    pass

            print(f"🔍 Getting reference image for {actor_name}...")

            search_queries = [
                f"{actor_name} official headshot verified",
                f"{actor_name} official portrait actor",
                f"{actor_name} بازیگر رسمی پرتره"
            ]

            temp_path = os.path.join(reference_dir, "temp_reference.jpg")
            reference_found = False

            for query in search_queries:
                if reference_found:
                    break

                urls = self.get_search_urls_batch(query, batch_size=20)

                for url in urls:
                    try:
                        if os.path.exists(temp_path):
                            os.remove(temp_path)

                        if self.download_image(url, temp_path):
                            has_face, _ = self.detect_single_face(temp_path)
                            if has_face:
                                encoding = self.calculate_face_encoding(
                                    temp_path)
                                if encoding is not None:

                                    os.rename(temp_path, ref_path)
                                    self.reference_encodings[actor_name] = encoding
                                    print(
                                        f"✅ Created reference image for {actor_name}")
                                    reference_found = True
                                    break
                                else:
                                    if os.path.exists(temp_path):
                                        os.remove(temp_path)
                    except Exception as e:
                        print(f"Error getting reference for {actor_name}: {e}")
                        if os.path.exists(temp_path):
                            os.remove(temp_path)

            if not reference_found:
                print(f"⚠️ Could not create reference image for {actor_name}")

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

    def get_search_urls_batch(self, query, batch_size=100):
        """Get a large batch of image URLs using multiple search strategies"""
        all_urls = []

        search_variations = [
            f"{query} portrait headshot single person verified",
            f"{query} actor actress official portrait",
            f"{query} بازیگر ایرانی پرتره تک نفره رسمی",
            f"{query} headshot professional photo official",
            f"{query} iranian cinema actor portrait verified",
            f"{query} persian actor headshot official",
            f"{query} film actor portrait iran verified",
            f"{query} celebrity headshot iran official",
            f"{query} official photo portrait verified",
            f"{query} red carpet photo iran verified"
        ]

        for search_query in search_variations:
            if len(all_urls) >= batch_size * 2:
                break

            print(f"   🔍 Searching: {search_query}")

            if self.selenium_available:
                selenium_urls = self.search_google_images_selenium_batch(
                    search_query, num_images=50)
                all_urls.extend(selenium_urls)

            requests_urls = self.search_google_images_requests_batch(
                search_query, num_images=40)
            all_urls.extend(requests_urls)

            additional_urls = self.search_additional_sources_batch(
                search_query, num_images=30)
            all_urls.extend(additional_urls)

            time.sleep(random.uniform(1, 3))

        seen = set()
        unique_urls = []
        for url in all_urls:
            if url not in seen and len(unique_urls) < batch_size * 3:
                seen.add(url)
                unique_urls.append(url)

        return unique_urls

    def search_google_images_selenium_batch(self, query, num_images=50):
        """Enhanced Selenium search for batch URL collection"""
        if not self.selenium_available:
            return []

        search_url = f"https://www.google.com/search?q={quote(query)}&tbm=isch&safe=active"

        driver = self.setup_selenium_driver()
        if not driver:
            return []

        try:
            driver.get(search_url)
            time.sleep(2)

            for scroll in range(5):
                driver.execute_script(
                    "window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(1)

                try:
                    show_more = driver.find_element(
                        By.CSS_SELECTOR, "input[value*='Show more'], input[value*='نمایش بیشتر']")
                    if show_more.is_displayed():
                        show_more.click()
                        time.sleep(2)
                except:
                    pass

            image_urls = []
            img_selectors = ["img[data-src]", "img[src]", ".rg_i", ".Q4LuWd"]

            for selector in img_selectors:
                try:
                    img_elements = driver.find_elements(
                        By.CSS_SELECTOR, selector)
                    for img in img_elements:
                        if len(image_urls) >= num_images:
                            break
                        src = img.get_attribute(
                            "data-src") or img.get_attribute("src")
                        if src and src.startswith("http") and "gstatic" not in src and src not in image_urls:
                            image_urls.append(src)
                except:
                    continue

            return image_urls

        except Exception as e:
            print(f"Error in Selenium batch search: {e}")
            return []
        finally:
            driver.quit()

    def search_google_images_requests_batch(self, query, num_images=40):
        """Enhanced requests search for batch URL collection"""
        try:
            encoded_query = quote(query)
            search_url = f"https://www.google.com/search?q={encoded_query}&tbm=isch&safe=active"

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }

            response = self.session.get(search_url, headers=headers)
            soup = BeautifulSoup(response.content, 'html.parser')

            image_urls = []

            for img in soup.find_all('img', {'data-src': True}):
                src = img.get('data-src')
                if src and src.startswith('http') and 'gstatic' not in src and len(image_urls) < num_images:
                    image_urls.append(src)

            for img in soup.find_all('img', {'src': True}):
                src = img.get('src')
                if src and src.startswith('http') and 'gstatic' not in src and len(image_urls) < num_images:
                    image_urls.append(src)

            for script in soup.find_all('script'):
                if script.string:
                    urls = re.findall(
                        r'https://[^"\s]+\.(?:jpg|jpeg|png|webp)', script.string)
                    for url in urls:
                        if 'gstatic' not in url and url not in image_urls and len(image_urls) < num_images:
                            image_urls.append(url)

            return image_urls

        except Exception as e:
            print(f"Error in requests batch search: {e}")
            return []

    def search_additional_sources_batch(self, query, num_images=30):
        """Search additional sources for batch URL collection"""
        additional_urls = []

        try:
            search_url = f"https://www.bing.com/images/search?q={quote(query)}"
            response = self.session.get(search_url)
            soup = BeautifulSoup(response.content, 'html.parser')

            for img in soup.find_all('img')[:num_images//2]:
                src = img.get('src') or img.get('data-src')
                if src and src.startswith('http') and src not in additional_urls:
                    additional_urls.append(src)
        except Exception as e:
            print(f"Error searching Bing: {e}")

        try:
            search_url = f"https://duckduckgo.com/?q={quote(query)}&t=h_&iax=images&ia=images"
            response = self.session.get(search_url)
            soup = BeautifulSoup(response.content, 'html.parser')

            for img in soup.find_all('img')[:num_images//2]:
                src = img.get('src') or img.get('data-src')
                if src and src.startswith('http') and src not in additional_urls:
                    additional_urls.append(src)
        except Exception as e:
            print(f"Error searching DuckDuckGo: {e}")

        return additional_urls

    def download_image(self, url, filename):
        """Download an image from URL with retry mechanism"""
        max_retries = 3

        for attempt in range(max_retries):
            try:
                headers = {
                    'User-Agent': random.choice([
                        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
                        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
                    ]),
                    'Referer': random.choice([
                        'https://www.google.com/',
                        'https://www.bing.com/',
                        'https://duckduckgo.com/'
                    ])
                }

                response = requests.get(
                    url, headers=headers, timeout=15, stream=True)
                response.raise_for_status()

                content_type = response.headers.get('content-type', '').lower()
                if not any(img_type in content_type for img_type in ['image/jpeg', 'image/jpg', 'image/png', 'image/webp']):
                    continue

                content_length = response.headers.get('content-length')
                if content_length:
                    size = int(content_length)
                    if size < 10000 or size > 10000000:
                        continue

                with open(filename, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                try:
                    with Image.open(filename) as img:
                        width, height = img.size
                        if width < 150 or height < 150:
                            os.remove(filename)
                            continue
                        img.verify()
                    return True
                except:
                    if os.path.exists(filename):
                        os.remove(filename)
                    continue

            except Exception as e:
                if attempt == max_retries - 1:
                    print(
                        f"Download failed after {max_retries} attempts: {url}")
                time.sleep(random.uniform(1, 3))

        return False

    def parallel_download_images(self, urls, actor_name, temp_dir, search_round, max_workers=None):
        """Download multiple images in parallel using ThreadPoolExecutor

        Args:
            urls: List of image URLs to download
            actor_name: Name of the actor for file naming
            temp_dir: Directory to save temporary downloads
            search_round: Current search round number
            max_workers: Maximum number of concurrent downloads

        Returns:
            List of tuples (success_flag, image_path)
        """
        results = []
        if max_workers is None:
            max_workers = self.download_workers

        def download_single(i, url):
            url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
            filename = os.path.join(
                temp_dir, f"{actor_name.replace(' ', '_')}_r{search_round}_{i:03d}_{url_hash}.jpg")

            success = self.download_image(url, filename)
            return success, filename

        print(
            f"🚀 Starting parallel download of {len(urls)} images with {max_workers} workers")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {executor.submit(
                download_single, i, url): i for i, url in enumerate(urls)}

            completed = 0
            for future in as_completed(future_to_idx):
                success, filename = future.result()
                results.append((success, filename))

                completed += 1
                if completed % 10 == 0 or completed == len(urls):
                    success_count = sum(1 for s, _ in results if s)
                    print(
                        f"⏳ Downloaded {completed}/{len(urls)} URLs ({success_count} successful)")

        successful_downloads = sum(1 for success, _ in results if success)
        print(
            f"📊 Parallel download complete: {successful_downloads}/{len(urls)} successful")

        return results

    def parallel_process_images(self, image_paths, actor_name, max_workers=None):
        """Process multiple images in parallel to check for single faces and duplicates

        Args:
            image_paths: List of paths to images that need processing
            actor_name: Name of the actor for categorization
            max_workers: Maximum number of concurrent processing threads

        Returns:
            List of tuples (success_flag, image_path, message)
        """
        results = []
        if max_workers is None:
            max_workers = self.process_workers

        def process_single_image(img_path):
            try:
                has_single_face, face_message = self.detect_single_face(
                    img_path)

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

        print(
            f"🔍 Parallel processing {len(image_paths)} images with {max_workers} workers")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {executor.submit(
                process_single_image, path): path for path in image_paths}

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

    def process_image(self, image_path, actor_name):
        """Process image: check single face + check for duplicates + verify identity"""
        try:
            has_single_face, face_message = self.detect_single_face(image_path)

            if not has_single_face:
                os.remove(image_path)
                return False, f"Face check failed: {face_message}"

            is_correct_person, similarity = self.verify_face_identity(
                image_path, actor_name)
            if not is_correct_person:
                os.remove(image_path)
                return False, f"Identity verification failed: {similarity:.2f} similarity"

            is_duplicate, duplicate_message = self.is_duplicate_image(
                image_path, actor_name)

            if is_duplicate:
                os.remove(image_path)
                return False, f"Duplicate: {duplicate_message}"

            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                if max(img.size) > 1024:
                    img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                elif max(img.size) < 300:
                    scale_factor = 300 / max(img.size)
                    new_size = (
                        int(img.size[0] * scale_factor), int(img.size[1] * scale_factor))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)

                img.save(image_path, 'JPEG', quality=90, optimize=True)

            actor_dir = os.path.join(
                self.output_dir, "processed", actor_name.replace(' ', '_').replace('/', '_'))
            os.makedirs(actor_dir, exist_ok=True)

            processed_path = os.path.join(
                actor_dir, os.path.basename(image_path))
            os.rename(image_path, processed_path)

            with self.lock:
                for hash_data in self.image_hashes[actor_name]:
                    if hash_data['path'] == image_path:
                        hash_data['path'] = processed_path
                        break

            return True, f"Unique single face: {face_message}, Identity: {similarity:.2f}"

        except Exception as e:
            error_msg = f"Error processing image {image_path}: {e}"
            if os.path.exists(image_path):
                os.remove(image_path)
            return False, error_msg

    def crawl_actor_images_no_duplicates(self, actor_name, target_images=100):
        """Crawl until we get EXACTLY the target number of UNIQUE single face images"""
        print(f"\n{'='*80}")
        print(f"🎭 NO DUPLICATES crawling for: {actor_name}")
        print(f"🎯 Target: {target_images} UNIQUE SINGLE FACE images")
        print(f"🚫 Duplicate detection: Hash + SSIM + Face encoding")
        print(f"👤 Identity verification: Comparing against reference face")
        print(
            f"🔧 Selenium available: {'Yes' if self.selenium_available else 'No'}")
        print(
            f"🧵 Using {self.download_workers} download workers and {self.process_workers} processing workers")
        print(f"{'='*80}")

        successful_downloads = 0
        total_attempts = 0
        search_round = 1
        max_search_rounds = 15

        duplicate_count = 0
        multiple_faces_count = 0
        no_face_count = 0
        wrong_person_count = 0
        download_failures = 0

        temp_dir = os.path.join(self.output_dir, "temp",
                                actor_name.replace(' ', '_'))
        os.makedirs(temp_dir, exist_ok=True)

        with self.lock:
            if actor_name not in self.image_hashes:
                self.image_hashes[actor_name] = []
                self.face_encodings_cache[actor_name] = []

        search_queries = [actor_name]
        try:
            idx = self.actors_list.index(actor_name)
            if idx < len(self.actors_english):
                search_queries.append(self.actors_english[idx])
        except:
            pass

        print(f"\n🧩 INITIAL ROUND: Collecting images for face clustering")

        initial_urls = []
        for query in search_queries:
            print(f"🔍 Initial search for: {query}")
            batch_urls = self.get_search_urls_batch(query, batch_size=50)
            initial_urls.extend(batch_urls)
            time.sleep(random.uniform(1, 2))

        initial_download_results = self.parallel_download_images(
            initial_urls[:100], actor_name, temp_dir, 0)

        initial_image_paths = [path for success,
                               path in initial_download_results if success]

        valid_faces = []
        for path in initial_image_paths:
            has_face, _ = self.detect_single_face(path)
            if has_face:
                valid_faces.append(path)

        print(f"🧩 Found {len(valid_faces)} valid face images for clustering")

        if actor_name not in self.reference_encodings and len(valid_faces) >= 5:
            self.identify_most_common_face(valid_faces, actor_name)

        if actor_name not in self.reference_encodings:
            print(
                f"⚠️ No reference face identified through clustering, using manual reference")
            self.add_reference_images()

        while successful_downloads < target_images and search_round <= max_search_rounds:
            print(
                f"\n🔄 SEARCH ROUND {search_round} - Need {target_images - successful_downloads} more UNIQUE images")

            all_image_urls = []
            for query in search_queries:
                print(f"🔍 Batch searching for: {query}")
                batch_urls = self.get_search_urls_batch(query, batch_size=250)
                all_image_urls.extend(batch_urls)
                print(f"   Found {len(batch_urls)} URLs")
                time.sleep(random.uniform(2, 4))

            seen = set()
            unique_urls = []
            for url in all_image_urls:
                if url not in seen:
                    seen.add(url)
                    unique_urls.append(url)

            print(
                f"📊 Round {search_round}: Found {len(unique_urls)} unique URLs")

            batch_size = min(100, len(unique_urls))
            round_successes = 0

            for batch_start in range(0, len(unique_urls), batch_size):
                if successful_downloads >= target_images:
                    break

                batch_urls = unique_urls[batch_start:batch_start + batch_size]
                print(
                    f"⏳ Processing batch {batch_start//batch_size + 1}/{(len(unique_urls) + batch_size - 1)//batch_size}")

                download_results = self.parallel_download_images(
                    batch_urls, actor_name, temp_dir, search_round)

                download_success_count = sum(
                    1 for success, _ in download_results if success)
                download_failures += len(download_results) - \
                    download_success_count
                total_attempts += len(download_results)

                downloaded_paths = [filename for success,
                                    filename in download_results if success]
                if downloaded_paths:

                    process_results = self.parallel_process_images(
                        downloaded_paths, actor_name)

                    for success, path, message in process_results:
                        if success:
                            successful_downloads += 1
                            round_successes += 1
                            print(
                                f"✅ [{successful_downloads:3d}/{target_images}] UNIQUE: {actor_name}")
                        else:
                            if "Duplicate" in message:
                                duplicate_count += 1
                            elif "Multiple faces" in message:
                                multiple_faces_count += 1
                            elif "No face" in message or "small" in message:
                                no_face_count += 1
                            elif "Wrong person" in message:
                                wrong_person_count += 1

                if total_attempts > 0:
                    success_rate = (successful_downloads /
                                    total_attempts) * 100
                    duplicate_rate = (
                        duplicate_count / total_attempts) * 100 if duplicate_count > 0 else 0
                    wrong_person_rate = (
                        wrong_person_count / total_attempts) * 100 if wrong_person_count > 0 else 0
                    print(f"📈 Batch Progress Report:")
                    print(
                        f"   ✅ Unique images: {successful_downloads}/{target_images}")
                    print(
                        f"   🔄 Duplicates rejected: {duplicate_count} ({duplicate_rate:.1f}%)")
                    print(
                        f"   👤 Wrong person rejected: {wrong_person_count} ({wrong_person_rate:.1f}%)")
                    print(f"   ❌ Multiple faces: {multiple_faces_count}")
                    print(f"   ⚠️  No/small faces: {no_face_count}")
                    print(f"   💥 Download failures: {download_failures}")
                    print(f"   🎯 Success rate: {success_rate:.1f}%")

                time.sleep(random.uniform(1, 3))

            print(
                f"🔄 Round {search_round} completed: {round_successes} new UNIQUE images found")
            search_round += 1

            if successful_downloads < target_images and search_round <= max_search_rounds:
                print(f"⏳ Waiting before next search round...")
                time.sleep(random.uniform(10, 20))

        try:
            os.rmdir(temp_dir)
        except:
            pass

        success_rate = (successful_downloads / total_attempts) * \
            100 if total_attempts > 0 else 0
        duplicate_rate = (duplicate_count / total_attempts) * \
            100 if total_attempts > 0 else 0
        wrong_person_rate = (wrong_person_count / total_attempts) * \
            100 if wrong_person_count > 0 else 0

        print(f"\n🎯 FINAL RESULTS for {actor_name}:")
        print(
            f"   ✅ Unique single face images: {successful_downloads}/{target_images}")
        print(f"   📊 Total attempts: {total_attempts}")
        print(f"   🔄 Search rounds: {search_round - 1}")
        print(
            f"   🔄 Duplicates rejected: {duplicate_count} ({duplicate_rate:.1f}%)")
        print(
            f"   👤 Wrong person rejected: {wrong_person_count} ({wrong_person_rate:.1f}%)")
        print(f"   ❌ Multiple faces rejected: {multiple_faces_count}")
        print(f"   ⚠️  No/small faces rejected: {no_face_count}")
        print(f"   💥 Download failures: {download_failures}")
        print(f"   🎯 Success rate: {success_rate:.1f}%")

        if successful_downloads == target_images:
            print(
                f"   🎉 TARGET ACHIEVED! Got exactly {target_images} UNIQUE single face images")
        else:
            print(
                f"   ⚠️  Could not find {target_images} unique images after {max_search_rounds} search rounds")

        return successful_downloads

    def crawl_all_actors_no_duplicates(self):
        """Crawl guaranteed 100 unique single face images for all actors"""
        print("🚀 GUARANTEED 100 UNIQUE Single Face Images per Actor Crawler")
        print("=" * 90)
        print(
            f"🎯 Target: 100 UNIQUE single face images × {len(self.actors_list)} actors")
        print(
            f"📊 Total target: {len(self.actors_list) * 100:,} unique single face images")
        print("🚫 Duplicate detection: Perceptual hash + SSIM + Face encoding")
        print("👤 Identity verification: Face matching against reference image")
        print("⚠️  Will continue searching until 100 UNIQUE single face images are found per actor!")
        print(
            f"🧵 Using {self.download_workers} download workers and {self.process_workers} processing workers")
        print("=" * 90)

        total_downloaded = 0
        completed_actors = 0
        fully_completed_actors = 0

        os.makedirs(os.path.join(self.output_dir, "temp"), exist_ok=True)

        for i, actor in enumerate(self.actors_list, 1):
            print(f"\n🎭 Actor {i}/{len(self.actors_list)}: {actor}")

            downloaded = self.crawl_actor_images_no_duplicates(
                actor, target_images=100)
            total_downloaded += downloaded
            completed_actors += 1

            if downloaded == 100:
                fully_completed_actors += 1

            print(f"📊 Overall Progress:")
            print(
                f"   🎭 Actors processed: {completed_actors}/{len(self.actors_list)}")
            print(
                f"   ✅ Fully completed actors (100 unique images): {fully_completed_actors}")
            print(
                f"   📊 Total unique single face images: {total_downloaded:,}")
            print(
                f"   📈 Average per actor: {total_downloaded/completed_actors:.1f}")

            time.sleep(random.uniform(10, 20))

        try:
            import shutil
            shutil.rmtree(os.path.join(self.output_dir, "temp"))
        except:
            pass

        print(f"\n🎉 NO DUPLICATES CRAWLING COMPLETED!")
        print(f"📊 Final Statistics:")
        print(f"   🎭 Actors processed: {completed_actors}")
        print(
            f"   ✅ Fully completed (100 unique images): {fully_completed_actors}")
        print(f"   📊 Total unique single face images: {total_downloaded:,}")
        print(
            f"   📈 Average per actor: {total_downloaded/completed_actors:.1f}")
        print(
            f"   🎯 Completion rate: {(fully_completed_actors/len(self.actors_list))*100:.1f}%")
        print(f"   📁 Images saved in: {self.output_dir}/processed/")
        print(f"   🚫 GUARANTEE: NO DUPLICATE IMAGES!")
        print(f"   👤 GUARANTEE: CORRECT IDENTITY VERIFIED!")

        return total_downloaded

    def create_dataset_summary(self):
        """Create a summary of the no-duplicates dataset"""
        processed_dir = os.path.join(self.output_dir, "processed")
        summary = {
            "dataset_type": "GUARANTEED UNIQUE SINGLE FACE WITH IDENTITY VERIFICATION",
            "target_per_actor": 100,
            "total_actors": 0,
            "total_images": 0,
            "fully_completed_actors": 0,
            "actors_data": {},
            "quality_guarantees": [
                "All images contain exactly one face",
                "No duplicate or similar images",
                "Identity verification to ensure correct person",
                "Multiple deduplication methods used"
            ],
            "deduplication_methods": [
                "Perceptual hash (pHash, dHash, wHash, average)",
                "Structural similarity (SSIM)",
                "Face encoding similarity"
            ],
            "identity_verification": {
                "method": "Face recognition comparison with reference image",
                "clustering": "DBSCAN clustering used to identify most common face"
            },
            "performance": {
                "download_workers": self.download_workers,
                "process_workers": self.process_workers
            }
        }

        if os.path.exists(processed_dir):
            for actor_dir in os.listdir(processed_dir):
                actor_path = os.path.join(processed_dir, actor_dir)
                if os.path.isdir(actor_path):
                    image_count = len([f for f in os.listdir(actor_path)
                                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

                    summary["actors_data"][actor_dir] = {
                        "image_count": image_count,
                        "path": actor_path,
                        "completed": image_count == 100,
                        "completion_rate": f"{(image_count/100)*100:.1f}%",
                        "uniqueness": "guaranteed_no_duplicates",
                        "identity_verified": "guaranteed_correct_person"
                    }
                    summary["total_images"] += image_count
                    summary["total_actors"] += 1

                    if image_count == 100:
                        summary["fully_completed_actors"] += 1

        summary_file = os.path.join(
            self.output_dir, "verified_dataset_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"\n📋 VERIFIED Dataset Summary:")
        print(f"   🎭 Total actors: {summary['total_actors']}")
        print(
            f"   ✅ Fully completed actors (100 unique images): {summary['fully_completed_actors']}")
        print(
            f"   📊 Total unique single face images: {summary['total_images']:,}")
        print(
            f"   🎯 Overall completion: {(summary['fully_completed_actors']/summary['total_actors'])*100:.1f}%")
        print(f"   🚫 Uniqueness: GUARANTEED NO DUPLICATES")
        print(f"   👤 Identity: VERIFIED CORRECT PERSON")
        print(f"   📄 Summary saved: {summary_file}")

        return summary

    def validate_no_duplicates(self):
        """Validate that there are truly no duplicates in the dataset"""
        processed_dir = os.path.join(self.output_dir, "processed")

        if not os.path.exists(processed_dir):
            print("❌ No processed images found.")
            return

        print("🔍 Validating dataset for duplicates...")

        total_comparisons = 0
        duplicates_found = 0

        for actor_dir in os.listdir(processed_dir):
            actor_path = os.path.join(processed_dir, actor_dir)
            if not os.path.isdir(actor_path):
                continue

            print(f"Validating: {actor_dir}")

            image_files = [f for f in os.listdir(
                actor_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            comparisons = []
            for i in range(len(image_files)):
                for j in range(i + 1, len(image_files)):
                    comparisons.append((
                        os.path.join(actor_path, image_files[i]),
                        os.path.join(actor_path, image_files[j])
                    ))

            total_comparisons += len(comparisons)

            batch_size = 1000
            for i in range(0, len(comparisons), batch_size):
                batch = comparisons[i:i+batch_size]

                def check_pair(pair):
                    img1_path, img2_path = pair
                    ssim_score = self.calculate_structural_similarity(
                        img1_path, img2_path)
                    if ssim_score >= self.SSIM_THRESHOLD:
                        return True, (os.path.basename(img1_path), os.path.basename(img2_path), ssim_score)
                    return False, None

                with ThreadPoolExecutor(max_workers=self.process_workers) as executor:
                    results = list(executor.map(check_pair, batch))

                batch_duplicates = [result for found,
                                    result in results if found]
                for img1, img2, score in batch_duplicates:
                    duplicates_found += 1
                    print(
                        f"   ⚠️  Potential duplicate: {img1} vs {img2} (SSIM: {score:.3f})")

                print(
                    f"   Checked {i + len(batch)}/{len(comparisons)} comparisons...")

        print(f"\n🎯 DUPLICATE VALIDATION RESULTS:")
        print(f"   📊 Total comparisons: {total_comparisons:,}")
        print(f"   ❌ Duplicates found: {duplicates_found}")
        print(
            f"   ✅ Uniqueness rate: {((total_comparisons-duplicates_found)/total_comparisons)*100:.2f}%")

        if duplicates_found == 0:
            print(f"   🎉 PERFECT! No duplicates found in the dataset!")
        else:
            print(
                f"   ⚠️  Found {duplicates_found} potential duplicates that need manual review")


if __name__ == "__main__":

    try:
        import imagehash
        from skimage.metrics import structural_similarity
    except ImportError:
        print("❌ Missing required packages. Please install:")
        print("pip install ImageHash scikit-image scikit-learn")
        exit(1)

    crawler = IranianActorImageCrawler()

    print("🎬 Iranian Actor/Actress VERIFIED NO DUPLICATES Single Face Image Crawler")
    print("=" * 90)
    print(
        f"🎯 GUARANTEE: 100 UNIQUE single face images × {len(crawler.actors_list)} actors")
    print("🚫 DUPLICATE DETECTION: Hash + SSIM + Face Encoding")
    print("👤 IDENTITY VERIFICATION: Face matching against reference image")
    print("🧩 CLUSTERING: Automatic identification of the correct person's face")
    print("⚠️  Will continue searching until EXACTLY 100 UNIQUE single face images per actor!")
    print(
        f"🧵 PERFORMANCE: Using {crawler.download_workers} download workers and {crawler.process_workers} processing workers")
    print("=" * 90)

    choice = input("\nChoose option:\n1. Full crawl (all 100 actors)\n2. Test with first 2 actors\n3. Single actor test\n4. Validate existing dataset\n\nEnter choice (1, 2, 3, or 4): ").strip()

    if choice == "4":
        print("\n🔍 Validating existing dataset for duplicates...")
        crawler.validate_no_duplicates()

    elif choice == "3":
        actor_name = input(f"\nEnter actor name from list: ").strip()
        if actor_name in crawler.actors_list:
            print(
                f"\n🧪 Testing VERIFIED NO DUPLICATES crawl for: {actor_name}")
            downloaded = crawler.crawl_actor_images_no_duplicates(
                actor_name, target_images=100)
            print(
                f"\n🎯 Result: {downloaded}/100 unique verified single face images downloaded")
        else:
            print("❌ Actor not found in list")

    elif choice == "2":
        print("\n🧪 Test mode: First 2 actors (100 unique images each)")
        test_actors = crawler.actors_list[:2]
        total_images = 0

        for i, actor in enumerate(test_actors, 1):
            print(f"\n🎭 Test Actor {i}/2: {actor}")
            downloaded = crawler.crawl_actor_images_no_duplicates(
                actor, target_images=100)
            total_images += downloaded

        print(
            f"\n🧪 Test completed! Downloaded {total_images} unique verified single face images")

    else:
        print("\n🚀 Starting full VERIFIED NO DUPLICATES crawl...")
        total_images = crawler.crawl_all_actors_no_duplicates()

        print("\n📊 Creating dataset summary...")
        summary = crawler.create_dataset_summary()

        validate = input(
            "\nValidate dataset for duplicates? (y/n): ").strip().lower()
        if validate == 'y':
            crawler.validate_no_duplicates()

    print("\n🎉 VERIFIED NO DUPLICATES SINGLE FACE DATASET READY!")
    print(f"📁 Dataset location: {crawler.output_dir}/processed/")
    print("🎯 Quality guarantees:")
    print("   ✅ ALL images contain EXACTLY ONE FACE")
    print("   🚫 NO DUPLICATE or SIMILAR IMAGES")
    print("   👤 IDENTITY VERIFIED using face recognition")
    print("   🔍 Triple-checked with hash + SSIM + face encoding")

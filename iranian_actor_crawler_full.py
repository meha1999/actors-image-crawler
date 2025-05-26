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
import asyncio
import aiohttp
import aiofiles
from multiprocessing import Pool, cpu_count
import pickle
import sqlite3
from functools import lru_cache
import logging

# Disable verbose logging
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('selenium').setLevel(logging.WARNING)


class FastIranianActorImageCrawler:
    def __init__(self, output_dir="iranian_actors_dataset"):
        self.output_dir = output_dir
        self.max_workers = cpu_count()  # Aggressive threading
        self.batch_size = 500  # Larger batches

        # Setup directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/processed", exist_ok=True)
        os.makedirs(f"{output_dir}/cache", exist_ok=True)

        # Initialize face detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Cache database for faster duplicate checking
        self.cache_db = os.path.join(output_dir, "cache", "image_cache.db")
        self.init_cache_db()

        # Selenium setup
        self.selenium_available = self.check_selenium_availability()
        self.selenium_pool = []  # Pool of selenium drivers

        # Thresholds (slightly relaxed for speed)
        self.HASH_SIMILARITY_THRESHOLD = 6  # Slightly less strict
        self.SSIM_THRESHOLD = 0.80  # Slightly less strict
        self.FACE_SIMILARITY_THRESHOLD = 0.75  # Slightly less strict

        # Session pool for requests
        self.session_pool = [self.create_session() for _ in range(8)]

        # Actors list (same as before)
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

    def init_cache_db(self):
        """Initialize SQLite database for caching"""
        conn = sqlite3.connect(self.cache_db)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS image_hashes (
                id INTEGER PRIMARY KEY,
                actor_name TEXT,
                image_path TEXT,
                phash TEXT,
                dhash TEXT,
                whash TEXT,
                average_hash TEXT,
                face_encoding BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.execute(
            'CREATE INDEX IF NOT EXISTS idx_actor ON image_hashes(actor_name)')
        conn.execute(
            'CREATE INDEX IF NOT EXISTS idx_hashes ON image_hashes(phash, dhash, whash, average_hash)')
        conn.commit()
        conn.close()

    def create_session(self):
        """Create optimized requests session"""
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        # Connection pooling
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=2
        )
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session

    def check_selenium_availability(self):
        """Quick selenium check"""
        try:
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            # Don't load images in selenium
            chrome_options.add_argument("--disable-images")
            chrome_options.add_argument(
                "--disable-javascript")  # Faster loading

            driver = webdriver.Chrome(options=chrome_options)
            driver.quit()
            print("✅ Selenium available")
            return True
        except Exception as e:
            print(f"⚠️ Selenium not available: {e}")
            return False

    def setup_selenium_driver_fast(self):
        """Setup ultra-fast selenium driver"""
        if not self.selenium_available:
            return None

        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--disable-images")
        chrome_options.add_argument("--disable-javascript")
        chrome_options.add_argument("--disable-plugins")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-web-security")
        chrome_options.add_argument("--aggressive-cache-discard")
        chrome_options.add_argument("--memory-pressure-off")
        chrome_options.add_argument("--max_old_space_size=4096")

        try:
            driver = webdriver.Chrome(options=chrome_options)
            driver.set_page_load_timeout(10)  # Fast timeout
            return driver
        except Exception as e:
            print(f"Error setting up fast driver: {e}")
            return None

    @lru_cache(maxsize=1000)
    def calculate_image_hash_cached(self, image_path_hash):
        """Cached hash calculation"""
        # This is a placeholder - actual implementation would need the image path
        pass

    def calculate_image_hash_fast(self, image_path):
        """Fast hash calculation with caching"""
        try:
            # Check if already in database
            conn = sqlite3.connect(self.cache_db)
            cursor = conn.execute(
                'SELECT phash, dhash, whash, average_hash FROM image_hashes WHERE image_path = ?',
                (image_path,)
            )
            result = cursor.fetchone()
            conn.close()

            if result:
                return {
                    'phash': result[0],
                    'dhash': result[1],
                    'whash': result[2],
                    'average': result[3]
                }

            # Calculate new hashes
            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # Resize for faster processing
                img.thumbnail((256, 256), Image.Resampling.LANCZOS)

                hashes = {
                    # Smaller hash size
                    'phash': str(imagehash.phash(img, hash_size=8)),
                    'dhash': str(imagehash.dhash(img, hash_size=8)),
                    'whash': str(imagehash.whash(img, hash_size=8)),
                    'average': str(imagehash.average_hash(img, hash_size=8))
                }

                return hashes
        except Exception as e:
            print(f"Error calculating hash: {e}")
            return None

    def is_duplicate_fast(self, image_path, actor_name):
        """Ultra-fast duplicate detection using database"""
        try:
            new_hashes = self.calculate_image_hash_fast(image_path)
            if not new_hashes:
                return True, "Could not calculate hash"

            conn = sqlite3.connect(self.cache_db)

            # Check against existing hashes for this actor
            cursor = conn.execute(
                'SELECT phash, dhash, whash, average_hash, image_path FROM image_hashes WHERE actor_name = ?',
                (actor_name,)
            )

            for row in cursor.fetchall():
                stored_hashes = {
                    'phash': row[0],
                    'dhash': row[1],
                    'whash': row[2],
                    'average': row[3]
                }

                # Quick hash comparison
                for hash_type in ['phash', 'dhash']:  # Only check most reliable hashes
                    if hash_type in new_hashes and hash_type in stored_hashes:
                        try:
                            hash1 = imagehash.hex_to_hash(
                                new_hashes[hash_type])
                            hash2 = imagehash.hex_to_hash(
                                stored_hashes[hash_type])
                            distance = hash1 - hash2

                            if distance <= self.HASH_SIMILARITY_THRESHOLD:
                                conn.close()
                                return True, f"Duplicate (hash {hash_type}): distance={distance}"
                        except:
                            continue

            # Store new hash
            conn.execute(
                'INSERT INTO image_hashes (actor_name, image_path, phash, dhash, whash, average_hash) VALUES (?, ?, ?, ?, ?, ?)',
                (actor_name, image_path, new_hashes['phash'], new_hashes['dhash'],
                 new_hashes['whash'], new_hashes['average'])
            )
            conn.commit()
            conn.close()

            return False, "Unique image"

        except Exception as e:
            print(f"Error in fast duplicate check: {e}")
            return False, "Error in duplicate check"

    def detect_face_fast(self, image_path):
        """Ultra-fast face detection"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return False, "Could not read image"

            # Resize for faster processing
            height, width = img.shape[:2]
            if max(height, width) > 800:
                scale = 800 / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                img = cv2.resize(img, (new_width, new_height))

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Fast face detection with relaxed parameters
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.2,  # Faster but less accurate
                minNeighbors=3,   # Less strict
                minSize=(50, 50),  # Smaller minimum
                maxSize=(int(gray.shape[1]*0.8), int(gray.shape[0]*0.8))
            )

            if len(faces) == 1:
                x, y, w, h = faces[0]
                if w >= 50 and h >= 50:
                    return True, f"Single face: {w}x{h}"

            return False, f"Face count: {len(faces)}"

        except Exception as e:
            return False, f"Error: {e}"

    async def download_image_async(self, session, url, filename):
        """Async image download"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }

            async with session.get(url, headers=headers, timeout=10) as response:
                if response.status == 200:
                    content = await response.read()

                    # Quick size check
                    if len(content) < 5000 or len(content) > 5000000:
                        return False

                    # Save file
                    async with aiofiles.open(filename, 'wb') as f:
                        await f.write(content)

                    # Quick image validation
                    try:
                        with Image.open(filename) as img:
                            width, height = img.size
                            if width < 100 or height < 100:
                                os.remove(filename)
                                return False
                        return True
                    except:
                        if os.path.exists(filename):
                            os.remove(filename)
                        return False

        except Exception as e:
            return False

    async def download_batch_async(self, urls, temp_dir, actor_name, batch_id):
        """Download a batch of images asynchronously"""
        connector = aiohttp.TCPConnector(limit=50, limit_per_host=10)
        timeout = aiohttp.ClientTimeout(total=30)

        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            tasks = []

            for i, url in enumerate(urls):
                filename = os.path.join(
                    temp_dir, f"{actor_name}_b{batch_id}_{i:03d}.jpg")
                task = self.download_image_async(session, url, filename)
                tasks.append((task, filename))

            results = []
            for task, filename in tasks:
                try:
                    success = await task
                    if success:
                        results.append(filename)
                except:
                    continue

            return results

    def process_images_batch(self, image_paths, actor_name):
        """Process multiple images in parallel"""
        def process_single(image_path):
            try:
                # Fast face detection
                has_face, face_msg = self.detect_face_fast(image_path)
                if not has_face:
                    os.remove(image_path)
                    return None, f"No face: {face_msg}"

                # Fast duplicate check
                is_dup, dup_msg = self.is_duplicate_fast(
                    image_path, actor_name)
                if is_dup:
                    os.remove(image_path)
                    return None, f"Duplicate: {dup_msg}"

                # Quick image optimization
                with Image.open(image_path) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')

                    # Resize if needed
                    if max(img.size) > 512:
                        img.thumbnail((512, 512), Image.Resampling.LANCZOS)
                        img.save(image_path, 'JPEG', quality=85, optimize=True)

                # Move to processed directory
                actor_dir = os.path.join(self.output_dir, "processed",
                                         actor_name.replace(' ', '_').replace('/', '_'))
                os.makedirs(actor_dir, exist_ok=True)

                processed_path = os.path.join(
                    actor_dir, os.path.basename(image_path))
                os.rename(image_path, processed_path)

                return processed_path, "Success"

            except Exception as e:
                if os.path.exists(image_path):
                    os.remove(image_path)
                return None, f"Error: {e}"

        # Process in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(process_single, image_paths))

        successful = [r for r in results if r[0] is not None]
        return len(successful)

    def get_urls_mega_batch(self, query, target_urls=1000):
        """Get massive batch of URLs using all methods"""
        all_urls = set()

        # Multiple search engines and variations
        search_engines = [
            ("google",
             f"https://www.google.com/search?q={quote(query)}&tbm=isch"),
            ("bing", f"https://www.bing.com/images/search?q={quote(query)}"),
            ("duckduckgo",
             f"https://duckduckgo.com/?q={quote(query)}&t=h_&iax=images&ia=images")
        ]

        query_variations = [
            f"{query} portrait headshot",
            f"{query} actor actress iranian",
            f"{query} بازیگر ایرانی",
            f"{query} celebrity photo",
            f"{query} official photo"
        ]

        def scrape_engine(engine_data):
            engine_name, base_url = engine_data
            urls = set()

            try:
                session = random.choice(self.session_pool)
                response = session.get(base_url, timeout=10)
                soup = BeautifulSoup(response.content, 'html.parser')

                # Extract image URLs
                for img in soup.find_all('img'):
                    src = img.get('src') or img.get('data-src')
                    if src and src.startswith('http') and 'gstatic' not in src:
                        urls.add(src)

                # Extract from scripts
                for script in soup.find_all('script'):
                    if script.string:
                        found_urls = re.findall(
                            r'https://[^"\s]+\.(?:jpg|jpeg|png|webp)', script.string)
                        for url in found_urls:
                            if 'gstatic' not in url:
                                urls.add(url)

            except Exception as e:
                print(f"Error scraping {engine_name}: {e}")

            return urls

        # Parallel scraping
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = []

            for variation in query_variations:
                for engine_name, base_template in search_engines:
                    url = base_template.replace(quote(query), quote(variation))
                    futures.append(executor.submit(
                        scrape_engine, (engine_name, url)))

            for future in as_completed(futures):
                try:
                    urls = future.result()
                    all_urls.update(urls)
                    if len(all_urls) >= target_urls:
                        break
                except Exception as e:
                    continue

        return list(all_urls)[:target_urls]

    def crawl_actor_ultra_fast(self, actor_name, target_images=100):
        """Ultra-fast crawling for one actor"""
        print(f"\n🚀 ULTRA-FAST crawling: {actor_name}")
        print(f"🎯 Target: {target_images} images")

        successful = 0
        temp_dir = os.path.join(self.output_dir, "temp",
                                actor_name.replace(' ', '_'))
        os.makedirs(temp_dir, exist_ok=True)

        # Get search queries
        queries = [actor_name]
        try:
            idx = self.actors_list.index(actor_name)
            if idx < len(self.actors_english):
                queries.append(self.actors_english[idx])
        except:
            pass

        batch_id = 0
        max_batches = 10

        while successful < target_images and batch_id < max_batches:
            print(
                f"🔄 Batch {batch_id + 1}: Need {target_images - successful} more")

            # Get massive URL batch
            all_urls = []
            for query in queries:
                urls = self.get_urls_mega_batch(query, target_urls=500)
                all_urls.extend(urls)

            # Remove duplicates
            unique_urls = list(set(all_urls))[:self.batch_size]
            print(f"   📊 Found {len(unique_urls)} unique URLs")

            if not unique_urls:
                break

            # Async download
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                downloaded_files = loop.run_until_complete(
                    self.download_batch_async(
                        unique_urls, temp_dir, actor_name, batch_id)
                )
                loop.close()
            except Exception as e:
                print(f"Async download error: {e}")
                downloaded_files = []

            print(f"   ⬇️ Downloaded: {len(downloaded_files)} images")

            if downloaded_files:
                # Parallel processing
                batch_successful = self.process_images_batch(
                    downloaded_files, actor_name)
                successful += batch_successful
                print(f"   ✅ Valid: {batch_successful} images")

            batch_id += 1

            if successful < target_images:
                time.sleep(1)  # Minimal delay

        # Cleanup
        try:
            os.rmdir(temp_dir)
        except:
            pass

        print(
            f"🎯 Result: {successful}/{target_images} images for {actor_name}")
        return successful

    def crawl_all_ultra_fast(self):
        """Ultra-fast crawling for all actors"""
        print("🚀 ULTRA-FAST CRAWLER STARTING")
        print(
            f"🎯 Target: {len(self.actors_list)} actors × 100 images = {len(self.actors_list) * 100:,} images")
        print(f"⚡ Max workers: {self.max_workers}")
        print(f"📦 Batch size: {self.batch_size}")

        start_time = time.time()
        total_downloaded = 0

        # Process actors in small groups for better resource management
        group_size = 4
        for i in range(0, len(self.actors_list), group_size):
            group = self.actors_list[i:i+group_size]

            print(
                f"\n🎭 Processing group {i//group_size + 1}: {len(group)} actors")

            # Parallel processing of actor group
            with ThreadPoolExecutor(max_workers=min(group_size, 4)) as executor:
                futures = [executor.submit(
                    self.crawl_actor_ultra_fast, actor, 100) for actor in group]

                for future in as_completed(futures):
                    try:
                        downloaded = future.result()
                        total_downloaded += downloaded
                    except Exception as e:
                        print(f"Error processing actor: {e}")

            # Progress update
            elapsed = time.time() - start_time
            avg_per_actor = total_downloaded / \
                (i + len(group)) if (i + len(group)) > 0 else 0
            estimated_total_time = (
                elapsed / (i + len(group))) * len(self.actors_list)
            remaining_time = estimated_total_time - elapsed

            print(
                f"📊 Progress: {i + len(group)}/{len(self.actors_list)} actors")
            print(f"📈 Total images: {total_downloaded:,}")
            print(f"⏱️ Average per actor: {avg_per_actor:.1f}")
            print(f"⏳ Estimated remaining: {remaining_time/60:.1f} minutes")

        elapsed = time.time() - start_time

        print(f"\n🎉 ULTRA-FAST CRAWLING COMPLETED!")
        print(f"📊 Total images: {total_downloaded:,}")
        print(f"⏱️ Total time: {elapsed/60:.1f} minutes")
        print(f"⚡ Speed: {total_downloaded/(elapsed/60):.1f} images/minute")
        print(f"📁 Images saved in: {self.output_dir}/processed/")

        return total_downloaded


if __name__ == "__main__":
    # Check required packages
    try:
        import imagehash
        from skimage.metrics import structural_similarity
        import aiohttp
        import aiofiles
    except ImportError:
        print("❌ Missing packages. Install with:")
        print("pip install ImageHash scikit-image aiohttp aiofiles")
        exit(1)

    crawler = FastIranianActorImageCrawler()

    print("⚡ ULTRA-FAST Iranian Actor Image Crawler")
    print("=" * 60)
    print(f"🎯 Target: 100 images × {len(crawler.actors_list)} actors")
    print(f"⚡ Max workers: {crawler.max_workers}")
    print(f"📦 Batch size: {crawler.batch_size}")
    print("🚀 Optimizations: Async downloads, parallel processing, caching")

    choice = input(
        "\nChoose:\n1. Ultra-fast full crawl\n2. Test single actor\n3. Test 3 actors\n\nChoice: ").strip()

    if choice == "2":
        actor = input(f"Enter actor name: ").strip()
        if actor in crawler.actors_list:
            start = time.time()
            result = crawler.crawl_actor_ultra_fast(actor, 100)
            elapsed = time.time() - start
            print(
                f"\n⚡ Speed test: {result} images in {elapsed:.1f}s ({result/elapsed*60:.1f} images/min)")
        else:
            print("❌ Actor not found")

    elif choice == "3":
        start = time.time()
        total = 0
        for actor in crawler.actors_list[:3]:
            result = crawler.crawl_actor_ultra_fast(actor, 100)
            total += result
        elapsed = time.time() - start
        print(
            f"\n⚡ Speed test: {total} images in {elapsed/60:.1f} minutes ({total/(elapsed/60):.1f} images/min)")

    else:
        total = crawler.crawl_all_ultra_fast()
        print(f"\n🎉 COMPLETED: {total:,} images downloaded!")

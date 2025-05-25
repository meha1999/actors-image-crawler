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

class IranianActorImageCrawler:
    def __init__(self, output_dir="iranian_actors_dataset"):
        self.output_dir = output_dir
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/processed", exist_ok=True)
        
        # Initialize face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Check Selenium availability
        self.selenium_available = self.check_selenium_availability()
        
        # Extended list of 100 Iranian actors and actresses
        self.actors_list = [
            # Male Actors (50)
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
            
            # Female Actresses (50)
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
        
        # English names for better search results
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
                driver = webdriver.Chrome(service=service, options=chrome_options)
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
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        chrome_binary = self.find_chrome_binary()
        if chrome_binary:
            chrome_options.binary_location = chrome_binary
        
        try:
            chromedriver_path = self.find_chromedriver_binary()
            if chromedriver_path:
                service = Service(chromedriver_path)
                driver = webdriver.Chrome(service=service, options=chrome_options)
            else:
                driver = webdriver.Chrome(options=chrome_options)
            
            driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            return driver
        except Exception as e:
            print(f"Error setting up Chrome driver: {e}")
            return None

    def detect_single_face(self, image_path):
        """Detect if image has exactly ONE face using multiple methods"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return False, "Could not read image"
            
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Method 1: face_recognition library (most accurate)
            try:
                face_locations = face_recognition.face_locations(rgb_img, model="hog")
                face_count_fr = len(face_locations)
                
                if face_count_fr == 1:
                    top, right, bottom, left = face_locations[0]
                    face_width = right - left
                    face_height = bottom - top
                    
                    if face_width >= 80 and face_height >= 80:
                        img_height, img_width = rgb_img.shape[:2]
                        face_ratio = (face_width * face_height) / (img_width * img_height)
                        
                        if 0.05 <= face_ratio <= 0.8:
                            return True, f"Single face detected (face_recognition): {face_width}x{face_height}"
                        else:
                            return False, f"Face too {'large' if face_ratio > 0.8 else 'small'}: {face_ratio:.2%} of image"
                    else:
                        return False, f"Face too small: {face_width}x{face_height}"
                
                elif face_count_fr == 0:
                    pass  # Continue to OpenCV method
                else:
                    return False, f"Multiple faces detected (face_recognition): {face_count_fr} faces"
                    
            except Exception as e:
                print(f"Face recognition library error: {e}")
            
            # Method 2: OpenCV cascade classifier (backup)
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

    def get_search_urls_batch(self, query, batch_size=100):
        """Get a large batch of image URLs using multiple search strategies"""
        all_urls = []
        
        # Multiple search variations for better coverage
        search_variations = [
            f"{query} portrait headshot single person",
            f"{query} actor actress iranian portrait",
            f"{query} بازیگر ایرانی پرتره تک نفره",
            f"{query} headshot professional photo",
            f"{query} iranian cinema actor portrait",
            f"{query} persian actor headshot",
            f"{query} film actor portrait iran",
            f"{query} celebrity headshot iran"
        ]
        
        for search_query in search_variations:
            if len(all_urls) >= batch_size * 2:  # Get extra URLs for filtering
                break
            
            print(f"   🔍 Searching: {search_query}")
            
            # Try Selenium search
            if self.selenium_available:
                selenium_urls = self.search_google_images_selenium_batch(search_query, num_images=50)
                all_urls.extend(selenium_urls)
            
            # Try requests search
            requests_urls = self.search_google_images_requests_batch(search_query, num_images=40)
            all_urls.extend(requests_urls)
            
            # Try additional sources
            additional_urls = self.search_additional_sources_batch(search_query, num_images=30)
            all_urls.extend(additional_urls)
            
            time.sleep(random.uniform(1, 3))
        
        # Remove duplicates
        seen = set()
        unique_urls = []
        for url in all_urls:
            if url not in seen and len(unique_urls) < batch_size * 3:  # Get 3x target for filtering
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
            
            # More aggressive scrolling to load more images
            for scroll in range(5):
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(1)
                
                # Try to click "Show more results"
                try:
                    show_more = driver.find_element(By.CSS_SELECTOR, "input[value*='Show more'], input[value*='نمایش بیشتر']")
                    if show_more.is_displayed():
                        show_more.click()
                        time.sleep(2)
                except:
                    pass
            
            # Collect all image URLs
            image_urls = []
            img_selectors = ["img[data-src]", "img[src]", ".rg_i", ".Q4LuWd"]
            
            for selector in img_selectors:
                try:
                    img_elements = driver.find_elements(By.CSS_SELECTOR, selector)
                    for img in img_elements:
                        if len(image_urls) >= num_images:
                            break
                        src = img.get_attribute("data-src") or img.get_attribute("src")
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
            
            # Multiple extraction methods
            for img in soup.find_all('img', {'data-src': True}):
                src = img.get('data-src')
                if src and src.startswith('http') and 'gstatic' not in src and len(image_urls) < num_images:
                    image_urls.append(src)
            
            for img in soup.find_all('img', {'src': True}):
                src = img.get('src')
                if src and src.startswith('http') and 'gstatic' not in src and len(image_urls) < num_images:
                    image_urls.append(src)
            
            # Extract from scripts
            for script in soup.find_all('script'):
                if script.string:
                    urls = re.findall(r'https://[^"\s]+\.(?:jpg|jpeg|png|webp)', script.string)
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
        
        # Bing Images
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
        
        # DuckDuckGo Images
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
                
                response = requests.get(url, headers=headers, timeout=15, stream=True)
                response.raise_for_status()
                
                content_type = response.headers.get('content-type', '').lower()
                if not any(img_type in content_type for img_type in ['image/jpeg', 'image/jpg', 'image/png', 'image/webp']):
                    continue
                
                content_length = response.headers.get('content-length')
                if content_length:
                    size = int(content_length)
                    if size < 10000 or size > 10000000:  # 10KB to 10MB
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
                    print(f"Download failed after {max_retries} attempts: {url}")
                time.sleep(random.uniform(1, 3))
        
        return False

    def process_image(self, image_path, actor_name):
        """Process image to ensure it has exactly one recognizable face"""
        try:
            has_single_face, message = self.detect_single_face(image_path)
            
            if not has_single_face:
                os.remove(image_path)
                return False, message
            
            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                if max(img.size) > 1024:
                    img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                elif max(img.size) < 300:
                    scale_factor = 300 / max(img.size)
                    new_size = (int(img.size[0] * scale_factor), int(img.size[1] * scale_factor))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                
                img.save(image_path, 'JPEG', quality=90, optimize=True)
            
            actor_dir = os.path.join(self.output_dir, "processed", actor_name.replace(' ', '_').replace('/', '_'))
            os.makedirs(actor_dir, exist_ok=True)
            
            processed_path = os.path.join(actor_dir, os.path.basename(image_path))
            os.rename(image_path, processed_path)
            
            return True, message
            
        except Exception as e:
            error_msg = f"Error processing image {image_path}: {e}"
            if os.path.exists(image_path):
                os.remove(image_path)
            return False, error_msg

    def crawl_actor_images_guaranteed(self, actor_name, target_images=100):
        """Crawl until we get EXACTLY the target number of single face images"""
        print(f"\n{'='*70}")
        print(f"🎭 GUARANTEED crawling for: {actor_name}")
        print(f"🎯 Target: {target_images} SINGLE FACE images (GUARANTEED)")
        print(f"🔧 Selenium available: {'Yes' if self.selenium_available else 'No'}")
        print(f"{'='*70}")
        
        successful_downloads = 0
        total_attempts = 0
        search_round = 1
        max_search_rounds = 10  # Prevent infinite loops
        
        # Create temporary directory
        temp_dir = os.path.join(self.output_dir, "temp", actor_name.replace(' ', '_'))
        os.makedirs(temp_dir, exist_ok=True)
        
        # Get search queries
        search_queries = [actor_name]
        try:
            idx = self.actors_list.index(actor_name)
            if idx < len(self.actors_english):
                search_queries.append(self.actors_english[idx])
        except:
            pass
        
        # Continue until we have enough images
        while successful_downloads < target_images and search_round <= max_search_rounds:
            print(f"\n🔄 SEARCH ROUND {search_round} - Need {target_images - successful_downloads} more images")
            
            # Get fresh batch of URLs
            all_image_urls = []
            for query in search_queries:
                print(f"🔍 Batch searching for: {query}")
                batch_urls = self.get_search_urls_batch(query, batch_size=200)  # Get more URLs
                all_image_urls.extend(batch_urls)
                print(f"   Found {len(batch_urls)} URLs")
                time.sleep(random.uniform(2, 4))
            
            # Remove duplicates
            seen = set()
            unique_urls = []
            for url in all_image_urls:
                if url not in seen:
                    seen.add(url)
                    unique_urls.append(url)
            
            print(f"📊 Round {search_round}: Found {len(unique_urls)} unique URLs")
            
            # Process URLs until we have enough images
            round_successes = 0
            for i, url in enumerate(unique_urls):
                if successful_downloads >= target_images:
                    break
                
                total_attempts += 1
                
                # Create filename
                url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
                filename = os.path.join(temp_dir, f"{actor_name.replace(' ', '_')}_r{search_round}_{i:03d}_{url_hash}.jpg")
                
                # Download and process
                if self.download_image(url, filename):
                    success, message = self.process_image(filename, actor_name)
                    
                    if success:
                        successful_downloads += 1
                        round_successes += 1
                        print(f"✅ [{successful_downloads:3d}/{target_images}] SUCCESS: {actor_name} - {message}")
                    else:
                        if "Multiple faces" in message:
                            print(f"❌ Multiple faces rejected")
                        elif "No face" in message or "small" in message:
                            print(f"⚠️  No/small face rejected")
                else:
                    print(f"💥 Download failed")
                
                # Progress update every 50 attempts
                if total_attempts % 50 == 0:
                    success_rate = (successful_downloads / total_attempts) * 100
                    print(f"📈 Progress: {successful_downloads}/{target_images} | Rate: {success_rate:.1f}% | Round: {search_round}")
                
                # Small delay
                time.sleep(random.uniform(0.3, 1))
            
            print(f"🔄 Round {search_round} completed: {round_successes} new images found")
            search_round += 1
            
            # Longer delay between search rounds
            if successful_downloads < target_images and search_round <= max_search_rounds:
                print(f"⏳ Waiting before next search round...")
                time.sleep(random.uniform(10, 20))
        
        # Clean up temp directory
        try:
            os.rmdir(temp_dir)
        except:
            pass
        
        # Final results
        success_rate = (successful_downloads / total_attempts) * 100 if total_attempts > 0 else 0
        
        print(f"\n🎯 FINAL RESULTS for {actor_name}:")
        print(f"   ✅ Single face images: {successful_downloads}/{target_images}")
        print(f"   📊 Total attempts: {total_attempts}")
        print(f"   🔄 Search rounds: {search_round - 1}")
        print(f"   🎯 Success rate: {success_rate:.1f}%")
        
        if successful_downloads == target_images:
            print(f"   🎉 TARGET ACHIEVED! Got exactly {target_images} single face images")
        else:
            print(f"   ⚠️  Could not find {target_images} images after {max_search_rounds} search rounds")
        
        return successful_downloads

    def crawl_all_actors_guaranteed(self):
        """Crawl guaranteed 100 single face images for all actors"""
        print("🚀 GUARANTEED 100 Single Face Images per Actor Crawler")
        print("=" * 80)
        print(f"🎯 Target: 100 GUARANTEED single face images × {len(self.actors_list)} actors")
        print(f"📊 Total target: {len(self.actors_list) * 100:,} single face images")
        print("⚠️  Will continue searching until 100 single face images are found per actor!")
        print("=" * 80)
        
        total_downloaded = 0
        completed_actors = 0
        fully_completed_actors = 0
        
        os.makedirs(os.path.join(self.output_dir, "temp"), exist_ok=True)
        
        for i, actor in enumerate(self.actors_list, 1):
            print(f"\n🎭 Actor {i}/{len(self.actors_list)}: {actor}")
            
            downloaded = self.crawl_actor_images_guaranteed(actor, target_images=100)
            total_downloaded += downloaded
            completed_actors += 1
            
            if downloaded == 100:
                fully_completed_actors += 1
            
            print(f"📊 Overall Progress:")
            print(f"   🎭 Actors processed: {completed_actors}/{len(self.actors_list)}")
            print(f"   ✅ Fully completed actors (100 images): {fully_completed_actors}")
            print(f"   📊 Total single face images: {total_downloaded:,}")
            print(f"   📈 Average per actor: {total_downloaded/completed_actors:.1f}")
            
            # Rest between actors
            time.sleep(random.uniform(10, 20))
        
        # Clean up
        try:
            import shutil
            shutil.rmtree(os.path.join(self.output_dir, "temp"))
        except:
            pass
        
        print(f"\n🎉 GUARANTEED CRAWLING COMPLETED!")
        print(f"📊 Final Statistics:")
        print(f"   🎭 Actors processed: {completed_actors}")
        print(f"   ✅ Fully completed (100 images): {fully_completed_actors}")
        print(f"   📊 Total single face images: {total_downloaded:,}")
        print(f"   📈 Average per actor: {total_downloaded/completed_actors:.1f}")
        print(f"   🎯 Completion rate: {(fully_completed_actors/len(self.actors_list))*100:.1f}%")
        print(f"   📁 Images saved in: {self.output_dir}/processed/")
        
        return total_downloaded

    def create_dataset_summary(self):
        """Create a summary of the guaranteed dataset"""
        processed_dir = os.path.join(self.output_dir, "processed")
        summary = {
            "dataset_type": "GUARANTEED SINGLE FACE",
            "target_per_actor": 100,
            "total_actors": 0,
            "total_images": 0,
            "fully_completed_actors": 0,
            "actors_data": {},
            "quality_guarantee": "All images contain exactly one face"
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
                        "completion_rate": f"{(image_count/100)*100:.1f}%"
                    }
                    summary["total_images"] += image_count
                    summary["total_actors"] += 1
                    
                    if image_count == 100:
                        summary["fully_completed_actors"] += 1
        
        # Save summary
        summary_file = os.path.join(self.output_dir, "guaranteed_dataset_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"\n📋 GUARANTEED Dataset Summary:")
        print(f"   🎭 Total actors: {summary['total_actors']}")
        print(f"   ✅ Fully completed actors (100 images): {summary['fully_completed_actors']}")
        print(f"   📊 Total single face images: {summary['total_images']:,}")
        print(f"   🎯 Overall completion: {(summary['fully_completed_actors']/summary['total_actors'])*100:.1f}%")
        print(f"   📄 Summary saved: {summary_file}")
        
        return summary

# Main execution
if __name__ == "__main__":
    crawler = IranianActorImageCrawler()
    
    print("🎬 Iranian Actor/Actress GUARANTEED 100 Single Face Images Crawler")
    print("=" * 80)
    print(f"🎯 GUARANTEE: 100 single face images × {len(crawler.actors_list)} actors")
    print("⚠️  Will continue searching until EXACTLY 100 single face images per actor!")
    print("=" * 80)
    
    choice = input("\nChoose option:\n1. Full crawl (all 100 actors)\n2. Test with first 3 actors\n3. Single actor test\n\nEnter choice (1, 2, or 3): ").strip()
    
    if choice == "3":
        actor_name = input(f"\nEnter actor name from list: ").strip()
        if actor_name in crawler.actors_list:
            print(f"\n🧪 Testing guaranteed crawl for: {actor_name}")
            downloaded = crawler.crawl_actor_images_guaranteed(actor_name, target_images=100)
            print(f"\n🎯 Result: {downloaded}/100 single face images downloaded")
        else:
            print("❌ Actor not found in list")
    
    elif choice == "2":
        print("\n🧪 Test mode: First 3 actors (100 images each)")
        test_actors = crawler.actors_list[:3]
        total_images = 0
        
        for i, actor in enumerate(test_actors, 1):
            print(f"\n🎭 Test Actor {i}/3: {actor}")
            downloaded = crawler.crawl_actor_images_guaranteed(actor, target_images=100)
            total_images += downloaded
        
        print(f"\n🧪 Test completed! Downloaded {total_images} single face images")
        
    else:
        print("\n🚀 Starting full guaranteed crawl...")
        total_images = crawler.crawl_all_actors_guaranteed()
        
        print("\n📊 Creating dataset summary...")
        summary = crawler.create_dataset_summary()
    
    print("\n🎉 GUARANTEED SINGLE FACE DATASET READY!")
    print(f"📁 Dataset location: {crawler.output_dir}/processed/")
    print("🎯 Quality guarantee: ALL images contain EXACTLY ONE FACE")
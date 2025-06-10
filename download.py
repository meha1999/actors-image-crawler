import requests
from bs4 import BeautifulSoup
import os
import time
import random
from urllib.parse import quote
import json
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from PIL import Image
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import multiprocessing


class IranianActorImageDownloader:
    def __init__(self, output_dir="iranian_actors_dataset"):
        self.output_dir = output_dir
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/raw_downloads", exist_ok=True)
        os.makedirs(f"{output_dir}/temp", exist_ok=True)

        self.selenium_available = self.check_selenium_availability()
        self.download_workers = self.get_optimal_worker_count()

        print(f"🧵 Using {self.download_workers} download workers")

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
            'مهدی پاکدل', 'امیرحسین آرمان', 'طناز طباطبایی', 'لیلا اوتادی', 'علی شادمان', 'بهنوش طباطبایی', 'پارسا پیروزفر', 'بهزاد خلج', 'شبنم قلی خانی', 'لیندا کیانی', ''
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

    def get_search_urls_batch(self, query, batch_size=100):
        """Get a large batch of image URLs using multiple search strategies"""
        all_urls = []

        search_variations = [
            f"{query} portrait headshot single person verified",
            f"{query} actor actress official portrait",
            f"{query} professional headshot photo",
            f"{query} official photo portrait verified",


            f"{query} بازیگر ایرانی پرتره تک نفره رسمی",
            f"{query} عکس رسمی بازیگر ایرانی",
            f"{query} پرتره بازیگر سینمای ایران",
            f"{query} تصویر رسمی هنرپیشه ایرانی",


            f"{query} iranian cinema actor portrait verified",
            f"{query} persian actor headshot official",
            f"{query} film actor portrait iran verified",
            f"{query} iranian movie star headshot",
            f"{query} persian cinema celebrity photo",


            f"{query} red carpet photo iran verified",
            f"{query} celebrity headshot iran official",
            f"{query} film festival photo iran",
            f"{query} press conference photo iranian actor",
            f"{query} movie premiere photo iran",


            f"{query} high resolution headshot professional",
            f"{query} studio portrait iranian actor",
            f"{query} professional photography iranian celebrity",
            f"{query} official publicity photo iran",
            f"{query} press kit photo iranian actor",


            f"{query} instagram official photo verified",
            f"{query} verified account photo iranian actor",
            f"{query} social media profile picture",


            f"{query} clear face photo high quality",
            f"{query} single person portrait professional",
            f"{query} face closeup professional photo",
            f"{query} headshot photography iranian star"
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
        """Download multiple images in parallel using ThreadPoolExecutor"""
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

    def download_actor_images(self, actor_name, target_images=300, max_search_rounds=5):
        """Download images for a single actor"""
        print(f"\n🎭 Starting download for: {actor_name}")

        actor_dir = os.path.join(self.output_dir, "raw_downloads", actor_name.replace(
            ' ', '_').replace('/', '_'))
        os.makedirs(actor_dir, exist_ok=True)

        temp_dir = os.path.join(self.output_dir, "temp",
                                actor_name.replace(' ', '_'))
        os.makedirs(temp_dir, exist_ok=True)

        successful_downloads = 0
        search_round = 1
        total_attempts = 0
        download_failures = 0

        while successful_downloads < target_images and search_round <= max_search_rounds:
            print(f"\n🔍 Search Round {search_round} for {actor_name}")

            unique_urls = self.get_search_urls_batch(
                actor_name, batch_size=100)
            print(
                f"📊 Round {search_round}: Found {len(unique_urls)} unique URLs")

            if not unique_urls:
                print(
                    f"⚠️ No URLs found for {actor_name} in round {search_round}")
                search_round += 1
                continue

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

                for success, filename in download_results:
                    if success and successful_downloads < target_images:
                        final_path = os.path.join(
                            actor_dir, os.path.basename(filename))
                        try:
                            os.rename(filename, final_path)
                            successful_downloads += 1
                            round_successes += 1
                            print(
                                f"✅ [{successful_downloads:3d}/{target_images}] Downloaded: {actor_name}")
                        except Exception as e:
                            print(f"Error moving file: {e}")

                time.sleep(random.uniform(1, 3))

            print(
                f"🔄 Round {search_round} completed: {round_successes} new images downloaded")
            search_round += 1

            if successful_downloads < target_images and search_round <= max_search_rounds:
                print(f"⏳ Waiting before next search round...")
                time.sleep(random.uniform(10, 20))

        try:
            import shutil
            shutil.rmtree(temp_dir)
        except:
            pass

        success_rate = (successful_downloads / total_attempts) * \
            100 if total_attempts > 0 else 0
        print(f"\n📊 Download Summary for {actor_name}:")
        print(
            f"   ✅ Successfully downloaded: {successful_downloads}/{target_images}")
        print(f"   🎯 Success rate: {success_rate:.1f}%")
        print(f"   💥 Download failures: {download_failures}")
        print(f"   📁 Images saved in: {actor_dir}")

        return successful_downloads

    def download_all_actors(self, target_images_per_actor=300):
        """Download images for all actors"""
        print(f"\n🚀 Starting download for {len(self.actors_list)} actors")
        print(f"🎯 Target: {target_images_per_actor} images per actor")

        total_downloaded = 0
        completed_actors = 0

        for i, actor_name in enumerate(self.actors_list, 1):
            print(f"\n{'='*60}")
            print(f"🎭 Actor {i}/{len(self.actors_list)}: {actor_name}")
            print(f"{'='*60}")

            downloaded = self.download_actor_images(
                actor_name, target_images_per_actor)
            total_downloaded += downloaded
            completed_actors += 1

            print(f"\n📊 Overall Progress:")
            print(
                f"   🎭 Actors processed: {completed_actors}/{len(self.actors_list)}")
            print(f"   📊 Total images downloaded: {total_downloaded:,}")
            print(
                f"   📈 Average per actor: {total_downloaded/completed_actors:.1f}")

            time.sleep(random.uniform(5, 10))

        print(f"\n🎉 DOWNLOAD COMPLETED!")
        print(f"📊 Final Statistics:")
        print(f"   🎭 Actors processed: {completed_actors}")
        print(f"   📊 Total images downloaded: {total_downloaded:,}")
        print(
            f"   📈 Average per actor: {total_downloaded/completed_actors:.1f}")
        print(f"   📁 Images saved in: {self.output_dir}/raw_downloads/")

        return total_downloaded

    def create_download_summary(self):
        """Create a summary of downloaded images"""
        summary = {
            "download_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_actors": 0,
            "total_images": 0,
            "actors_data": {}
        }

        raw_downloads_dir = os.path.join(self.output_dir, "raw_downloads")
        if os.path.exists(raw_downloads_dir):
            for actor_dir in os.listdir(raw_downloads_dir):
                actor_path = os.path.join(raw_downloads_dir, actor_dir)
                if os.path.isdir(actor_path):
                    image_count = len([f for f in os.listdir(actor_path)
                                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

                    summary["actors_data"][actor_dir] = {
                        "image_count": image_count,
                        "path": actor_path
                    }
                    summary["total_images"] += image_count
                    summary["total_actors"] += 1

        summary_file = os.path.join(self.output_dir, "download_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"📄 Download summary saved to: {summary_file}")
        return summary


if __name__ == "__main__":
    downloader = IranianActorImageDownloader()
    downloader.download_all_actors(target_images_per_actor=300)
    downloader.create_download_summary()

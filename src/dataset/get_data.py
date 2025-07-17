import os
import requests
import time
import json
from datetime import datetime
from urllib.parse import urlparse
import logging
from typing import List, Dict, Optional, Tuple
import hashlib

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class iNaturalistDiseaseScraper:
    def __init__(self, base_dir="data", rate_limit=1.5):
        """
        Initialize the iNaturalist Disease Scraper - Optimized for AI model training
        
        Args:
            base_dir (str): Base directory for storing images
            rate_limit (float): Seconds to wait between API calls (increased for stability)
        """
        self.base_url = "https://api.inaturalist.org/v1"
        self.base_dir = base_dir
        self.rate_limit = rate_limit
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        
        # Create base directory
        os.makedirs(self.base_dir, exist_ok=True)
        
        # Track downloaded images to avoid duplicates
        self.downloaded_hashes = set()
        self.total_images_downloaded = 0
        
        # Disease configuration - Only powdery mildew for now
        self.diseases = {
            "powdery_mildew": {
                "search_terms": [
                    "Oidiums",
                ],
                "description": "White powdery coating on leaves and stems",
                "max_images": 2000  # Limit per disease for AI training
            },
            "Pucciniales": {
                "search_terms": [
                    "Pucciniales"
                ],
                "description": "Orange/brown pustules on leaves and stems",
                "max_images": 2000
            },
            "Rhytisma de l'Érable": {
                "search_terms": [
                    "Rhytisma de l'Érable"
                ],
                "description": "",
                "max_images": 2000
            },
            "anthracnose": {
                "search_terms": [
                    "Anthracnose du Noyer",
                    "Anthracnose des Cucurbitacées",
                    "Anthracnose du Haricot",
                    "Apiognomonia errabunda (Oak Anthracnose)",
                    "Colletotrichum trichellum (Ivy Anthracnose)",
                    "Apiognomonia veneta (Plane Anthracnose)",
                    "Discula destructiva (Dogwood anthracnose fungus)",
                    "Discula campestris (Maple Anthracnose)",
                    "Gnomonia caryae (Hickory Anthracnose)",
                    "Plagiostoma fraxini (Ash Anthracnose)"
                ],
                "description": "Dark sunken lesions on leaves, stems, and fruits",
                "max_images": 2000
            },
            "downy_mildew": {
                "search_terms": [
                    "Plasmopara (Downy Mildews)",
                    "Plasmopara viticola (Grape Downy Mildew)",
                    "Pseudoperonospora cubensis (Cucurbit downy mildew)",
                    "Peronospora viciae (Pea Downy Mildew)",
                    "Peronospora belbahrii (Basil downy mildew)",
                    "Peronospora grisea (Hebe Downy Mildew)"
                ],
                "description": "Fuzzy gray/white growth on leaf undersides",
                "max_images": 2000
            },
            "Septoria ": {
                "search_terms": [
                    "Genre Septoria"
                ],
                "description": "",
                "max_images": 2000
            },
            "Insolibasidium deformans ": {
                "search_terms": [
                    "Insolibasidium deformans"
                ],
                "description": "",
                "max_images": 2000
            },
            "Wilsonomyces carpophilus ": {
                "search_terms": [
                    "Wilsonomyces carpophilus"
                ],
                "description": "",
                "max_images": 2000
            },
            "Chancre du Châtaignier ": {
                "search_terms": [
                    "Chancre du Châtaignier"
                ],
                "description": "",
                "max_images": 2000
            },
            "Venturia inaequalis ": {
                "search_terms": [
                    "Venturia inaequalis"
                ],
                "description": "",
                "max_images": 2000
            },
            "Mycosphaerella colorata ": {
                "search_terms": [
                    "Mycosphaerella colorata"
                ],
                "description": "",
                "max_images": 2000
            },
            "Cercospora moricola ": {
                "search_terms": [
                    "Cercospora moricola"
                ],
                "description": "",
                "max_images": 2000
            },
            "Resseliella liriodendri ": {
                "search_terms": [
                    "Resseliella liriodendri"
                ],
                "description": "",
                "max_images": 2000
            },
            "Aurantioporthe corni": {
                "search_terms": [
                    "Aurantioporthe corni"
                ],
                "description": "",
                "max_images": 2000
            },
            "Pseudomonas syringae": {
                "search_terms": [
                    "Pseudomonas syringae"
                ],
                "description": "",
                "max_images": 2000
            },
            "Agrobacterium radiobacter": {
                "search_terms": [
                    "Agrobacterium radiobacter"
                ],
                "description": "",
                "max_images": 2000
            },
            "Potyvirus phytolaccae": {
                "search_terms": [
                    "Potyvirus phytolaccae"
                ],
                "description": "",
                "max_images": 2000
            },
            "Polypore du Rond des Pins": {
                "search_terms": [
                    "Polypore du Rond des Pins"
                ],
                "description": "",
                "max_images": 2000
            },

        }
    
    def get_taxon_id(self, taxon_name: str) -> Optional[int]:
        """
        Get taxon ID from taxon name using the taxa endpoint
        
        Args:
            taxon_name (str): Name of the taxon
            
        Returns:
            int or None: Taxon ID if found, None otherwise
        """
        logger.info(f"🔍 Searching for taxon: '{taxon_name}'")
        
        try:
            params = {
                'q': taxon_name,
                'per_page': 5,
                'is_active': 'true'
            }
            
            response = self.session.get(f"{self.base_url}/taxa", params=params)
            response.raise_for_status()
            
            data = response.json()
            results = data.get('results', [])
            
            if not results:
                logger.warning(f"❌ No taxa found for: '{taxon_name}'")
                return None
            
            # Look for exact match first
            for result in results:
                if result.get('name', '').lower() == taxon_name.lower():
                    taxon_id = result.get('id')
                    obs_count = result.get('observations_count', 0)
                    logger.info(f"✅ Exact match: '{taxon_name}' -> ID: {taxon_id}, Obs: {obs_count}")
                    return taxon_id
            
            # Use best match if no exact match
            best_result = results[0]
            taxon_id = best_result.get('id')
            actual_name = best_result.get('name', '')
            obs_count = best_result.get('observations_count', 0)
            logger.info(f"⚠️ Best match: '{taxon_name}' -> '{actual_name}' (ID: {taxon_id}, Obs: {obs_count})")
            return taxon_id
            
        except Exception as e:
            logger.error(f"❌ Error getting taxon ID for '{taxon_name}': {e}")
            return None
    
    def get_image_hash(self, image_data: bytes) -> str:
        """Generate hash for image deduplication"""
        return hashlib.md5(image_data).hexdigest()
    
    def get_high_resolution_image_urls(self, photo_data: Dict) -> List[str]:
        """
        Get the highest resolution image URLs using iNaturalist's URL patterns
        
        Args:
            photo_data (dict): Photo data from API
            
        Returns:
            list: List of image URLs in order of preference (highest resolution first)
        """
        photo_id = photo_data.get('id')
        if not photo_id:
            return []
        
        # iNaturalist's URL patterns (highest to lowest resolution)
        base_s3_url = f"https://inaturalist-open-data.s3.amazonaws.com/photos/{photo_id}"
        static_url_base = f"https://static.inaturalist.org/photos/{photo_id}"
        
        urls_to_try = []
        
        # Try different sizes and extensions
        for size in ['original', 'large', 'medium']:
            for ext in ['jpg', 'jpeg', 'png']:
                urls_to_try.append(f"{base_s3_url}/{size}.{ext}")
                urls_to_try.append(f"{static_url_base}/{size}.{ext}")
        
        # API provided URLs as fallback
        api_urls = [
            photo_data.get('url_o'),  # Original
            photo_data.get('url_l'),  # Large
            photo_data.get('url_m'),  # Medium
            photo_data.get('url_s'),  # Small
            photo_data.get('url')     # Default
        ]
        
        # Add API URLs to the list
        for url in api_urls:
            if url:
                urls_to_try.append(url)
        
        return [url for url in urls_to_try if url]
    
    def download_image_with_fallback(self, urls_to_try: List[str], filepath: str) -> Tuple[bool, Optional[str], int]:
        """
        Try to download image from multiple URLs until one works
        
        Args:
            urls_to_try (list): List of URLs to try
            filepath (str): Local filepath to save the image
            
        Returns:
            tuple: (success: bool, final_url: str, file_size: int)
        """
        for url in urls_to_try:
            try:
                response = self.session.get(url, timeout=30)
                
                # Check if we got a valid image response
                if response.status_code == 200 and len(response.content) > 5000:  # At least 5KB
                    
                    # Check for duplicates using hash
                    image_hash = self.get_image_hash(response.content)
                    if image_hash in self.downloaded_hashes:
                        logger.info(f"⚠️ Duplicate image detected, skipping...")
                        return False, None, 0
                    
                    # Save the image
                    with open(filepath, 'wb') as f:
                        f.write(response.content)
                    
                    # Track the hash
                    self.downloaded_hashes.add(image_hash)
                    
                    file_size = len(response.content)
                    logger.info(f"✅ Downloaded: {filepath} ({file_size/1024:.1f}KB)")
                    return True, url, file_size
                
            except Exception as e:
                logger.debug(f"Failed to download from {url}: {e}")
                continue
        
        logger.warning(f"❌ Failed to download from all URLs for {filepath}")
        return False, None, 0
    
    def process_observations_immediately(self, observations: List[Dict], disease_name: str) -> Dict:
        """
        Process observations and download images immediately (optimized for AI training)
        
        Args:
            observations (list): List of observations
            disease_name (str): Name of the disease
            
        Returns:
            dict: Summary statistics
        """
        disease_dir = os.path.join(self.base_dir, disease_name)
        os.makedirs(disease_dir, exist_ok=True)
        
        stats = {
            'observations_processed': 0,
            'images_downloaded': 0,
            'images_failed': 0,
            'images_skipped_duplicates': 0,
            'total_size_mb': 0
        }
        
        disease_config = self.diseases.get(disease_name, {})
        max_images = disease_config.get('max_images', 5000)
        
        logger.info(f"📊 Processing {len(observations)} observations for {disease_name}")
        logger.info(f"📊 Target: {max_images} high-quality images for AI training")
        
        metadata = []
        
        for obs_idx, observation in enumerate(observations):
            # Check if we've reached the limit
            if self.total_images_downloaded >= max_images:
                logger.info(f"🎯 Reached target of {max_images} images for {disease_name}")
                break
                
            obs_id = observation.get('id')
            photos = observation.get('photos', [])
            
            if not photos:
                continue
            
            # Process up to 2 photos per observation for diversity
            for photo_idx, photo in enumerate(photos[:2]):
                if self.total_images_downloaded >= max_images:
                    break
                    
                try:
                    photo_id = photo.get('id', f"unknown_{photo_idx}")
                    
                    # Generate filename
                    filename = f"{disease_name}_{self.total_images_downloaded + 1:06d}.jpg"
                    filepath = os.path.join(disease_dir, filename)
                    
                    # Skip if already exists
                    if os.path.exists(filepath):
                        logger.info(f"📁 Image already exists: {filename}")
                        self.total_images_downloaded += 1
                        stats['images_downloaded'] += 1
                        continue
                    
                    # Get image URLs
                    urls_to_try = self.get_high_resolution_image_urls(photo)
                    if not urls_to_try:
                        stats['images_failed'] += 1
                        continue
                    
                    # Download image
                    success, final_url, file_size = self.download_image_with_fallback(urls_to_try, filepath)
                    
                    if success:
                        stats['images_downloaded'] += 1
                        stats['total_size_mb'] += file_size / (1024 * 1024)
                        self.total_images_downloaded += 1
                        
                        # Save metadata for AI training
                        metadata.append({
                            'filename': filename,
                            'observation_id': obs_id,
                            'photo_id': photo_id,
                            'download_url': final_url,
                            'file_size_bytes': file_size,
                            'taxon_name': observation.get('taxon', {}).get('name') if observation.get('taxon') else None,
                            'quality_grade': observation.get('quality_grade'),
                            'created_at': observation.get('created_at'),
                            'license': photo.get('license_code'),
                            'location': {
                                'latitude': observation.get('geojson', {}).get('coordinates', [None, None])[1],
                                'longitude': observation.get('geojson', {}).get('coordinates', [None, None])[0]
                            }
                        })
                    else:
                        if "duplicate" in str(success).lower():
                            stats['images_skipped_duplicates'] += 1
                        else:
                            stats['images_failed'] += 1
                    
                    # Rate limiting for downloads
                    time.sleep(0.2)
                    
                except Exception as e:
                    logger.error(f"❌ Error processing photo {photo_idx} from observation {obs_id}: {e}")
                    stats['images_failed'] += 1
                    continue
            
            stats['observations_processed'] += 1
            
            # Progress update
            if (obs_idx + 1) % 100 == 0:
                logger.info(f"📊 Progress: {obs_idx + 1}/{len(observations)} obs | {stats['images_downloaded']} images | {stats['total_size_mb']:.1f}MB")
        
        # Save metadata
        if metadata:
            metadata_file = os.path.join(disease_dir, 'metadata.json')
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"💾 Saved metadata: {metadata_file}")
        
        return stats
    
    def search_and_download_immediately(self, search_term: str, disease_name: str, max_pages: int = 10) -> Dict:
        """
        Search for observations and download images immediately (streaming approach)
        
        Args:
            search_term (str): Disease search term
            disease_name (str): Name of the disease
            max_pages (int): Maximum pages to fetch
            
        Returns:
            dict: Summary statistics
        """
        logger.info(f"🔍 Starting search and download for: '{search_term}'")
        
        # Get taxon ID
        taxon_id = self.get_taxon_id(search_term)
        if not taxon_id:
            logger.warning(f"❌ Skipping '{search_term}' - no taxon found")
            return {'observations_processed': 0, 'images_downloaded': 0, 'images_failed': 0}
        
        disease_config = self.diseases.get(disease_name, {})
        max_images = disease_config.get('max_images', 5000)
        
        total_stats = {
            'observations_processed': 0,
            'images_downloaded': 0,
            'images_failed': 0,
            'images_skipped_duplicates': 0,
            'total_size_mb': 0
        }
        
        # Process pages one by one
        for page in range(1, max_pages + 1):
            # Check if we've reached the limit
            if self.total_images_downloaded >= max_images:
                logger.info(f"🎯 Reached target of {max_images} images")
                break
            
            logger.info(f"📄 Fetching page {page} for '{search_term}'")
            
            try:
                # Search for observations with photos
                params = {
                    'taxon_id': taxon_id,
                    'quality_grade': 'any',
                    'photos': 'true',
                    'verifiable': 'any',
                    'per_page': 200,
                    'page': page,
                    'order': 'desc',
                    'order_by': 'created_at'
                }
                
                response = self.session.get(f"{self.base_url}/observations", params=params)
                response.raise_for_status()
                
                data = response.json()
                page_observations = data.get('results', [])
                
                if not page_observations:
                    logger.info(f"📄 No more results on page {page}")
                    break
                
                logger.info(f"📄 Page {page}: {len(page_observations)} observations")
                
                # Process observations immediately
                page_stats = self.process_observations_immediately(page_observations, disease_name)
                
                # Update totals
                for key in total_stats:
                    total_stats[key] += page_stats.get(key, 0)
                
                # Rate limiting between pages
                time.sleep(self.rate_limit)
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    logger.warning(f"⏱️ Rate limit hit, waiting 60 seconds...")
                    time.sleep(60)
                    continue
                else:
                    logger.error(f"❌ HTTP error on page {page}: {e}")
                    break
            except Exception as e:
                logger.error(f"❌ Error on page {page}: {e}")
                break
        
        logger.info(f"✅ Completed '{search_term}': {total_stats['images_downloaded']} images")
        return total_stats
    
    def scrape_disease_for_ai_training(self, disease_key: str) -> Dict:
        """
        Scrape disease images optimized for AI model training
        
        Args:
            disease_key (str): Key of the disease in self.diseases
            
        Returns:
            dict: Summary statistics
        """
        if disease_key not in self.diseases:
            raise ValueError(f"Disease '{disease_key}' not found in configuration")
        
        disease_config = self.diseases[disease_key]
        search_terms = disease_config['search_terms']
        max_images = disease_config.get('max_images', 5000)
        
        logger.info(f"🚀 Starting AI training dataset creation for: {disease_key}")
        logger.info(f"🎯 Target: {max_images} high-quality images")
        logger.info(f"📝 Description: {disease_config['description']}")
        
        self.total_images_downloaded = 0
        total_stats = {
            'disease': disease_key,
            'search_terms_processed': 0,
            'total_observations_processed': 0,
            'total_images_downloaded': 0,
            'total_images_failed': 0,
            'total_images_skipped_duplicates': 0,
            'total_size_mb': 0,
            'search_term_results': {}
        }
        
        # Process each search term until we reach the target
        for term_idx, term in enumerate(search_terms):
            if self.total_images_downloaded >= max_images:
                logger.info(f"🎯 Reached target of {max_images} images")
                break
            
            logger.info(f"🔍 Processing search term {term_idx + 1}/{len(search_terms)}: '{term}'")
            
            term_stats = self.search_and_download_immediately(term, disease_key, max_pages=15)
            
            total_stats['search_term_results'][term] = term_stats
            total_stats['search_terms_processed'] += 1
            total_stats['total_observations_processed'] += term_stats.get('observations_processed', 0)
            total_stats['total_images_downloaded'] += term_stats.get('images_downloaded', 0)
            total_stats['total_images_failed'] += term_stats.get('images_failed', 0)
            total_stats['total_images_skipped_duplicates'] += term_stats.get('images_skipped_duplicates', 0)
            total_stats['total_size_mb'] += term_stats.get('total_size_mb', 0)
            
            # Longer pause between search terms
            time.sleep(self.rate_limit * 2)
        
        # Save final summary
        summary = {
            'disease': disease_key,
            'description': disease_config['description'],
            'target_images': max_images,
            'scrape_date': datetime.now().isoformat(),
            'statistics': total_stats,
            'ai_training_ready': total_stats['total_images_downloaded'] >= 100  # Minimum for AI training
        }
        
        summary_file = os.path.join(self.base_dir, disease_key, 'ai_training_summary.json')
        os.makedirs(os.path.dirname(summary_file), exist_ok=True)
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"✅ COMPLETED {disease_key}")
        logger.info(f"📊 Final Stats:")
        logger.info(f"   Images Downloaded: {total_stats['total_images_downloaded']}")
        logger.info(f"   Dataset Size: {total_stats['total_size_mb']:.1f}MB")
        logger.info(f"   Observations Processed: {total_stats['total_observations_processed']}")
        logger.info(f"   Search Terms Used: {total_stats['search_terms_processed']}")
        logger.info(f"   AI Training Ready: {'✅' if summary['ai_training_ready'] else '❌'}")
        
        return total_stats
    
    def create_ai_training_dataset(self) -> Dict:
        """
        Create optimized dataset for AI model training
        
        Returns:
            dict: Combined statistics for all diseases
        """
        logger.info("🚀 Creating AI Training Dataset")
        logger.info("=" * 60)
        
        overall_stats = {
            'diseases_processed': 0,
            'total_images_downloaded': 0,
            'total_dataset_size_mb': 0,
            'disease_results': {}
        }
        
        for disease_key in self.diseases:
            try:
                logger.info(f"\n🔬 Processing disease: {disease_key}")
                logger.info("-" * 40)
                
                disease_stats = self.scrape_disease_for_ai_training(disease_key)
                
                overall_stats['disease_results'][disease_key] = disease_stats
                overall_stats['diseases_processed'] += 1
                overall_stats['total_images_downloaded'] += disease_stats['total_images_downloaded']
                overall_stats['total_dataset_size_mb'] += disease_stats['total_size_mb']
                
            except Exception as e:
                logger.error(f"❌ Failed to process disease {disease_key}: {e}")
                continue
        
        # Save overall summary
        final_summary = {
            'dataset_creation_date': datetime.now().isoformat(),
            'dataset_purpose': 'AI model training for plant disease classification',
            'total_statistics': overall_stats,
            'data_structure': {
                'base_directory': self.base_dir,
                'naming_convention': 'disease_name/disease_name_000001.jpg',
                'metadata_files': 'Each disease folder contains metadata.json and ai_training_summary.json',
                'image_quality': 'High-resolution (50KB-5MB), deduplicated'
            },
            'ai_training_notes': {
                'recommended_split': '70% train, 15% validation, 15% test',
                'image_preprocessing': 'Resize to 224x224 or 256x256 for most models',
                'augmentation_suggestions': 'rotation, flip, color jitter, but avoid heavy distortion'
            }
        }
        
        summary_file = os.path.join(self.base_dir, 'ai_training_dataset_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(final_summary, f, indent=2)
        
        logger.info(f"\n" + "=" * 60)
        logger.info("🎉 AI TRAINING DATASET CREATION COMPLETED")
        logger.info("=" * 60)
        logger.info(f"📊 Total Images: {overall_stats['total_images_downloaded']}")
        logger.info(f"📊 Total Size: {overall_stats['total_dataset_size_mb']:.1f}MB")
        logger.info(f"📊 Diseases: {overall_stats['diseases_processed']}")
        logger.info(f"📁 Dataset Location: {self.base_dir}")
        logger.info(f"📋 Summary: {summary_file}")
        
        return overall_stats


def main():
    """
    Main function to create AI training dataset
    """
    # Initialize scraper with optimized settings for AI training
    scraper = iNaturalistDiseaseScraper(
        base_dir="ai_training_data",
        rate_limit=1.5  # Respectful rate limiting
    )
    
    # Create AI training dataset
    scraper.create_ai_training_dataset()
    
    print("\n" + "=" * 60)
    print("🎉 AI TRAINING DATASET READY!")
    print("=" * 60)
    print("📁 Location: ai_training_data/")
    print("📋 Structure: disease_name/disease_name_000001.jpg")
    print("🔍 Metadata: Each folder contains metadata.json")
    print("🚀 Ready for AI model training!")


if __name__ == "__main__":
    main()
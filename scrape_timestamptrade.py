#!/usr/bin/env python
# coding: utf-8

# In[1]:


# pip install requests beautifulsoup4 tqdm diskcache tenacity ipywidgets


# In[ ]:





# In[13]:


import requests
from bs4 import BeautifulSoup
import json
from tqdm.auto import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential
import diskcache
import itertools
from multiprocessing.pool import ThreadPool, Pool
import time
from functools import partial
import pickle

# Initialize disk cache with custom JSONDisk
cache = diskcache.Cache(
    'cache_directory',
    disk=diskcache.JSONDisk,
    disk_compress_level=1,
    typed=True,
    statistics=0,  # False
    tag_index=0,  # False
    eviction_policy='least-recently-stored',
    size_limit=2**30,  # 1gb
    cull_limit=10,
    sqlite_auto_vacuum=1,  # FULL
    sqlite_cache_size=2**13,  # 8,192 pages
    sqlite_journal_mode='wal',
    sqlite_mmap_size=2**26,  # 64mb
    sqlite_synchronous=1,  # NORMAL
    disk_min_file_size=2**15,  # 32kb
    disk_pickle_protocol=pickle.DEFAULT_PROTOCOL,
)

@cache.memoize()
@retry(stop=stop_after_attempt(3), wait=wait_exponential())
def get_studio_hashes():
    response = requests.get("https://timestamp.trade/studios")
    response.raise_for_status()
    soup = BeautifulSoup(response.content, 'html.parser')
    return [a['href'].split('/')[-1] for a in soup.find_all('a', href=True) if a['href'].startswith('/studio/')]

@cache.memoize()
@retry(stop=stop_after_attempt(3), wait=wait_exponential())
def fetch_scene_data(scene_hash):
    json_url = f"https://timestamp.trade/scene/{scene_hash}"
    response = requests.get(json_url)
    return response.text

def parse_scene_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    data = {
        'title': '',
        'markers': [],
        'tags': [],
        'stash_ids': [],
        'other_ids': [],
        'urls': [],
        'performers': [],
        'hashes': [],
        'galleries': [],
        'movies': [],
        'description': ''
    }

    # Title
    title_tag = soup.find('h3')
    if title_tag:
        data['title'] = str(title_tag.get_text(strip=True))

    def parse_list_section(header_text=None, item_parser=None, header_finder=None):
        header = header_finder(soup) if header_finder else soup.find('h4', string=header_text)
        if not header:
            return []

        # Find the next <ul> sibling
        ul = header.find_next_sibling('ul')
        if not ul:
            return []

        return [item for li in ul.find_all('li') if (item := item_parser(li))]

    # Markers
    def parse_marker_item(li):
        text_parts = li.get_text(strip=True).split('-')
        description = text_parts[0].strip()
        time = text_parts[-1].strip() if len(text_parts) > 1 else ''
        tag_a = li.find('a')
        return {
            'description': str(description),
            'tag_name': str(tag_a.get_text(strip=True)) if tag_a else '',
            'tag_href': str(tag_a['href']) if tag_a and tag_a.has_attr('href') else '',
            'time': str(time)
        }

    data['markers'] = parse_list_section('markers:', parse_marker_item)

    # Tags
    def parse_tag_item(li):
        tag_a = li.find('a')
        return {
            'tag_name': str(tag_a.get_text(strip=True)) if tag_a else '',
            'tag_href': str(tag_a['href']) if tag_a and tag_a.has_attr('href') else ''
        }

    data['tags'] = parse_list_section('tags', parse_tag_item)

    # Stash IDs
    def parse_stashid_item(li):
        stash_a = li.find('a')
        return {
            'id': str(stash_a.get_text(strip=True)) if stash_a else '',
            'url': str(stash_a['href']) if stash_a and stash_a.has_attr('href') else ''
        }

    data['stash_ids'] = parse_list_section('stashid:', parse_stashid_item)

    # Other IDs
    def find_other_ids_header(soup):
        return soup.find('h4', string=lambda x: x and "Other id's" in x)

    def parse_other_id_item(li):
        text = li.get_text()
        if ' - ' in text:
            id_type, id_value = map(str.strip, text.split(' - ', 1))
            return {'id_type': str(id_type), 'id_value': str(id_value)}
        return None

    data['other_ids'] = parse_list_section(
        item_parser=parse_other_id_item, header_finder=find_other_ids_header
    )

    # URLs
    def parse_url_item(li):
        url_a = li.find('a')
        return str(url_a['href']) if url_a and url_a.has_attr('href') else None

    data['urls'] = [url for url in parse_list_section('urls:', parse_url_item) if url]

    # Performers
    def parse_performer_item(li):
        performer_a = li.find('a')
        return {
            'name': str(performer_a.get_text(strip=True)) if performer_a else '',
            'href': str(performer_a['href']) if performer_a and performer_a.has_attr('href') else ''
        }

    data['performers'] = parse_list_section('performers:', parse_performer_item)

    # Hashes
    def parse_hash_item(li):
        parts = li.get_text(strip=True).split(' - ')
        if len(parts) == 3:
            return {
                'hash': str(parts[0]),
                'value': str(parts[1]),
                'type': str(parts[2])
            }
        return None

    data['hashes'] = parse_list_section('hashes:', parse_hash_item)

    # Galleries
    def parse_gallery_item(li):
        gallery_a = li.find('a')
        return {
            'name': str(gallery_a.get_text(strip=True)) if gallery_a else '',
            'href': str(gallery_a['href']) if gallery_a and gallery_a.has_attr('href') else ''
        }

    data['galleries'] = parse_list_section('galleries:', parse_gallery_item)

    # Movies
    def parse_movie_item(li):
        movie_a = li.find('a')
        if movie_a:
            return {
                'name': str(movie_a.get_text(strip=True)),
                'href': str(movie_a['href']) if movie_a.has_attr('href') else ''
            }
        return None

    data['movies'] = parse_list_section('movies:', parse_movie_item)

    # Description
    main_div = soup.find('div', class_='main')
    if main_div:
        paragraphs = main_div.find_all('p')
        if paragraphs:
            data['description'] = str(paragraphs[-1].get_text(strip=True))

    return data

@cache.memoize()
@retry(stop=stop_after_attempt(3), wait=wait_exponential())
def get_studio_data(studio_hash):
    json_url = f"https://timestamp.trade/json-studio/{studio_hash}"
    response = requests.get(json_url)
    return response.json()

def get_scene_data(scene_hash):
    # start_time = time.time()
    html = fetch_scene_data(scene_hash)
    # fetch_time = time.time() - start_time
    # print(f"Fetching scene {scene_hash} took {fetch_time:.2f}s")
    
    # start_time = time.time()
    parsed = parse_scene_html(html)
    # parse_time = time.time() - start_time
    # print(f"Parsing scene {scene_hash} took {parse_time:.2f}s")
    
    return parsed


# scene_hash = studio_hashes[0]
# html = fetch_scene_data(scene_hash)
# parsed = parse_scene_html(html)


# In[14]:


NPROC = 40

studio_hashes = get_studio_hashes()

with open('studio_hashes.json', 'w') as f:
    json.dump(studio_hashes, f, indent=4)
    print(f"Saved {len(studio_hashes)} studio hashes to studio_hashes.json")

with ThreadPool(NPROC) as pool:
    studio_data = list(tqdm(
        pool.map(get_studio_data, studio_hashes),
        total=len(studio_hashes),
        desc="Scraping studios"
    ))

# Print or process the data as needed
with open('studio_data.json', 'w') as f:
    json.dump(list(zip(studio_hashes, studio_data)), f, indent=4)
    print(f"Saved {len(studio_data)} studio data to studio_data.json")


scene_hashes = list(set(list(itertools.chain.from_iterable([
    studio['scenes'] for studio in studio_data
]))))

with open('scene_hashes.json', 'w') as f:
    json.dump(scene_hashes, f, indent=4)
    print(f"Saved {len(scene_hashes)} scene hashes to scene_hashes.json")


# In[12]:


pool = ThreadPool(40)
# pool = Pool(20)

with pool:
    scene_data = list(tqdm(
        pool.imap(get_scene_data, scene_hashes),
        total=len(scene_hashes),
        desc="Scraping studio scenes"
    ))

with open('scene_data.json', 'w') as f:
    json.dump(list(zip(scene_hashes, scene_data)), f, indent=4)
    print(f"Saved {len(scene_data)} scene data to scene_data.json")


# In[ ]:





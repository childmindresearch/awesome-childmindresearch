#!/usr/bin/env python3
# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "requests>=2.31.0",
#     "PyGithub>=2.1.1",
#     "openai>=1.1.1",
#     "python-dotenv>=1.0.0"
# ]
# ///

"""
Generate an awesome-list README.md for all public repositories in a GitHub organization.
Uses a three-pass approach with OpenAI's API to:
1. Generate descriptions for each repository
2. Create coherent categories based on all descriptions
3. Categorize each repository using the consistent categories
Implements caching to only process repositories with changed READMEs.
"""

import os
import sys
import json
import time
import base64
import hashlib
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import requests
import openai
from github import Github, Repository, GithubException
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file if present (for local testing)
load_dotenv()

# Constants
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Fall back to childmindresearch if no org specified or available
ORG_NAME = os.environ.get("GITHUB_REPOSITORY", "").split("/")[0]  # Extract org name from repo name
if not ORG_NAME:
    ORG_NAME = "childmindresearch"
    logger.info(f"No organization specified, using default: {ORG_NAME}")

README_PATH = "README.md"
CACHE_PATH = ".github/cache/repo_cache.json"
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds


def load_cache() -> Dict[str, Any]:
    """Load the repository cache from file."""
    try:
        if os.path.exists(CACHE_PATH):
            with open(CACHE_PATH, 'r') as f:
                cache_data = json.load(f)
                logger.info(f"Cache loaded from {CACHE_PATH}")
                return cache_data
        else:
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
            logger.info(f"No cache found at {CACHE_PATH}, creating a new one")
    except Exception as e:
        logger.error(f"Error loading cache: {e}")
    
    return {"repositories": {}, "last_updated": "", "categories": []}


def save_cache(cache: Dict[str, Any]):
    """Save the repository cache to file."""
    try:
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        with open(CACHE_PATH, 'w') as f:
            json.dump(cache, f, indent=2)
            logger.info(f"Cache saved to {CACHE_PATH}")
    except Exception as e:
        logger.error(f"Error saving cache: {e}")


def get_readme_hash(readme_content: str) -> str:
    """Generate a hash for the README content."""
    if not readme_content:
        return ""
    return hashlib.md5(readme_content.encode('utf-8')).hexdigest()


def get_public_repos() -> List[Repository.Repository]:
    """Get all public repositories for the organization with rate limit handling.
    Skips archived repositories and forks."""
    try:
        # Initialize with rate limit handling
        if GITHUB_TOKEN:
            github_client = Github(GITHUB_TOKEN, retry=3, per_page=30)
        else:
            # Use anonymous access when no token is provided
            github_client = Github(per_page=30)
            logger.warning("No GitHub token provided. Using anonymous access with lower rate limits.")
            
        org = github_client.get_organization(ORG_NAME)
        all_repos = list(org.get_repos(type="public"))
        
        # Filter out archived repositories and forks
        active_repos = [repo for repo in all_repos if not repo.archived and not repo.fork]
        
        logger.info(f"Found {len(all_repos)} public repositories in the {ORG_NAME} organization")
        logger.info(f"Filtered to {len(active_repos)} active, non-fork repositories")
        
        return active_repos
    except GithubException as e:
        logger.error(f"Error accessing organization {ORG_NAME}: {e}")
        if e.status == 404:
            logger.error(f"Organization {ORG_NAME} not found. Check if the name is correct.")
        elif e.status == 403:
            logger.error("Rate limit exceeded. Consider using a token with higher rate limits or add delay between requests.")
        sys.exit(1)


def get_repo_readme(repo: Repository.Repository) -> Optional[str]:
    """Get the README content for a repository."""
    try:
        content = repo.get_readme()
        decoded_content = base64.b64decode(content.content).decode("utf-8")
        return decoded_content
    except Exception as e:
        logger.warning(f"Error getting README for {repo.name}: {e}")
        return None


def generate_repo_description(repo: Repository.Repository, readme_content: Optional[str]) -> Dict[str, Any]:
    """First pass: Generate a description for a repository."""
    repo_info = {
        "name": repo.name,
        "url": repo.html_url,
        "stars": repo.stargazers_count,
        "description": repo.description or "",
        "topics": repo.get_topics(),
        "language": repo.language,
    }
    
    # Check if OpenAI API key is available
    if not OPENAI_API_KEY:
        logger.error("No OpenAI API key provided. Using original description.")
        return {
            **repo_info,
            "generated_description": repo_info["description"] or f"A {repo_info['language'] or ''} repository."
        }

    # Create a prompt for OpenAI
    prompt = f"""
You'll be given information about a GitHub repository and its README content.
Please write a clear, concise description (1-2 sentences) that explains what this repository does.
Focus on the key functionality and purpose of the repository.

Repository Information:
- Name: {repo_info['name']}
- Original description: {repo_info['description']}
- Primary language: {repo_info['language'] or 'Not specified'}
- Topics/tags: {', '.join(repo_info['topics']) if repo_info['topics'] else 'None'}

README Content:
{readme_content[:4000] if readme_content else 'No README found'}

Your response should be just the description text, nothing else.
"""

    for attempt in range(MAX_RETRIES):
        try:
            openai.api_key = OPENAI_API_KEY
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                temperature=0.2,
                messages=[
                    {"role": "system", "content": "You're a helpful assistant that writes clear and concise descriptions for GitHub repositories."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Get the generated description
            description = response.choices[0].message.content.strip()
            logger.info(f"Generated description for {repo.name}")
            
            return {
                **repo_info,
                "generated_description": description
            }
            
        except Exception as e:
            logger.error(f"Error calling OpenAI API for {repo.name} (attempt {attempt+1}): {e}")
        
        # Wait before retrying
        if attempt < MAX_RETRIES - 1:
            backoff_time = RETRY_DELAY * (2 ** attempt)
            logger.info(f"Retrying in {backoff_time} seconds...")
            time.sleep(backoff_time)
    
    # Fallback if all retries fail
    logger.warning(f"Failed to generate description for {repo.name} after {MAX_RETRIES} attempts, using fallback")
    return {
        **repo_info,
        "generated_description": repo_info["description"] or f"A {repo_info['language'] or ''} repository."
    }


def determine_categories(repo_data: List[Dict[str, Any]]) -> List[str]:
    """Second pass: Determine coherent categories based on all repository descriptions."""
    # Check if OpenAI API key is available
    if not OPENAI_API_KEY:
        logger.error("No OpenAI API key provided. Using default categories.")
        return ["Uncategorized"]

    # Create a consolidated list of repositories with their descriptions
    repo_descriptions = []
    for repo in repo_data:
        repo_descriptions.append({
            "name": repo["name"],
            "language": repo["language"] or "Not specified",
            "description": repo["generated_description"],
            "topics": repo["topics"]
        })
    
    # Create a prompt for OpenAI - simplified to avoid JSON parsing issues
    prompt = f"""
Based on the following list of GitHub repositories and their descriptions, create 8-12 coherent, descriptive categories that would best organize this collection.
The categories should be clear, specific, and reflect the common themes and purposes across these repositories.

Repository List:
{json.dumps(repo_descriptions, indent=2)}

Please respond with a simple comma-separated list of category names.
For example: "Category 1, Category 2, Category 3"

DO NOT use any JSON formatting, bullet points, numbers, explanations, or additional text.
"""

    for attempt in range(MAX_RETRIES):
        try:
            openai.api_key = OPENAI_API_KEY
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                temperature=0.3,
                messages=[
                    {"role": "system", "content": "You are a categorization tool that responds ONLY with comma-separated category names, without any additional text, explanations, formatting, quotation marks, or brackets."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Get the raw text response
            content = response.choices[0].message.content.strip()
            
            # Parse comma-separated list
            if content:
                categories = [cat.strip() for cat in content.split(',')]
                if categories:
                    logger.info(f"Determined {len(categories)} categories: {', '.join(categories)}")
                    return categories
            
            logger.error("Empty or invalid category list response")
            
        except Exception as e:
            logger.error(f"Error calling OpenAI API for category determination (attempt {attempt+1}): {e}")
        
        # Wait before retrying
        if attempt < MAX_RETRIES - 1:
            backoff_time = RETRY_DELAY * (2 ** attempt)
            logger.info(f"Retrying in {backoff_time} seconds...")
            time.sleep(backoff_time)
    
    # Fallback if all retries fail
    logger.warning(f"Failed to determine categories after {MAX_RETRIES} attempts, using default")
    return ["Neuroimaging", "Data Processing", "Utilities", "Web Development", "Machine Learning"]


def categorize_single_repo(repo_info: Dict[str, Any], categories: List[str]) -> Dict[str, Any]:
    """Categorize a single repository using the determined categories."""
    # Check if OpenAI API key is available
    if not OPENAI_API_KEY:
        logger.error("No OpenAI API key provided. Using simple categorization.")
        repo_info["categories"] = ["Uncategorized"]
        return repo_info

    # Create an extremely simple prompt for OpenAI - asking for a plain text comma-separated list
    prompt = f"""
I need to categorize a GitHub repository into one or more predefined categories.

Repository:
Name: {repo_info['name']}
Language: {repo_info['language'] or 'Not specified'}
Description: {repo_info['generated_description']}
Topics: {', '.join(repo_info['topics']) if repo_info['topics'] else 'None'}

Available categories:
{", ".join(categories)}

Please respond ONLY with 1-3 applicable categories from the list above, separated by commas.
For example: "Category A, Category B"

DO NOT include any explanations, formatting, quotation marks, brackets, or other content.
"""

    for attempt in range(MAX_RETRIES):
        try:
            openai.api_key = OPENAI_API_KEY
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                temperature=0.1,  # Lower temperature for more predictable outputs
                messages=[
                    {"role": "system", "content": "You are a categorization tool that responds ONLY with comma-separated category names, without any additional text, explanations, formatting, quotation marks, or brackets."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Get the raw text response
            content = response.choices[0].message.content.strip()
            
            # Parse comma-separated list
            if content:
                # Simple string split
                repo_categories = [cat.strip() for cat in content.split(',')]
                
                # Filter to only include valid categories
                valid_categories = [cat for cat in repo_categories if cat in categories]
                
                # If no valid categories were found, use the first available category
                if not valid_categories and categories:
                    valid_categories = [categories[0]]
                
                repo_info["categories"] = valid_categories
                logger.info(f"Categorized {repo_info['name']}: {', '.join(valid_categories)}")
                return repo_info
            else:
                logger.error(f"Empty response for {repo_info['name']}")
                
        except Exception as e:
            logger.error(f"Error calling OpenAI API for categorization of {repo_info['name']} (attempt {attempt+1}): {e}")
        
        # Wait before retrying
        if attempt < MAX_RETRIES - 1:
            backoff_time = RETRY_DELAY * (2 ** attempt)
            logger.info(f"Retrying in {backoff_time} seconds...")
            time.sleep(backoff_time)
    
    # Fallback if all retries fail
    logger.warning(f"Failed to categorize {repo_info['name']} after {MAX_RETRIES} attempts, using fallback")
    repo_info["categories"] = ["Uncategorized"]
    return repo_info


def categorize_repositories(repo_data: List[Dict[str, Any]], categories: List[str]) -> List[Dict[str, Any]]:
    """Third pass: Categorize repositories using the determined categories."""
    # Process each repository individually
    categorized_repos = []
    
    for i, repo in enumerate(repo_data):
        # Add a small delay to avoid rate limits
        if i > 0 and i % 5 == 0:
            time.sleep(1)
        
        # Categorize the repository
        categorized_repo = categorize_single_repo(repo, categories)
        categorized_repos.append(categorized_repo)
    
    return categorized_repos


def organize_by_category(repo_data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Organize repositories by category."""
    categories: Dict[str, List[Dict[str, Any]]] = {}
    
    for repo in repo_data:
        for category in repo["categories"]:
            normalized_category = category.strip()
            if normalized_category not in categories:
                categories[normalized_category] = []
            categories[normalized_category].append(repo)
    
    # Sort categories alphabetically, but put "Uncategorized" at the end
    sorted_categories = {}
    for k in sorted(categories.keys()):
        if k != "Uncategorized":
            sorted_categories[k] = categories[k]
    
    # Add uncategorized at the end if it exists
    if "Uncategorized" in categories:
        sorted_categories["Uncategorized"] = categories["Uncategorized"]
    
    return sorted_categories


def generate_readme(categorized_repos: Dict[str, List[Dict[str, Any]]], processed_count: int = 0, cached_count: int = 0) -> str:
    """Generate the awesome-list README.md content with improved format for large repo collections."""
    try:
        if GITHUB_TOKEN:
            github_client = Github(GITHUB_TOKEN)
        else:
            github_client = Github()
            
        org = github_client.get_organization(ORG_NAME)
        org_name = org.name or ORG_NAME
    except Exception as e:
        logger.warning(f"Error getting organization name: {e}")
        org_name = ORG_NAME
    
    total_repos = processed_count + cached_count
    cache_percentage = 0 if total_repos == 0 else round((cached_count / total_repos) * 100)
    
    # Get the unique set of repositories
    unique_repos = set()
    for repos in categorized_repos.values():
        for repo in repos:
            unique_repos.add(repo["name"])
            
    # Use the count of unique repositories
    total_repo_count = len(unique_repos)
    
    readme = f"""# Awesome {org_name} Projects

A curated list of {total_repo_count} active, non-fork public repositories from the {org_name} organization.

*This list is automatically generated using [OpenAI GPT](https://openai.com/gpt-4/) for description and categorization. Archived repositories and forks are excluded.*

## Table of Contents

"""
    
    # Table of contents - more structured
    for category in categorized_repos.keys():
        slug = category.lower().replace(' ', '-').replace('/', '').replace('(', '').replace(')', '')
        repo_count = len(categorized_repos[category])
        readme += f"- [{category} ({repo_count})](#{slug})\n"
    
    readme += """

"""
    
    # List repositories by category in a more condensed format
    for category, repos in categorized_repos.items():
        slug = category.lower().replace(' ', '-').replace('/', '').replace('(', '').replace(')', '')
        readme += f"## {category}\n\n"
        
        # Sort repositories by stars (descending)
        sorted_repos = sorted(repos, key=lambda x: x["stars"], reverse=True)
        
        # Create a table for repositories
        readme += "| Repository | Language | Description |\n"
        readme += "| ---------- | -------- | ----------- |\n"
        
        for repo in sorted_repos:
            # Format language badge
            lang_badge = ""
            if repo["language"]:
                safe_language = repo["language"].replace("#", "sharp").replace("+", "plus").replace(" ", "")
                lang_badge = f"![{repo['language']}](https://img.shields.io/badge/-{safe_language}-blue?style=flat-square)"
            
            # Format stars badge - using text instead of emoji
            stars_badge = f"![Stars: {repo['stars']}](https://img.shields.io/github/stars/{ORG_NAME}/{repo['name']}?style=flat-square)"
            
            # Format repo name with badges
            repo_cell = f"[**{repo['name']}**]({repo['url']}) {stars_badge}"
            
            # Add row to table
            readme += f"| {repo_cell} | {lang_badge} | {repo['generated_description']} |\n"
        
        readme += "\n"
    
    # Add footer
    update_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
    
    readme += f"""
## About this list

This awesome list is automatically generated weekly using a GitHub Action.
Last updated: {update_time}

"""

    if total_repos > 0:
        readme += f"*In this update: {cached_count} repositories ({cache_percentage}%) were served from cache, and {processed_count} repositories were freshly analyzed.*\n"
    
    return readme


def add_rate_limit_delay(attempt: int = 0) -> None:
    """Add delay to avoid hitting rate limits too quickly."""
    # Exponential backoff with jitter
    min_delay = 1 * (2 ** attempt)  # 1, 2, 4, 8, ...
    max_delay = min_delay * 1.5
    delay = random.uniform(min_delay, max_delay)
    logger.info(f"Adding delay of {delay:.2f}s to avoid rate limits")
    time.sleep(delay)


def main():
    logger.info(f"Generating awesome list for {ORG_NAME} organization...")
    
    # Load cache
    cache = load_cache()
    cache_updated = False
    
    # Get all public repositories
    repos = get_public_repos()
    
    # Initialize counters and data storage
    repo_data = []
    processed_count = 0
    cached_count = 0
    
    # =========================================================
    # FIRST PASS: Generate descriptions for each repository
    # =========================================================
    logger.info("FIRST PASS: Generating repository descriptions")
    
    for i, repo in enumerate(repos):
        # Add small delay every few requests to avoid hitting rate limits
        if i > 0 and i % 10 == 0 and not GITHUB_TOKEN:
            add_rate_limit_delay(i // 30)  # Increase delay every 30 repos
        
        repo_full_name = f"{ORG_NAME}/{repo.name}"
        
        # Get repository README
        readme_content = get_repo_readme(repo)
        readme_hash = get_readme_hash(readme_content)
        
        # Check if repository is in cache and README hasn't changed
        if (repo_full_name in cache.get("repositories", {}) and 
            cache["repositories"][repo_full_name].get("readme_hash") == readme_hash and
            readme_hash != ""):  # Don't use cache if there was no README
            
            logger.info(f"Using cached data for repository: {repo.name}")
            cached_count += 1
            repo_info = cache["repositories"][repo_full_name]["data"]
            
            # Update any fields that might have changed but don't require re-processing
            repo_info.update({
                "stars": repo.stargazers_count,
                "topics": repo.get_topics(),
                "language": repo.language,
                "url": repo.html_url,
            })
            
        else:
            logger.info(f"Processing repository: {repo.name}")
            processed_count += 1
            
            # Generate description for repository
            repo_info = generate_repo_description(repo, readme_content)
            
            # Update cache
            if "repositories" not in cache:
                cache["repositories"] = {}
                
            cache["repositories"][repo_full_name] = {
                "readme_hash": readme_hash,
                "last_updated": datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC'),
                "data": repo_info
            }
            cache_updated = True
        
        repo_data.append(repo_info)
    
    logger.info(f"Repository processing complete: {processed_count} processed, {cached_count} used from cache")
    
    # Save cache after first pass
    if cache_updated:
        cache["last_updated"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
        save_cache(cache)
        logger.info("Cache updated after first pass (descriptions)")
        cache_updated = False
    
    # =========================================================
    # SECOND PASS: Determine categories based on all descriptions
    # =========================================================
    logger.info("SECOND PASS: Determining categories based on all repositories")
    
    # Use cached categories if no repositories were processed and cache has categories
    if processed_count == 0 and "categories" in cache and cache["categories"]:
        categories = cache["categories"]
        logger.info(f"Using {len(categories)} cached categories: {', '.join(categories)}")
    else:
        # Determine new categories based on all repositories
        categories = determine_categories(repo_data)
        
        # Update cache with new categories
        cache["categories"] = categories
        cache_updated = True
        
        # Save cache after second pass
        if cache_updated:
            cache["last_updated"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
            save_cache(cache)
            logger.info("Cache updated after second pass (categories)")
            cache_updated = False
    
    # =========================================================
    # THIRD PASS: Categorize repositories using determined categories
    # =========================================================
    logger.info("THIRD PASS: Categorizing repositories using determined categories")
    
    # Process each repository individually
    for i, repo_info in enumerate(repo_data):
        repo_full_name = f"{ORG_NAME}/{repo_info['name']}"
        
        # Check if we already have categories and they're valid
        has_valid_categories = (
            "categories" in repo_info and 
            repo_info["categories"] and 
            all(cat in categories for cat in repo_info["categories"])
        )
        
        if has_valid_categories and processed_count == 0:
            # Use existing categories if they're valid and we didn't process any new repos
            logger.info(f"Using cached categories for {repo_info['name']}: {', '.join(repo_info['categories'])}")
        else:
            # Add a small delay to avoid rate limits
            if i > 0 and i % 5 == 0:
                time.sleep(1)
            
            # Categorize the repository
            repo_info = categorize_single_repo(repo_info, categories)
            
            # Update cache after each repo categorization
            if repo_full_name in cache["repositories"]:
                cache["repositories"][repo_full_name]["data"] = repo_info
                save_cache(cache)
                logger.info(f"Cached updated for {repo_info['name']}")
    
    # Organize repositories by category
    categorized_repos = organize_by_category(repo_data)
    logger.info(f"Organized repositories into {len(categorized_repos)} categories")
    
    # Generate README.md
    readme_content = generate_readme(categorized_repos, processed_count, cached_count)
    
    # Write README.md to file with explicit UTF-8 encoding
    try:
        Path(README_PATH).write_text(readme_content, encoding='utf-8')
        logger.info(f"Awesome list generated at {README_PATH}")
    except UnicodeEncodeError as e:
        logger.error(f"Unicode encoding error: {e}")
        # Fallback: Try to replace problematic characters
        filtered_content = ''.join(c if ord(c) < 128 else '?' for c in readme_content)
        Path(README_PATH).write_text(filtered_content, encoding='utf-8')
        logger.info(f"Awesome list generated at {README_PATH} with filtered content")
    except Exception as e:
        logger.error(f"Error writing README file: {e}")
        # Last resort fallback: Write to a different file
        Path("README.filtered.md").write_text(readme_content.encode('ascii', 'ignore').decode('ascii'), encoding='utf-8')
        logger.info("Fallback README generated at README.filtered.md")


if __name__ == "__main__":
    start_time = time.time()
    try:
        # Check if OpenAI API key is available
        if not OPENAI_API_KEY:
            logger.warning("OPENAI_API_KEY environment variable is not set. Categorization will be limited.")
        
        # Initialize GitHub client
        if not GITHUB_TOKEN:
            logger.warning("GITHUB_TOKEN environment variable is not set. Using anonymous access with rate limits.")
        
        main()
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)
    finally:
        execution_time = time.time() - start_time
        logger.info(f"Script completed in {execution_time:.2f} seconds")
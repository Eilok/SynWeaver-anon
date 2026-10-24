"""
Task Synthesis Prompts

This module provides prompt templates for generating high-level tasks
from state transition triplets, following SynthAgent's approach.

Key Features:
    - Reverse task synthesis from (before_screenshot, action, after_screenshot) triplets
    - Support for previously generated tasks as context to avoid duplication
    - Website-specific task examples
    - JSON output format for structured parsing
    
Note:
    - For scroll actions: use (from_screenshot, to_screenshot)
    - For other actions: use (screenshot_op with element highlighted, to_screenshot)
    - Only screenshot-based generation is supported (no text-only mode)
"""

from typing import Optional, List, Dict, Any
import numpy as np

# Website-specific task examples for better task generation
WEBSITE_TASK_EXAMPLES = {
    "shopping": [
        "Find the price of iPhone 13.",
        "Summary reviews for Stove Top Turkey Stuffing Mix.",
        "What is the date of my first order?",
    ],
    "reddit": [
        "Tell me the top 5 posts in the 'AskReddit' subreddit.",
        "DisLike all submissions created by Private_Zannon in 'Music' subreddit.",
        "Ask for product recommendations for tennis rackets in 'Sports' subreddit.",
    ],
    'shopping_admin': [
        "What is the top-1 best-selling product in 2022?",
        "Delete all pending negative reviews",
        "Make all Taurus Elements Shell as out of stock",
    ],
    "gitlab": [
        "Create an issue in a11yproject repo with title '401 bad gateway'. Assign the issue to Roshanjossey. Set due date to be the end of 2030",
        "Star the top eight most stared repos in Gitlab",
        "Who else have access to my repo prism-theme, show me their usernames",
    ],
    "map": [
        "Tell me the full address of all international airports that are within a driving distance of 50 km to Carnegie Mellon University",
        "Measure distance between Carnegie Music Hall and UPMC Shadyside by walking",
        "Find the hotel around CMU main campus"
    ],
    "default": [
        "Navigate to the main settings page",
        "Search for a specific item using the search bar",
        "Fill out a form with required information",
        "Click on a navigation link to explore a different section",
        "Submit feedback or contact information",
    ],
}

# Website descriptions for context
WEBSITE_DESCRIPTIONS = {
    "shopping": "OneStopMarket is a web-based retail platform presenting a variety of products across categories like home & kitchen, electronics, personal care, and groceries.",
    "shopping_admin": "This website is the admin dashboard of a Magento-based e-commerce platform. It provides a central interface for managing an online store, including sales, products, customers, marketing, and reports.",
    "reddit": "Reddit is a social news and discussion platform where users can post content, comment, upvote/downvote, and participate in topic-specific communities called subreddits.",
    "gitlab": "GitLab is a web-based DevOps platform that provides git based version control, CI/CD, and project management tools. This site can be used to manage software development projects, track issues, and code editing.",
    "map": "This website is OpenStreetMap, a mapping platform where users can view, edit, search, and navigate the world.",
    "default": "A web application with various interactive features including navigation, search, forms, and content management.",
}


def _image_to_base64(image: np.ndarray) -> Optional[str]:
    """
    Convert numpy image to base64 data URL.

    Thin wrapper delegating to common.llm.image_encoding.to_data_url.
    Returns None for a missing/non-ndarray image (unchanged behaviour).
    """
    from llm.image_encoding import to_data_url
    if image is None or not isinstance(image, np.ndarray):
        return None
    return to_data_url(image)


def get_website_description(website_name: str) -> str:
    """
    Get the description for a website.
    
    Args:
        website_name: Name of the website
        
    Returns:
        Website description string
    """
    site_key = website_name.lower() if website_name.lower() in WEBSITE_DESCRIPTIONS else "default"
    return WEBSITE_DESCRIPTIONS.get(site_key, WEBSITE_DESCRIPTIONS["default"])


def get_task_examples(website_name: str) -> List[str]:
    """
    Get task examples for a website.
    
    Args:
        website_name: Name of the website
        
    Returns:
        List of example task strings
    """
    site_key = website_name.lower() if website_name.lower() in WEBSITE_TASK_EXAMPLES else "default"
    return WEBSITE_TASK_EXAMPLES.get(site_key, WEBSITE_TASK_EXAMPLES["default"])


def format_task_examples(website_name: str) -> str:
    """
    Format task examples as a string for prompts.
    
    Args:
        website_name: Name of the website
        
    Returns:
        Formatted examples string
    """
    examples = get_task_examples(website_name)
    return "\n".join(f"- {example}" for example in examples)


def prompt_generate_high_level_task(
    website_name: str,
    website_intro: str,
    current_action_str: str,
    before_screenshot: Optional[np.ndarray],
    after_screenshot: Optional[np.ndarray],
    task_examples: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Generate prompt for high-level task synthesis from a state transition.
    
    This uses the exact SynthAgent prompt format for task generation.
    
    Args:
        website_name: Name of the website (e.g., 'shopping', 'reddit')
        website_intro: Brief description of the website
        current_action_str: Human-readable description of the current action
        before_screenshot: Screenshot before action (numpy array)
            - For scroll actions: from_screenshot (before state)
            - For other actions: screenshot_op (with element highlighted in red box)
        after_screenshot: Screenshot after action (numpy array)
        task_examples: Optional formatted task examples string
        
    Returns:
        List of message dictionaries for LLM call
    """
    # Get task examples if not provided
    if task_examples is None:
        task_examples = format_task_examples(website_name)
    
    # Build the prompt following SynthAgent's exact format
    prompt = f'''You are a GUI (Graphical User Interface) expert capable of analyzing interface changes and envisioning executable tasks or instructions. Given a GUI interface change caused by an action (e.g., clicking or typing) and the corresponding element highlighted in red boxes, you are required to analyze the interface and generate related tasks.

Your task is to envision tasks based on the current action and the resulting changes in the screenshots. The output should include three components:

1. Sub-Instruction: Create a natural language instruction for the current action based on the interface changes it caused. The instruction should be concise, clear, and actionable, incorporating specific details critical to the task, such as elements, file names, timestamps, or other relevant content visible in the screenshots. For example:
   - "Click on the 'Add to Cart' button next to the product to add it to your shopping cart."
   - "Type 'OpenAI' into the search bar to find relevant articles."
   - "Scroll down to view the latest blog posts on the homepage."

2. Analysis: Carefully analyze the before-and-after screenshots step by step, focusing on the changes caused by the action. Then, examine key elements in both screenshots and consider possible operations based on these elements. For example: "The previous screen displayed the main interface of a shopping website, featuring multiple product categories and several showcased items. After clicking the 'Sign Up' button, the interface transitioned to a login page where an email and password can be entered to log into an account. The login page also provides other options, such as recovering a password, creating a new account, or logging in with a Google account."

3. High-Level Instruction: Based on the before-and-after screenshots, the action, and the analysis, generate a high-level task that you believe can be completed within the current interface. There are three types of tasks:
   - Information seeking: The user wants to obtain certain information from the webpage, such as product details, reviews, map information, or route comparisons. Please propose clear and specific questions that need an explicit answer, and avoid asking for summary-type questions, such as "summarize the information about a product."
   - Site navigation: The user wants to navigate to a specific page or state.
   - Content modification: The user wants to modify the content of a webpage or its settings.

The high-level instruction should be creative. You need to deeply analyze the elements and executable actions on the interface to generate realistic, valuable, and executable tasks that can be completed within the current GUI. The instruction should be specific, actionable, and goal-oriented, ensuring the task can be completed on the current GUI by including all critical specifics such as file names, relevant timings, or required details.

Below is a brief description of the current website: {website_intro}

Here are some examples of High-Level Instruction for reference:
{task_examples}

Current Action: {current_action_str}

Website Name: {website_name}

Before-action Screenshot: <image is provided in the first attachment> (the action's target element is highlighted in red box if applicable)

After-action Screenshot: <image is provided in the second attachment>

Please generate tasks that can be completed on the current platform, and avoid tasks that are unrelated to the current website.

You MUST respond in the following JSON format: (no extra commentary):
```json
{{
  "Sub-Instruction": "xxx",
  "Analysis": "xxx",
  "High-Level-Instruction": "xxx"
}}
```
RETURN ONLY THE JSON I ASKED FOR.
'''

    # Build message content with images
    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    
    # Add before screenshot (with element highlighted for non-scroll actions)
    if before_screenshot is not None:
        base64_url = _image_to_base64(before_screenshot)
        if base64_url:
            content.append({
                "type": "image_url",
                "image_url": {"url": base64_url}
            })
    
    # Add after screenshot
    if after_screenshot is not None:
        base64_url = _image_to_base64(after_screenshot)
        if base64_url:
            content.append({
                "type": "image_url",
                "image_url": {"url": base64_url}
            })
    
    return [{"role": "user", "content": content}]

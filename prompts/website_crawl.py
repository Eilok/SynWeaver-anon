from typing import Literal

from llm.image_encoding import to_data_url


FILTER_LISTS_BY_TEXT = """
You are a webpage structure analysis model.  
Your task is to determine whether the elements inside an HTML list (such as `<ul>` or `<ol>`) represent **repeated functionality** (only one element or one group of elements needs to be clicked) or **distinct functionality** (all elements must be clicked).

You will be given the textual structure of a list, including each element’s index and text content.  
Your goal is to analyze the semantics and structural patterns to decide whether the list items lead to **similar** or **different** functional outcomes.

## Objective

1. **If the list items represent repeated or symmetric functionality**: Only **one** representative item or **one group of items** should be clicked.

2. **If the list items represent distinct functionality**: All items must be clicked.

## Classification Criteria

### **1. Treat the items as *functionally identical* (decision = "single") if:**

All or most items exhibit these characteristics:

- Text content represents members of the same category (e.g., product categories, product titles, tags, results)
- Elements have a highly repetitive structure (e.g., repeated `<li>`, repeated `<a>`, repeated `<div>` templates)
- Items correspond to the same type of resource (e.g., products, categories, search results)
- Clicking each item typically leads to pages of the **same structure but different content** (e.g., different product detail pages)
- The list visually or semantically presents a “set of options of the same type”

If these conditions are met, treat the list as **repeated functionality**.

### **2. Treat the items as *functionally different* (decision = "all") if:**

Any of the following are true:

- Text content clearly indicates different action targets (e.g., “new communication”, “Sign In”, “search communication”, “profile”)
- Items represent different task categories (e.g., login, registration, account management)
- The list acts as a navigation menu with distinct entry points  
- Clicking each item evidently results in **different functionality**, not symmetrical options

If any such condition applies, treat the list as **distinct functionality**.

## Output Format

You must output a JSON object in the following format:
```json
{
  "decision": "single" or "all",
  "reason": "Explanation of the decision",
  "selected_indices": when the decision is "single", return [list of indices]. Otherwise, return an empty list.
}
```

## Example

### Example 1

Input:
```
<ul>
[21]<a> Black Friday </a>
[22]<a> Amazon Haul </a>
[23]<span> Medical Care </span>
[24]<button>  </button>
[25]<a> Best Sellers </a>
[26]<a> Amazon Basics </a>
[27]<a> New Releases </a>
[28]<a> Books </a>
[29]<a> Registry </a>
[30]<span> Groceries </span>
[31]<button>  </button>
[32]<span> Gift Cards </span>
</ul>
```

Output:
```json
{
  "decision": "all",
  "reason": "The list items represent distinct functional entry points rather than repeated or symmetric options. Items such as 'Black Friday', 'Amazon Haul', 'Medical Care', 'Best Sellers' all lead to different content categories or features. They do not share a repeated structure or identical functional behavior. Therefore, each item must be treated as a unique function.",
  "selected_indices": []
}
```

### Example 2

Input:
```
<ul>
[12]<img>  </img>
[13]<span> Outstanding Fiction of the past 25 years </span>
[14]<img>  </img>
[15]<span> Unforgettable Nonfiction of the past 25 years </span>
[16]<img>  </img>
[17]<span> Most buzzworthy books of the past 25 years </span>
[18]<img>  </img>
[19]<span> Editors’ personal favorites </span>
</ul>
```

Output:
```
{
  "decision": "single",
  "reason": "The list items represent a set of thematically similar options: each entry pairs an image with a descriptive label for a specific curated book collection. The structure is repetitive, and clicking any item is expected to lead to a page of the same functional type (a book collection page), differing only in content. Therefore, these items exhibit repeated functionality.",
  "selected_indices": [12, 13]
}
```
"""

FORM_FILL_PROMPT = """
You are an intelligent form filling assistant. Your task is to generate reasonable test values for each input field in a web form based on the form context and field semantics.

## Context

You will receive:
1. **Page URL**: The current page URL to help understand the context
2. **Accessibility Tree**: A structured representation of the page content around the form
3. **Form Fields**: A list of input fields in the form, each with:
   - Index number (1, 2, 3, ...)
   - Tag type (input/textarea/select)
   - Input type (for input elements: text, email, password, etc.)
   - Name attribute
   - Aria-label
   - Placeholder text
   - Current text content

## Your Task

Generate appropriate test values for each form field. The values should be:
1. **Semantically appropriate**: Match the expected content type (email for email fields, phone for phone fields, etc.)
2. **Realistic**: Look like real user input, not obviously fake
3. **Valid**: Pass basic validation (proper email format, valid phone format, etc.)
4. **Contextual**: Consider the form's purpose based on URL and accessibility tree

## Output Format

Return a JSON object with a "values" array containing the fill value for each field **in order**:

```json
{
  "values": ["value_for_field_1", "value_for_field_2", "value_for_field_3", ...],
  "reasoning": "Brief explanation of the values chosen"
}
```

**Important**:
- The array length MUST match the number of input fields
- Each value corresponds to the field at that index position
- For select elements, provide the option text to select
- For password fields, use a strong test password like "TestPass123!"
- For date fields, use ISO format (YYYY-MM-DD)

## Examples

### Example 1: Login Form

Input:
- URL: https://example.com/login
- Fields:
  1. input[type=email] name="email" placeholder="Enter your email"
  2. input[type=password] name="password" placeholder="Password"

Output:
```json
{
  "values": ["testuser@example.com", "TestPass123!"],
  "reasoning": "Standard login form with email and password fields"
}
```

### Example 2: Registration Form

Input:
- URL: https://shop.example.com/register
- Fields:
  1. input[type=text] name="first_name" aria-label="First Name"
  2. input[type=text] name="last_name" aria-label="Last Name"
  3. input[type=email] name="email"
  4. input[type=tel] name="phone" placeholder="Phone number"

Output:
```json
{
  "values": ["John", "Smith", "john.smith@example.com", "555-123-4567"],
  "reasoning": "E-commerce registration form requiring personal contact information"
}
```
"""

COMPARE_STATES_PROMPT = """
You are an expert web page analyzer. Your task is to determine whether two screenshots represent the **same functional state** of a web page, even if some minor visual elements differ.

# Background

During web crawling and state replay, certain transient UI elements (like toast notifications, temporary alerts, or dynamic content) may appear or disappear. However, the underlying page state remains the same. We need to identify whether two screenshots represent the same functional page state while ignoring these superficial differences.

# Task

Compare the two provided screenshots and determine if they represent the **same web page state**.

# What to Consider as the SAME State

Two screenshots should be considered the **same state** if they share:

1. **Core Page Structure**:
   - Same page layout and main content areas
   - Same navigation elements (header, footer, sidebar)
   - Same primary content sections

2. **Functional Elements**:
   - Same interactive elements (buttons, links, forms, input fields)
   - Same product listings, article content, or data tables
   - Same menu items and navigation options

3. **Page Identity**:
   - Same URL path (if visible)
   - Same page title or heading
   - Same main purpose/function

# What to IGNORE (Acceptable Differences)

The following differences should be **ignored** when comparing states:

1. **Transient Notifications**:
   - Toast messages (success/error/warning notifications)
   - Temporary alert boxes
   - Popup hints or tooltips
   - Tutorial/onboarding overlays
   - Cookie consent banners

2. **Dynamic Content**:
   - Timestamps (e.g., "2 minutes ago" vs "5 minutes ago")
   - Real-time counters (shopping cart quantity, notification badges)
   - Live data feeds (stock prices, weather updates)
   - Advertisement content
   - Personalized recommendations

3. **Visual State Changes**:
   - Hover effects or focus states
   - Loading spinners or progress indicators
   - Scroll position
   - Animation states
   - Dropdown menu open/closed states

4. **Session-Specific Elements**:
   - Session IDs or tokens in URLs
   - CSRF tokens
   - Temporary promotional banners
   - A/B testing variations (minor UI tweaks)

# Decision Criteria

- **SAME**: If the core page structure, main content, and functional elements are identical (ignoring the acceptable differences listed above)
- **DIFFERENT**: If there are substantial differences in layout, content, navigation, or functionality

# Output Format

Provide your analysis in the following JSON format:

```json
{
  "decision": "same" or "different",
  "confidence": 0.0 to 1.0,
  "reasoning": "Simple explanation of your decision",
}
```
"""

def get_crawler_prompt(type: Literal["filter_lists", "compare_state", "form_fill"]) -> str:
    """get the prompt template by type"""
    if type == "filter_lists":
        return FILTER_LISTS_BY_TEXT
    elif type == "compare_state":
        return COMPARE_STATES_PROMPT
    elif type == "form_fill":
        return FORM_FILL_PROMPT
    else:
        raise ValueError(f"Unsupported prompt type: {type}")
    
def build_user_prompt_for_filter_list(list_structure: dict) -> str:
    """build the user prompt for filtering list structures"""
    user_prompt = f"<{list_structure['containerTag']}>\n"
    for item in list_structure["items"]:
        user_prompt += f"[{item['page_id']}]<{item['tag']}> {item['text']} </{item['tag']}>\n"
    user_prompt += f"</{list_structure['containerTag']}>"

    return user_prompt


def build_user_prompt_for_form_fill(
    url: str,
    accessibility_tree_str: str,
    input_fields: list
) -> str:
    """
    Build the user prompt for LLM-based form filling.
    
    Args:
        url: Current page URL
        accessibility_tree_str: Accessibility tree string for context
        input_fields: List of Element objects or dicts representing form input fields
        
    Returns:
        Formatted user prompt string
    """
    # Build field descriptions
    field_descriptions = []
    for i, field in enumerate(input_fields, start=1):
        # Handle both Element objects and dicts
        if hasattr(field, 'tag'):
            tag = field.tag
            name = field.name or ""
            aria_label = field.aria_label or ""
            text = field.text or ""
            input_type = getattr(field, 'input_type', '') or ""
            placeholder = getattr(field, 'placeholder', '') or ""
        else:
            tag = field.get('tag', '')
            name = field.get('name', '')
            aria_label = field.get('ariaLabel', '')
            text = field.get('text', '')
            input_type = field.get('inputType', '')
            placeholder = field.get('placeholder', '')
        
        # Build description
        desc_parts = [f"{i}. <{tag}>"]
        if input_type:
            desc_parts.append(f"type=\"{input_type}\"")
        if name:
            desc_parts.append(f"name=\"{name}\"")
        if aria_label:
            desc_parts.append(f"aria-label=\"{aria_label}\"")
        if placeholder:
            desc_parts.append(f"placeholder=\"{placeholder}\"")
        if text:
            desc_parts.append(f"text=\"{text}\"")
        
        field_descriptions.append(" ".join(desc_parts))
    
    # Truncate accessibility tree if too long (keep ~2000 chars)
    max_tree_len = 2000
    if len(accessibility_tree_str) > max_tree_len:
        accessibility_tree_str = accessibility_tree_str[:max_tree_len] + "\n... (truncated)"
    
    user_prompt = f"""## Page URL
{url}

## Accessibility Tree Context
```
{accessibility_tree_str}
```

## Form Fields
{chr(10).join(field_descriptions)}

Please generate appropriate fill values for each field in order."""
    
    return user_prompt

def build_user_prompt_for_compare_state(screenshot1_bytes: bytes, screenshot2_bytes: bytes) -> list[dict]:
    """build the user prompt for comparing two states"""
    user_prompt: list[dict] = [
        {"type": "image_url", "image_url": {"url": to_data_url(screenshot1_bytes)}},
        {"type": "image_url", "image_url": {"url": to_data_url(screenshot2_bytes)}},
        {"type": "text", "text": "Compare the two states"},
    ]

    return user_prompt

# FIXME
def _get_prompt(
        type: Literal["filter_by_vision", "filter_by_text"]
) -> str:
    """get the prompt template by type"""
    if type == "filter_by_vision":
        return FILTER_ELEMENTS_BY_VISION
    elif type == "filter_by_text":
        return FILTER_ELEMENTS_BY_TEXT
    else:
        raise ValueError(f"Unsupported prompt type: {type}")

# FIXME
def _build_user_prompt_for_filter(elements: list[dict], screenshot_bytes: bytes | None = None) -> str | list[dict]:
    """build the user prompt for filtering elements"""
    element_descriptions = []
    for elem in elements:
        desc = f"[{elem['page_id']}] Tag: <{elem['tag']}>, Text: {elem['text']}, XPath: {elem['xpath']}"
        element_descriptions.append(desc)
    elements_info = "\n".join(element_descriptions)

    if screenshot_bytes:
        user_prompt = []
        user_prompt.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": to_data_url(screenshot_bytes)
                },
            }
        )
        text = (
            f"Here are the UI elements on the web page:\n"
            f"{elements_info}\n"
            "Now, filter the redundant sibling elements."
        )
        user_prompt.append(
            {
                "type": "text",
                "text": text,
            }
        )
    else:
        user_prompt = (
            f"Here are the UI elements on the web page:\n"
            f"{elements_info}\n"
            "Now, filter the redundant sibling elements."
        )

    return user_prompt
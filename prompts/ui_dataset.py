"""
Prompt Templates for UI Dataset Generation

This module contains prompt templates for generating UI training data
using a teacher VLM model. Each prompt is designed to elicit specific
types of descriptions for fine-tuning student models.

Prompt Categories:
    1. Page Description + QA: Describe page content and generate QA pairs
    2. Element Description + State Transition: Describe element functionality and state changes
"""

from typing import Literal


# =============================================================================
# Page Description + Page QA Prompts
# =============================================================================

PAGE_DESCRIPTION_AND_QA_SYSTEM_PROMPT = """You are an expert UI analyst tasked with analyzing web page screenshots and accessibility trees. Your analysis will be used to train AI models to understand web interfaces.

You will receive:
1. A screenshot of a web page
2. The page's accessibility tree

Your task is to provide TWO types of analysis:

## Part 1: Page Description
Provide a comprehensive description of the page that includes:
- Visual layout and structure (header, navigation, main content, sidebar, footer)
- All visible UI elements and their apparent functions
- Visual styling (colors, typography, spacing)

## Part 2: Page QA
Generate ONE informative question-answer pair about the page. The question should:
- Ask about specific information visible on the page
- Be answerable ONLY by looking at the screenshot
- Focus on extracting factual information (not opinions)
- Examples: "What is the main heading?", "How many products are displayed?", "What navigation options are available?"

## Output Format
You MUST respond in the following JSON format:
```json
{
    "page_description": "Your detailed page description here...",
    "question": "Your question about the page content...",
    "answer": "The answer based on the screenshot..."
}
```"""


PAGE_DESCRIPTION_AND_QA_USER_TEMPLATE = """Analyze the following web page:

URL: {url}

Accessibility Tree:
```
{accessibility_tree}
```

Please provide:
1. A detailed description of this page
2. One question-answer pair about specific information visible on the page

Remember to respond in the specified JSON format."""


# =============================================================================
# Element Description + State Transition Prompts
# =============================================================================

ELEMENT_AND_TRANSITION_SYSTEM_PROMPT = """You are an expert UI analyst tasked with analyzing web page interactions. Your analysis will be used to train AI models to understand how UI elements work and how pages change after user actions.

You will receive:
1. A "before" screenshot showing the page with a marked element
2. An "after" screenshot showing the page after the action was performed
3. Information about the action taken

Your task is to provide TWO types of analysis:

## Part 1: Element Description
Describe the functionality of the interacted element and how users would typically interact with it in the interface.

## Part 2: State Transition Description
Describe the changes between the before and after states.

IMPORTANT: In the state transition description, describe ONLY what changed on the page. Do NOT mention the action taken (click, fill, etc.) - focus purely on the observable differences between the two screenshots.

## Output Format
You MUST respond in the following JSON format:
```json
{
    "element_description": "Description of the element's functionality...",
    "state_transition": "Description of how the page changed (without mentioning the action)..."
}
```"""


ELEMENT_AND_TRANSITION_USER_TEMPLATE = """Analyze this web page interaction:

Action Performed: {action}
Element: {element}
URL Before: {from_url}
URL After: {to_url}

The first image shows the page BEFORE the action (with the target element highlighted).
The second image shows the page AFTER the action was performed.

Please provide:
1. A description of the element's functionality
2. A description of how the page changed (without mentioning what action was taken)

Remember to respond in the specified JSON format."""


# =============================================================================
# Helper Functions
# =============================================================================

def get_ui_dataset_prompt(
    prompt_type: Literal["page_description_qa", "element_transition"]
) -> str:
    """
    Get the system prompt for a specific dataset type.
    
    Args:
        prompt_type: Type of prompt to retrieve
        
    Returns:
        System prompt string
    """
    if prompt_type == "page_description_qa":
        return PAGE_DESCRIPTION_AND_QA_SYSTEM_PROMPT
    elif prompt_type == "element_transition":
        return ELEMENT_AND_TRANSITION_SYSTEM_PROMPT
    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}")


def build_page_description_qa_prompt(
    url: str,
    accessibility_tree: str
) -> str:
    """
    Build the user prompt for page description and QA generation.
    
    Args:
        url: Page URL
        accessibility_tree: Full accessibility tree content
        
    Returns:
        Formatted user prompt string
    """
    return PAGE_DESCRIPTION_AND_QA_USER_TEMPLATE.format(
        url=url,
        accessibility_tree=accessibility_tree
    )


def build_element_transition_prompt(
    action: str,
    element: str,
    from_url: str,
    to_url: str
) -> str:
    """
    Build the user prompt for element description and state transition generation.
    
    Args:
        action: Action description (e.g., "Click on <button>Submit</button>")
        element: Element description
        from_url: URL before action
        to_url: URL after action
        
    Returns:
        Formatted user prompt string
    """
    return ELEMENT_AND_TRANSITION_USER_TEMPLATE.format(
        action=action,
        element=element,
        from_url=from_url,
        to_url=to_url
    )


# =============================================================================
# Student Model Training Prompts (for the generated datasets)
# =============================================================================

# These are the instruction templates used in the final training data

PAGE_DESCRIPTION_INSTRUCTION = """Describe this web page in detail based on the screenshot and accessibility tree.

Accessibility Tree:
```
{accessibility_tree}
```

Include information about:
- Visual layout and structure (header, navigation, main content, sidebar, footer)
- All visible UI elements and their apparent functions
- Visual styling (colors, typography, spacing)"""

PAGE_QA_INSTRUCTION = """Look at this web page screenshot and accessibility tree, then answer the following question:

Accessibility Tree:
```
{accessibility_tree}
```

Question: {question}"""

ELEMENT_DESCRIPTION_INSTRUCTION = """The highlighted element in this screenshot is about to be interacted with.

Current page accessibility tree:
```
{accessibility_tree}
```

Target element: {element}

Describe the functionality of the interacted element and how users would typically interact with it in the interface."""

STATE_PREDICTION_INSTRUCTION = """Given the current page state and the action to be performed, predict how the page will change.

Current page accessibility tree:
```
{accessibility_tree}
```

Action to perform: {action}

Describe only what changes you expect to see on the page after this action."""

ACTION_PREDICTION_INSTRUCTION = """Compare these two screenshots showing a web page before and after a user action.

Accessibility Tree (Before):
```
{from_accessibility_tree}
```

Accessibility Tree (After):
```
{to_accessibility_tree}
```

Based on the visual changes and accessibility tree differences, predict what action the user took.

Your answer should include:
1. A description of what changed
2. The predicted action including the action and the node (e.g., 'Click on [576] link 'add the blog')"""


def get_student_instruction(
    instruction_type: Literal[
        "page_description", 
        "page_qa", 
        "element_description",
        "state_prediction",
        "action_prediction"
    ],
    **kwargs
) -> str:
    """
    Get the instruction template for student model training.
    
    Args:
        instruction_type: Type of instruction
        **kwargs: Template variables (e.g., question, action)
        
    Returns:
        Formatted instruction string
    """
    templates = {
        "page_description": PAGE_DESCRIPTION_INSTRUCTION,
        "page_qa": PAGE_QA_INSTRUCTION,
        "element_description": ELEMENT_DESCRIPTION_INSTRUCTION,
        "state_prediction": STATE_PREDICTION_INSTRUCTION,
        "action_prediction": ACTION_PREDICTION_INSTRUCTION,
    }
    
    template = templates.get(instruction_type)
    if template is None:
        raise ValueError(f"Unknown instruction type: {instruction_type}")
    
    return template.format(**kwargs) if kwargs else template

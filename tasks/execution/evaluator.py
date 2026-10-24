"""
Evaluator Module

This module provides automatic task result verification based on SynthAgent's
evaluator implementation.

Key Features:
1. StringEvaluator: exact_match, must_include, fuzzy_match (LLM-based)
2. URLEvaluator: URL matching with query parameter support
3. HTMLContentEvaluator: Check HTML content in page
4. EvaluatorComb: Combine multiple evaluators

Ported from SynthAgent's syn/evaluators.py with adaptations for this project.
"""

import os
import html
import time
import json
import logging
import urllib.parse
import collections
from urllib.parse import urlparse
from typing import Optional, Dict, Any, List, Union

try:
    from nltk.tokenize import word_tokenize
except ImportError:
    # Fallback if nltk is not available
    def word_tokenize(text: str) -> List[str]:
        return text.split()

try:
    import requests
except ImportError:
    requests = None

from tasks.models import (
    ActionType, LowLevelTask, HighLevelTask,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# Utility Functions
# ==============================================================================

def robust_json_loads(text: str | dict) -> Optional[Dict[str, Any]]:
    """
    Robustly parse JSON from text.

    Thin wrapper delegating to llm.json_parsing.parse_json (unified
    implementation; strips // comments and fences, repairs light damage).

    Args:
        text: Text to parse

    Returns:
        Parsed dict or None
    """
    from llm.json_parsing import parse_json
    if isinstance(text, dict):
        return text
    result = parse_json(text)
    return result if isinstance(result, dict) else None


# ==============================================================================
# Helper Functions for WebArena environments
# ==============================================================================

def get_env_url(env_name: str) -> str:
    """Get environment URL from environment variables."""
    return os.environ.get(env_name.upper(), f"http://127.0.0.1:7770")


def shopping_get_auth_token() -> str:
    """Get authentication token for shopping site."""
    if requests is None:
        logger.warning("requests module not available for shopping auth")
        return ""
    
    shopping_url = os.environ.get("SHOPPING", "http://127.0.0.1:7770")
    accounts = {
        "username": os.environ.get("SHOPPING_ADMIN_USER", "admin"),
        "password": os.environ.get("SHOPPING_ADMIN_PASS", "admin123"),
    }
    
    try:
        response = requests.post(
            url=f"{shopping_url}/rest/default/V1/integration/admin/token",
            headers={"content-type": "application/json"},
            data=json.dumps(accounts),
            timeout=30,
        )
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        logger.warning(f"Failed to get shopping auth token: {e}")
    
    return ""


def shopping_get_latest_order_url() -> str:
    """Get the latest order url from the shopping website."""
    if requests is None:
        return ""
    
    shopping_url = os.environ.get("SHOPPING", "http://127.0.0.1:7770")
    token = shopping_get_auth_token()
    if not token:
        return ""
    
    try:
        header = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        params = {
            "searchCriteria[sortOrders][0][field]": "created_at",
            "searchCriteria[sortOrders][0][direction]": "DESC",
            "searchCriteria[pageSize]": "1",
        }
        response = requests.get(
            f"{shopping_url}/rest/V1/orders",
            params=params,
            headers=header,
            timeout=30,
        )
        if response.status_code == 200:
            response_obj = response.json()["items"][0]
            order_id = int(response_obj["increment_id"])
            return f"{shopping_url}/sales/order/view/order_id/{order_id}/"
    except Exception as e:
        logger.warning(f"Failed to get latest order URL: {e}")
    
    return ""


def shopping_get_sku_latest_review_author(sku: str) -> str:
    """Get the author of the latest review for a shopping SKU.

    Ported verbatim (behavior-wise) from WebArena's
    evaluation_harness/helper_functions.py so that benchmark
    ``program_html`` locators of the form
    ``func:shopping_get_sku_latest_review_author('<sku>')`` resolve.
    """
    if requests is None:
        return ""

    shopping_url = os.environ.get("SHOPPING", "http://127.0.0.1:7770")
    token = shopping_get_auth_token()
    if not token:
        return ""

    try:
        header = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        response = requests.get(
            f"{shopping_url}/rest/V1/products/{sku}/reviews",
            headers=header,
            timeout=30,
        )
        if response.status_code != 200:
            return ""
        response_obj = response.json()
        if len(response_obj) == 0:
            return ""
        return response_obj[-1]["nickname"]
    except Exception as e:
        logger.warning(f"Failed to get SKU latest review author: {e}")

    return ""


def shopping_get_sku_latest_review_rating(sku: str) -> str:
    """Get the rating (percent) of the latest review for a shopping SKU.

    Ported from WebArena's evaluation_harness/helper_functions.py so that
    benchmark ``program_html`` locators of the form
    ``func:shopping_get_sku_latest_review_rating('<sku>')`` resolve.
    """
    if requests is None:
        return ""

    shopping_url = os.environ.get("SHOPPING", "http://127.0.0.1:7770")
    token = shopping_get_auth_token()
    if not token:
        return ""

    try:
        header = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        response = requests.get(
            f"{shopping_url}/rest/V1/products/{sku}/reviews",
            headers=header,
            timeout=30,
        )
        if response.status_code != 200:
            return ""
        response_obj = response.json()
        if len(response_obj) == 0:
            return ""
        # WebArena asserts the first rating is the "Rating" dimension.
        if response_obj[0]["ratings"][0]["rating_name"] != "Rating":
            return ""
        return str(response_obj[-1]["ratings"][0]["percent"])
    except Exception as e:
        logger.warning(f"Failed to get SKU latest review rating: {e}")

    return ""


def reddit_get_post_url(url: str) -> str:
    """Get the post url from a Reddit URL."""
    domain = urlparse(url).netloc
    tok_url = urlparse(url).path.split("/")
    
    if len(tok_url) < 4:
        return url
    if tok_url[1] != "f":
        return url
    
    subreddit = tok_url[2]
    post_id = tok_url[3]
    scheme = urlparse(url).scheme
    return f"{scheme}://{domain}/f/{subreddit}/{post_id}/"


def gitlab_get_project_member_role(page: Any, account_name: str) -> str:
    """Get GitLab project member role."""
    try:
        account_idx = page.evaluate(
            f"""(() => {{
                const elements = document.querySelectorAll("td[data-label='Account'] span.gl-avatar-labeled-sublabel");
                let index = -1;
                for(let i = 0; i < elements.length; i++) {{
                    if(elements[i].outerText === '@{account_name}') {{
                        index = i;
                        break;
                    }}
                }}
                return index;
            }})()"""
        )
        
        if account_idx >= 0:
            role = page.evaluate(
                f"""(() => {{
                    return document.querySelectorAll("td.col-max-role span")[{account_idx}].outerText;
                }})()"""
            )
            return role
    except Exception:
        pass

    return ""


# WebArena's benchmark data references this helper by its historical (misspelled)
# name ``gitlab_get_project_memeber_role`` inside ``program_html`` locators, e.g.
# ``func:gitlab_get_project_memeber_role(__page__, 'byteblaze')``. The locator is
# resolved via ``eval()`` against this module's namespace, so the misspelled name
# MUST exist or evaluation raises NameError → empty content → false negative.
# Keep the correctly-spelled definition above and expose the misspelled alias.
gitlab_get_project_memeber_role = gitlab_get_project_member_role


# ==============================================================================
# Base Evaluator
# ==============================================================================

class Evaluator:
    """
    Base evaluator class.
    
    Provides LLM-based fuzzy matching for semantic equivalence checking.
    """
    
    def __init__(
        self,
        eval_config: Dict[str, Any],
        task: str,
        last_action_summary: str,
        page: Any,
        llm_client: Any = None,
        llm_eval_config: Optional[Dict[str, Any]] = None,
        url_mapping: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize evaluator.
        
        Args:
            eval_config: Evaluation configuration dict
            task: Task description
            last_action_summary: Summary of last action (model's answer)
            page: Playwright page object
            llm_client: LLM client for fuzzy matching
            llm_eval_config: LLM configuration for evaluation
            url_mapping: Mapping of URL placeholders to actual URLs
                         e.g., {'__GITLAB__': 'http://localhost:8023/explore'}
        """
        self.eval_config = eval_config
        self.task = task
        self.last_action_summary = last_action_summary
        self.page = page
        self.llm_client = llm_client
        self.url_mapping = url_mapping or {}
        
        if llm_eval_config is None:
            self.llm_eval_config = {
                "model": "gpt-4o",
                "temperature": 0.0,
                "max_tokens": 768,
            }
        else:
            self.llm_eval_config = llm_eval_config
    
    def _replace_url_placeholders(self, url: str) -> str:
        """
        Replace URL placeholders with actual URLs.

        Args:
            url: URL string potentially containing placeholders like __GITLAB__

        Returns:
            URL with placeholders replaced

        Note:
            The execution-layer url_mapping intentionally appends a landing-page
            suffix to ``__GITLAB__`` (``.../explore``) so the agent starts on the
            GitLab explore page. WebArena's reference URLs, however, follow the
            canonical convention where ``__GITLAB__`` carries **no** path suffix
            (see webarena README: ``GITLAB=<domain>:8023``); e.g.
            ``__GITLAB__/dashboard/todos``. Substituting the execution value
            verbatim would inject a spurious ``/explore`` segment into every
            GitLab reference URL and break ``url_match``. We therefore strip the
            ``/explore`` landing suffix here, in the evaluation layer only, so
            URL matching uses the canonical base while execution keeps its
            landing page untouched. ``__SHOPPING_ADMIN__`` is left as-is because
            its ``/admin`` suffix is part of the canonical WebArena convention.
        """
        if not url or not self.url_mapping:
            return url

        result = url
        for placeholder, actual_url in self.url_mapping.items():
            if placeholder not in result:
                continue
            eval_url = actual_url
            if placeholder == "__GITLAB__":
                eval_url = eval_url.rstrip("/")
                if eval_url.endswith("/explore"):
                    eval_url = eval_url[: -len("/explore")]
            result = result.replace(placeholder, eval_url)
        return result
    
    def __call__(self) -> float:
        """Execute evaluation. Override in subclasses."""
        raise NotImplementedError
    
    def llm_fuzzy_match(self, pred: str, reference: str, question: str) -> float:
        """
        Check whether the prediction matches the reference with LLM.
        
        Uses semantic equivalence checking with detailed grading.
        
        Args:
            pred: Predicted answer
            reference: Reference answer
            question: The original question/task
            
        Returns:
            Score (0.0 for incorrect, 1.0 for correct/partially correct)
        """
        if self.llm_client is None:
            logger.warning("LLM client not available for fuzzy match, falling back to exact match")
            return 1.0 if pred.lower().strip() == reference.lower().strip() else 0.0
        
        message = f"""You are grading a student's answer for semantic equivalence.

Task:
Decide if the student's answer is (a) semantically equivalent to the reference answer, or (b) fully contains the reference answer with no contradictions. Ignore wording, phrasing, order, and surface-level differences. Focus only on meaning.

Special rule for "N/A":
- The string "N/A" means "not achievable".
- If and only if the reference answer is "N/A", then a correct student answer can be either "N/A" or a clear, valid reason why the task is not achievable.
- If the reference is not "N/A", then a student answer of "N/A" (or excuses) is incorrect.

Judgment labels (choose exactly one):
- correct
- partially correct
- incorrect

Guidelines:
- correct: Student's answer matches the reference meaning or fully contains all essential information from the reference without introducing contradictions.
- partially correct: Student's answer captures some but not all essential information, or mixes correct info with omissions/uncertainties (no hard contradiction with the reference).
- incorrect: Student's answer conflicts with the reference, misses essential information entirely, or relies on excuses when the reference is not "N/A".

Edge cases:
- Minor numeric rounding (e.g., 3.14 vs 3.1415) → acceptable.
- Additional, non-contradictory details → allowed.
- Any contradiction of key facts in the reference → incorrect.

Now grade the student's answer for this item:

question={question}
reference answer={reference}
student answer={pred}

Respond in the following json format:
{{
    "analysis": "step-by-step analysis of the grading process, explaining why you make the judgment",
    "judgment": "exactly one of 'correct', 'partially correct', or 'incorrect'"
}}
RETURN ONLY THE DICTIONARY I ASKED FOR WITHOUT ANY COMMENTARY."""
        
        messages = [{"role": "user", "content": message}]
        
        try:
            response = self.llm_client.complete(
                user=messages[-1]["content"],
                json_mode=True,
                temperature=self.llm_eval_config.get("temperature"),
                max_tokens=self.llm_eval_config.get("max_tokens"),
            )
            
            if response is None:
                logger.error("LLM fuzzy match got None response")
                return 0.0
            
            response_text = response.lower() if isinstance(response, str) else str(response).lower()
            
            # Try to parse as JSON
            response_data = robust_json_loads(response_text)
            
            if response_data and 'judgment' in response_data:
                judgment = response_data['judgment'].lower()
            else:
                judgment = response_text
            
            if "incorrect" in judgment:
                score = 0.0
            elif 'partially correct' in judgment or 'correct' in judgment:
                score = 1.0
            else:
                logger.error(f"Unexpected LLM fuzzy match response: {judgment}")
                score = 0.0
            
            logger.debug(
                f"LLM fuzzy match: score={score}, "
                f"question={question[:50]}..., pred={pred[:50]}..., ref={reference[:50]}..."
            )
            return score
            
        except Exception as e:
            logger.error(f"LLM fuzzy match failed: {e}")
            return 0.0
    
    def llm_ua_match(self, pred: str, reference: str, question: str) -> float:
        """
        Check unachievable task reason matching with LLM.
        
        Args:
            pred: Reported unachievable reason
            reference: Actual unachievable reason
            question: The original task
            
        Returns:
            Score (0.0 for different, 1.0 for same)
        """
        if self.llm_client is None:
            logger.warning("LLM client not available for UA match")
            return 0.0
        
        message = f"""task: {question}
actual unachievable reason: {reference}
reported unachievable reason: {pred}

The task described above is inherently unachievable due to the reason specified under 'actual unachievable reason'. An individual previously attempted this task and was unable to complete it. They provided a reason for their failure, which is listed under 'reported unachievable reason'. Your role is to review both the actual and reported reasons. Determine if the reported reason aligns with the actual reason, even if implicitly. If the stated reason is in line with the actual reason, respond with 'same'. Otherwise, respond with 'different'."""
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": message},
        ]
        
        try:
            response = self.llm_client.complete(
                system=messages[0]["content"],
                user=messages[1]["content"],
                temperature=self.llm_eval_config.get("temperature"),
                max_tokens=self.llm_eval_config.get("max_tokens"),
            )
            
            if response is None:
                return 0.0
            
            response_text = response.lower() if isinstance(response, str) else str(response).lower()
            
            if "different" in response_text:
                return 0.0
            elif "same" in response_text:
                return 1.0
            else:
                logger.error(f"Unexpected LLM UA match response: {response_text}")
                return 0.0
                
        except Exception as e:
            logger.error(f"LLM UA match failed: {e}")
            return 0.0


# ==============================================================================
# String Evaluator
# ==============================================================================

class StringEvaluator(Evaluator):
    """
    Check whether the answer is correct with:
    - exact_match: answer is exactly the same as reference
    - must_include: each phrase in reference must be in answer
    - fuzzy_match: answer is semantically similar (LLM-based)
    """
    
    @staticmethod
    def clean_answer(answer: str) -> str:
        """Clean and normalize answer string."""
        answer = str(answer).strip()
        if answer.startswith("'") and answer.endswith("'"):
            answer = answer[1:-1]
        elif answer.startswith('"') and answer.endswith('"'):
            answer = answer[1:-1]
        return answer.lower()
    
    def exact_match(self, ref: str, pred: str, question: Optional[str] = None) -> float:
        """
        Exact match with optional LLM fallback.
        
        Args:
            ref: Reference answer
            pred: Predicted answer
            question: Optional question for LLM fallback
            
        Returns:
            Score (0.0 or 1.0)
        """
        if self.clean_answer(pred) == self.clean_answer(ref):
            return 1.0
        elif question is not None:
            # Fall back to fuzzy match
            return self.llm_fuzzy_match(reference=ref, pred=pred, question=question)
        else:
            return 0.0
    
    @staticmethod
    def must_include(ref: str, pred: str, tokenize: bool = False) -> float:
        """
        Check if reference is included in prediction.
        
        Args:
            ref: Reference phrase to find
            pred: Predicted answer
            tokenize: Whether to tokenize for single-word matching
            
        Returns:
            Score (0.0 or 1.0)
        """
        clean_ref = StringEvaluator.clean_answer(ref)
        clean_pred = StringEvaluator.clean_answer(pred)
        
        # Tokenize if ref is a single word to prevent false positives
        if tokenize and len(clean_ref) == 1 and len(word_tokenize(clean_ref)) == 1:
            tok_pred = word_tokenize(clean_pred)
            return float(clean_ref in tok_pred)
        else:
            return float(clean_ref in clean_pred)
    
    def __call__(self) -> float:
        """Execute string evaluation."""
        pred = self.clean_answer(self.last_action_summary)
        question = self.task
        
        score = 1.0
        reference_answers = self.eval_config.get("reference_answers", {})
        
        for approach, value in reference_answers.items():
            if approach == "exact_match":
                score *= self.exact_match(ref=value, pred=pred, question=question)
                
            elif approach == "must_include":
                if isinstance(value, list):
                    for must_value in value:
                        score *= self.must_include(
                            ref=must_value,
                            pred=pred,
                            tokenize=(len(value) == 1),
                        )
                else:
                    score *= self.must_include(ref=value, pred=pred)
                    
            elif approach == "fuzzy_match":
                if value == "N/A":
                    # Task is unachievable. Match WebArena upstream exactly: use a
                    # PLAIN exact match here (no `question` → no LLM fuzzy fallback).
                    # Passing `question` would make exact_match fall back to
                    # llm_fuzzy_match, which can wrongly accept non-"N/A" answers
                    # as equivalent (false positive on unachievable tasks).
                    score *= self.exact_match(ref=value, pred=pred)
                    if score != 1:
                        # Try UA match with string_note
                        string_note = self.eval_config.get("string_note", "")
                        if string_note:
                            score = 1.0 * self.llm_ua_match(
                                pred=pred,
                                reference=string_note,
                                question=self.task,
                            )
                else:
                    if isinstance(value, list):
                        for reference in value:
                            score *= self.llm_fuzzy_match(
                                reference=reference, pred=pred, question=question
                            )
                    else:
                        score *= self.llm_fuzzy_match(
                            reference=value, pred=pred, question=question
                        )
        
        return score


# ==============================================================================
# URL Evaluator
# ==============================================================================

class URLEvaluator(Evaluator):
    """Check URL matching."""
    
    def __call__(self) -> float:
        """Execute URL evaluation."""
        
        def clean_url(url: str) -> str:
            return str(url).rstrip("/")
        
        def parse_url(url: str) -> tuple:
            """Parse URL into base path and query components."""
            parsed_url = urllib.parse.urlparse(url)
            base_path = parsed_url.netloc + parsed_url.path
            query = urllib.parse.parse_qs(parsed_url.query)
            return base_path, query
        
        def parse_urls(urls: List[str]) -> tuple:
            """Parse a list of URLs."""
            base_paths = []
            queries = collections.defaultdict(set)
            for url in urls:
                base_path, query = parse_url(url)
                base_paths.append(base_path)
                for k, v in query.items():
                    queries[k].update(v)
            return base_paths, queries
        
        pred = clean_url(self.page.url)
        reference_url = self.eval_config.get("reference_url", "")
        # Replace URL placeholders (e.g., __GITLAB__ -> http://localhost:8023)
        reference_url = self._replace_url_placeholders(reference_url)
        ref_urls = reference_url.split(" |OR| ")
        ref_urls = [clean_url(url) for url in ref_urls]
        
        matching_rule = self.eval_config.get("url_note", "GOLD in PRED")
        
        if matching_rule == "GOLD in PRED":
            ref_base_paths, ref_queries = parse_urls(ref_urls)
            pred_base_path, pred_query = parse_url(pred)
            
            # Check base path
            base_score = float(
                any(ref_base_path in pred_base_path for ref_base_path in ref_base_paths)
            )
            
            # Check query parameters
            query_score = 1.0
            for k, possible_values in ref_queries.items():
                query_score *= float(
                    any(
                        possible_ref_value in pred_query.get(k, [])
                        for possible_ref_value in possible_values
                    )
                )
            
            score = base_score * query_score
        else:
            logger.warning(f"Unknown URL matching rule: {matching_rule}")
            score = 0.0
        
        return score


# ==============================================================================
# HTML Content Evaluator
# ==============================================================================

class HTMLContentEvaluator(Evaluator):
    """Check whether specific contents appear in the page."""
    
    def __call__(self) -> float:
        """Execute HTML content evaluation."""
        targets = self.eval_config.get("program_html", [])
        
        score = 1.0
        for target in targets:
            target_url = target.get("url", "last")
            
            # Handle function-based URL
            if target_url.startswith("func"):
                func = target_url.split("func:")[1]
                func = func.replace("__last_url__", self.page.url)
                try:
                    target_url = eval(func)
                except Exception as e:
                    logger.error(f"Failed to evaluate URL function: {e}")
                    continue
            
            # Replace URL placeholders (e.g., __GITLAB__ -> http://localhost:8023)
            target_url = self._replace_url_placeholders(target_url)
            
            locator = target.get("locator", "")
            
            # Navigate to target URL if not 'last'
            if target_url != "last":
                try:
                    self.page.goto(target_url)
                    time.sleep(3)  # Wait for page load
                except Exception as e:
                    logger.error(f"Failed to navigate to {target_url}: {e}")
                    score *= 0.0
                    continue
            
            # Get element content
            selected_element = ""
            
            if not locator.strip():
                # Use full page content
                selected_element = self.page.content()
            elif locator.startswith("document.") or locator.startswith("[...document."):
                # JavaScript locator
                if "prep_actions" in target:
                    try:
                        for prep_action in target["prep_actions"]:
                            self.page.evaluate(f"() => {prep_action}")
                    except Exception:
                        pass
                
                try:
                    selected_element = str(self.page.evaluate(f"() => {locator}"))
                    if not selected_element:
                        selected_element = ""
                except Exception:
                    selected_element = ""
            elif locator.startswith("func:"):
                # Helper function
                page = self.page
                func = locator.split("func:")[1]
                func = func.replace("__page__", "page")
                try:
                    selected_element = eval(func)
                except Exception as e:
                    logger.error(f"Failed to evaluate locator function: {e}")
                    selected_element = ""
            else:
                logger.warning(f"Unknown locator: {locator}")
                continue
            
            # Unescape HTML entities
            selected_element = html.unescape(str(selected_element))
            
            # Check required contents
            required_contents = target.get("required_contents", {})
            
            if "exact_match" in required_contents:
                required = required_contents["exact_match"]
                cur_score = self.exact_match_helper(ref=required, pred=selected_element)
                score *= float(cur_score)
                
            elif "must_include" in required_contents:
                required = required_contents["must_include"]
                if isinstance(required, list):
                    for content in required:
                        content_or = content.split(" |OR| ")
                        cur_score = any(
                            StringEvaluator.must_include(ref=c, pred=selected_element, tokenize=False)
                            for c in content_or
                        )
                        score *= float(cur_score)
                else:
                    score *= float(StringEvaluator.must_include(ref=required, pred=selected_element))
            else:
                logger.warning(f"Unknown required_contents: {required_contents.keys()}")
        
        return score
    
    def exact_match_helper(self, ref: str, pred: str) -> float:
        """Helper for exact match without LLM fallback."""
        return 1.0 if StringEvaluator.clean_answer(pred) == StringEvaluator.clean_answer(ref) else 0.0


# ==============================================================================
# Evaluator Combiner
# ==============================================================================

class EvaluatorComb:
    """Combine multiple evaluators."""
    
    def __init__(self, evaluators: List[Evaluator]):
        self.evaluators = evaluators
    
    def __call__(self) -> int:
        """Execute all evaluators and combine scores."""
        score = 1.0
        for evaluator in self.evaluators:
            cur_score = evaluator()
            score *= cur_score
        return int(score)


# ==============================================================================
# Evaluator Router
# ==============================================================================

def evaluator_router(
    eval_config: Dict[str, Any],
    task: str,
    last_action_summary: str,
    page: Any,
    llm_client: Any = None,
    llm_eval_config: Optional[Dict[str, Any]] = None,
    url_mapping: Optional[Dict[str, str]] = None,
) -> EvaluatorComb:
    """
    Router to get the appropriate evaluator(s).
    
    Args:
        eval_config: Evaluation configuration
        task: Task description
        last_action_summary: Summary of last action
        page: Playwright page object
        llm_client: LLM client for fuzzy matching
        llm_eval_config: LLM configuration
        url_mapping: Mapping of URL placeholders to actual URLs
        
    Returns:
        EvaluatorComb containing all relevant evaluators
    """
    eval_types = eval_config.get('eval_types', [])
    
    params = {
        'eval_config': eval_config,
        'task': task,
        'last_action_summary': last_action_summary,
        'page': page,
        'llm_client': llm_client,
        'llm_eval_config': llm_eval_config,
        'url_mapping': url_mapping,
    }
    
    evaluators: List[Evaluator] = []
    
    for eval_type in eval_types:
        if eval_type == "string_match":
            evaluators.append(StringEvaluator(**params))
        elif eval_type == "url_match":
            evaluators.append(URLEvaluator(**params))
        elif eval_type == "program_html":
            evaluators.append(HTMLContentEvaluator(**params))
        else:
            logger.warning(f"eval_type {eval_type} is not supported")
    
    return EvaluatorComb(evaluators)


# ==============================================================================
# Task Evaluator (High-level Interface)
# ==============================================================================

class TaskEvaluator:
    """
    High-level evaluator wrapper for task verification.
    
    Provides a simple interface to evaluate task completion
    using the appropriate evaluator based on eval_config.
    """
    
    def __init__(
        self,
        llm_client: Any = None,
        llm_eval_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize task evaluator.
        
        Args:
            llm_client: LLM client for fuzzy matching
            llm_eval_config: LLM configuration for evaluation
        """
        self.llm_client = llm_client
        self.llm_eval_config = llm_eval_config or {
            "model": "gpt-4o",
            "temperature": 0.0,
            "max_tokens": 768,
        }
    
    def evaluate(
        self,
        task_dict: Dict[str, Any],
        high_level_task: HighLevelTask,
        page: Any,
        url_mapping: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate task completion.
        
        Args:
            task_dict: Task configuration with 'eval' field
            high_level_task: Executed high-level task with trajectories
            page: Playwright page object
            url_mapping: Mapping of URL placeholders to actual URLs
            
        Returns:
            Dict with 'auto_eval' status and details
        """
        result = {
            'auto_eval': 'NA',
            'score': None,
            'error': None,
        }
        
        # Check if eval config exists
        eval_config = task_dict.get('eval')
        if not eval_config or not isinstance(eval_config, dict):
            logger.debug(f"No eval config found for task: {task_dict.get('task', 'unknown')[:50]}")
            return result
        
        try:
            # Get last action summary
            last_action_summary = self._get_last_action_summary(high_level_task)
            task = task_dict.get('task', '')
            
            # Get evaluator
            evaluator_comb = evaluator_router(
                eval_config=eval_config,
                task=task,
                last_action_summary=last_action_summary,
                page=page,
                llm_client=self.llm_client,
                llm_eval_config=self.llm_eval_config,
                url_mapping=url_mapping,
            )
            
            # Execute evaluation
            score = evaluator_comb()
            
            result['score'] = score
            if score == 0:
                result['auto_eval'] = 'failed'
            elif score == 1:
                result['auto_eval'] = 'success'
            else:
                result['auto_eval'] = 'partial'
            
            logger.info(f"Auto-eval result: {result['auto_eval']} (score={score})")
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            result['error'] = str(e)
        
        return result
    
    def _get_last_action_summary(self, high_level_task: HighLevelTask) -> str:
        """Extract last action summary from trajectory."""
        if not high_level_task.trajectories:
            return "No actions executed"
        
        last_task = high_level_task.trajectories[-1]
        
        # Try to get summary from NONE/STOP action value
        if last_task.action and last_task.action.action_type in {ActionType.NONE, ActionType.STOP}:
            if last_task.action.value:
                return last_task.action.value
        
        # Fall back to state summary
        if last_task.state_after and last_task.state_after.summary:
            return last_task.state_after.summary
        
        if last_task.curr_state and last_task.curr_state.summary:
            return last_task.curr_state.summary
        
        # Fall back to task description
        if last_task.task:
            return last_task.task
        
        return "No summary available"


# ==============================================================================
# Convenience Function
# ==============================================================================

def evaluate_task_result(
    task_dict: Dict[str, Any],
    high_level_task: HighLevelTask,
    page: Any,
    llm_client: Any = None,
    llm_eval_config: Optional[Dict[str, Any]] = None,
    url_mapping: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Convenience function for task evaluation.
    
    Args:
        task_dict: Task configuration with 'eval' field
        high_level_task: Executed task
        page: Playwright page
        llm_client: Optional LLM client for fuzzy matching
        llm_eval_config: Optional LLM configuration
        url_mapping: Mapping of URL placeholders to actual URLs
        
    Returns:
        Evaluation result dict
    """
    evaluator = TaskEvaluator(
        llm_client=llm_client,
        llm_eval_config=llm_eval_config,
    )
    return evaluator.evaluate(task_dict, high_level_task, page, url_mapping=url_mapping)

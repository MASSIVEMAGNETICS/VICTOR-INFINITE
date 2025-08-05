# FILE: VictorSwarm_v4_0_0_TITANCORE_ADAPTIVEGRID_main.py
# VERSION: v4.0.0-TITANCORE-ADAPTIVEGRID
# NAME: VictorSwarm Orchestrator
# AUTHOR: Brandon "iambandobandz" Emery x Victor x SCOS-E x Gemini Enhancer
# PURPOSE: Highly enhanced headless GPT swarm orchestrator with external config/workflow,
#          advanced error handling, self-healing attempt, state management, and robust
#          browser interaction for complex, multi-step task execution via MoE patterns.
# LICENSE: Proprietary - Massive Magnetics / Ethica AI / BHeard Network

import asyncio
import logging
import json
import time
import os
import sys
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple, Set
from playwright.async_api import (
    async_playwright,
    Error as PlaywrightError,
    Page,
    Browser,
    BrowserContext,
    TimeoutError as PlaywrightTimeoutError,
    Playwright,
)

# --- Global Variables ---
CONFIG: Dict[str, Any] = {}
WORKFLOW: List[Dict[str, Any]] = []
SELECTORS: Dict[str, str] = {}
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Configuration Loading ---

def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """Loads configuration from a JSON file."""
    path = os.path.join(SCRIPT_DIR, config_path)
    try:
        with open(path, 'r') as f:
            config_data = json.load(f)
        # Basic validation (presence of key sections)
        required_sections = ["general", "playwright", "swarm", "selectors"]
        if not all(section in config_data for section in required_sections):
            raise ValueError(f"Config file missing required sections: {required_sections}")
        logging.info(f"✅ Configuration loaded successfully from {path}")
        return config_data
    except FileNotFoundError:
        logging.critical(f"🔥 CRITICAL: Configuration file not found at {path}. Exiting.")
        sys.exit(1)
    except json.JSONDecodeError:
        logging.critical(f"🔥 CRITICAL: Failed to parse JSON from config file {path}. Exiting.")
        sys.exit(1)
    except ValueError as e:
        logging.critical(f"🔥 CRITICAL: Invalid config data in {path}: {e}. Exiting.")
        sys.exit(1)
    except Exception as e:
        logging.critical(f"🔥 CRITICAL: Unexpected error loading config {path}: {e}. Exiting.", exc_info=True)
        sys.exit(1)


def load_workflow(workflow_path: str = "workflow.json") -> List[Dict[str, Any]]:
    """Loads workflow definition from a JSON file."""
    path = os.path.join(SCRIPT_DIR, workflow_path)
    try:
        with open(path, 'r') as f:
            workflow_data = json.load(f)
        if not isinstance(workflow_data, list):
             raise ValueError("Workflow file should contain a JSON list of task objects.")
        # Basic validation of tasks
        for i, task in enumerate(workflow_data):
            required_keys = ["step", "task_id", "role", "prompt", "output_key"]
            if not all(key in task for key in required_keys):
                raise ValueError(f"Task #{i+1} in workflow is missing required keys: {required_keys}. Task: {task}")
        logging.info(f"✅ Workflow definition loaded successfully from {path}")
        return workflow_data
    except FileNotFoundError:
        logging.critical(f"🔥 CRITICAL: Workflow file not found at {path}. Exiting.")
        sys.exit(1)
    except json.JSONDecodeError:
        logging.critical(f"🔥 CRITICAL: Failed to parse JSON from workflow file {path}. Exiting.")
        sys.exit(1)
    except ValueError as e:
        logging.critical(f"🔥 CRITICAL: Invalid workflow data in {path}: {e}. Exiting.")
        sys.exit(1)
    except Exception as e:
        logging.critical(f"🔥 CRITICAL: Unexpected error loading workflow {path}: {e}. Exiting.", exc_info=True)
        sys.exit(1)


# --- Enums and Helper Classes ---

class AgentState(Enum):
    """Represents the detailed current state of a swarm agent."""
    OFFLINE = 0
    INITIALIZING = 1
    IDLE = 2          # Ready for task
    SENDING_PROMPT = 3
    PROCESSING = 4    # Waiting for GPT generation
    FETCHING_RESPONSE = 5
    RECOVERING = 6    # Attempting self-healing
    ERROR_INIT = 7    # Failed initialization permanently
    ERROR_RUNTIME = 8 # Failed during operation (might be recoverable)
    CLOSED = 9        # Page explicitly closed

class SwarmAgent:
    """Represents a single GPT instance (browser tab) in the swarm."""
    def __init__(self, role: str, context: BrowserContext, agent_id: int, config: Dict[str, Any]):
        self.role = role
        self.agent_id = agent_id
        self.context = context
        self.config = config # Store relevant config sections
        self.swarm_config = config['swarm']
        self.playwright_config = config['playwright']
        self.selectors = config['selectors']
        self.page: Optional[Page] = None
        self.state: AgentState = AgentState.OFFLINE
        self.last_error: Optional[str] = None
        self.logger = logging.getLogger(f"Agent-{self.role}-{self.agent_id}")
        self.message_count = 0 # Track messages to help find the latest response

    async def initialize(self, attempt_count: int = 1) -> bool:
        """Initializes the agent by opening a new page and navigating. Includes retries."""
        max_retries = self.swarm_config.get('agent_init_retries', 3)
        retry_delay = self.swarm_config.get('agent_init_retry_delay', 5)
        nav_timeout = self.playwright_config.get('navigation_timeout', 30000)
        action_timeout = self.playwright_config.get('action_timeout', 20000)
        openai_url = self.config['general'].get('openai_url', 'https://chat.openai.com')
        textarea_selector = self.selectors.get('textarea', "textarea[id='prompt-textarea']")
        login_indicator_selector = self.selectors.get('login_indicator') # Selector for login button/link

        self.state = AgentState.INITIALIZING
        self.logger.info(f"Initializing (Attempt {attempt_count}/{max_retries})...")

        try:
            if self.page and not self.page.is_closed():
                await self.page.close() # Ensure previous attempt's page is closed
            self.page = await self.context.new_page()
            self.logger.debug(f"Navigating to {openai_url}")
            await self.page.goto(openai_url, timeout=nav_timeout, wait_until="domcontentloaded")

            # Check for login page indicator - cookies might be invalid
            if login_indicator_selector:
                login_element = self.page.locator(login_indicator_selector).first
                try:
                    # Brief wait to see if login element appears
                    await login_element.wait_for(timeout=3000, state="visible")
                    self.logger.error("❌ Login page detected. Cookies might be invalid or expired.")
                    self.state = AgentState.ERROR_INIT
                    self.last_error = "Initialization failed: Login page detected (invalid cookies?)."
                    await self.close() # Close the page if login detected
                    return False
                except PlaywrightTimeoutError:
                    self.logger.debug("Login indicator not found, proceeding.")
                    pass # Expected case - login indicator not present

            # Wait for the main text area to be visible as a sign of page readiness
            await self.page.wait_for_selector(textarea_selector, timeout=action_timeout, state="visible")
            self.state = AgentState.IDLE
            self.last_error = None
            self.logger.info("✅ Initialized successfully.")
            return True

        except (PlaywrightError, PlaywrightTimeoutError) as e:
            self.logger.warning(f"⚠️ Initialization attempt {attempt_count}/{max_retries} failed: {type(e).__name__} - {e}")
            if self.page and not self.page.is_closed():
                await self.page.close()
                self.page = None
            if attempt_count < max_retries:
                await asyncio.sleep(retry_delay)
                return await self.initialize(attempt_count + 1) # Recursive call for retry
            else:
                self.state = AgentState.ERROR_INIT
                self.last_error = f"Failed to initialize after {max_retries} attempts."
                self.logger.error(f"❌ {self.last_error}")
                return False
        except Exception as e: # Catch unexpected errors
            self.logger.error(f"💥 Unexpected error during initialization attempt {attempt_count}: {e}", exc_info=True)
            if self.page and not self.page.is_closed():
                await self.page.close()
                self.page = None
            self.state = AgentState.ERROR_INIT
            self.last_error = f"Unexpected initialization error: {e}"
            return False

    async def _robust_action(self, action_name: str, coro: asyncio.coroutine, retries: int, delay: int, timeout: int) -> Any:
        """Wrapper for playwright actions with configurable retries and timeout."""
        for attempt in range(retries + 1):
            try:
                # Use asyncio.wait_for for timeout control over the action
                return await asyncio.wait_for(coro, timeout=timeout / 1000)
            except (PlaywrightError, PlaywrightTimeoutError, asyncio.TimeoutError) as e:
                self.logger.warning(f"⚠️ Action '{action_name}' attempt {attempt+1}/{retries+1} failed: {type(e).__name__} - {e}")
                if attempt < retries:
                    await asyncio.sleep(delay)
                else:
                    self.state = AgentState.ERROR_RUNTIME # Mark as runtime error
                    self.last_error = f"Action '{action_name}' failed after {retries+1} attempts: {type(e).__name__}"
                    self.logger.error(f"❌ {self.last_error}")
                    raise # Re-raise the specific exception after final attempt
            except Exception as e: # Catch other unexpected errors during the action
                self.state = AgentState.ERROR_RUNTIME
                self.last_error = f"Unexpected error during action '{action_name}': {e}"
                self.logger.error(f"💥 {self.last_error}", exc_info=True)
                raise

    async def send_prompt(self, prompt: str) -> bool:
        """Sends a prompt to the agent's chat interface with robustness."""
        if self.state != AgentState.IDLE or not self.page or self.page.is_closed():
            self.logger.error(f"❌ Cannot send prompt in state {self.state.name} or page closed/missing.")
            self.state = AgentState.ERROR_RUNTIME # Consider this a runtime error
            self.last_error = "Attempted to send prompt while not idle or page invalid."
            return False

        self.state = AgentState.SENDING_PROMPT
        self.logger.info(f"📨 Sending prompt...")
        self.logger.debug(f"Prompt content (truncated): {prompt[:150]}...")

        action_retries = self.swarm_config.get('agent_action_retries', 2)
        action_delay = self.swarm_config.get('agent_action_retry_delay', 3)
        action_timeout = self.playwright_config.get('action_timeout', 20000)
        textarea_selector = self.selectors.get('textarea', "textarea[id='prompt-textarea']")
        send_button_selector = self.selectors.get('send_button', "button[data-testid='send-button']")

        try:
            # 1. Find the text area
            textarea = self.page.locator(textarea_selector).first
            await self._robust_action("wait_textarea", textarea.wait_for(state="visible"), action_retries, action_delay, action_timeout)

            # 2. Fill the text area
            await self._robust_action("fill_textarea", textarea.fill(prompt), action_retries, action_delay, action_timeout)

            # 3. Find the send button and ensure it's enabled
            send_button = self.page.locator(send_button_selector).first
            await self._robust_action("wait_send_enabled", send_button.wait_for(state="enabled"), action_retries, action_delay, action_timeout)

            # 4. Click the send button
            await self._robust_action("click_send_button", send_button.click(), action_retries, action_delay, action_timeout)

            self.message_count += 2 # User prompt + Assistant response expected
            self.state = AgentState.PROCESSING # Transition state after successful send
            self.logger.info("✅ Prompt sent successfully.")
            return True

        except (PlaywrightError, PlaywrightTimeoutError, asyncio.TimeoutError, Exception) as e:
            # Errors logged and state set within _robust_action or caught here
            self.logger.error(f"❌ Failed to send prompt: {self.last_error or e}")
            # Attempt self-healing if enabled and the error is a runtime error
            await self.attempt_self_heal("send_prompt_failure")
            return False # Return False as sending failed

    async def get_response(self) -> Optional[str]:
        """Waits for and retrieves the latest response from the agent with robustness."""
        if self.state != AgentState.PROCESSING or not self.page or self.page.is_closed():
            self.logger.error(f"❌ Cannot get response in state {self.state.name} or page closed/missing.")
            self.state = AgentState.ERROR_RUNTIME
            self.last_error = "Attempted to get response while not processing or page invalid."
            return None # Indicate failure

        self.state = AgentState.FETCHING_RESPONSE
        self.logger.info("⏳ Waiting for response generation...")

        start_time = time.time()
        max_wait = self.playwright_config.get('max_response_wait_time', 240)
        poll_interval = self.playwright_config.get('response_poll_interval', 1.5)
        action_timeout = self.playwright_config.get('action_timeout', 20000) # Timeout for intermediate checks
        generating_indicator_selector = self.selectors.get('generating_indicator', "button[aria-label='Stop generating']")
        send_button_selector = self.selectors.get('send_button', "button[data-testid='send-button']")
        message_block_selector = self.selectors.get('message_block', "div[data-message-author-role]")
        response_content_selector = self.selectors.get('assistant_response_content', ".markdown")

        try:
            # --- Wait for generation to START ---
            # Check if the generating indicator appears OR the send button becomes disabled
            await asyncio.wait_for(
                 self.page.wait_for_function(
                    f"""
                    () => {{
                        const genIndicator = document.querySelector('{generating_indicator_selector}');
                        const sendButton = document.querySelector('{send_button_selector}');
                        return (genIndicator !== null) || (sendButton && sendButton.disabled);
                    }}
                    """,
                    timeout=action_timeout, # Timeout for generation to START
                    polling=500 # ms check frequency
                 ),
                 timeout=(action_timeout / 1000) + 5 # Add buffer to wait_for timeout
            )
            self.logger.debug("Generation start detected.")

            # --- Wait for generation to FINISH ---
            # Check if the generating indicator disappears AND the send button becomes enabled
            await asyncio.wait_for(
                self.page.wait_for_function(
                     f"""
                    () => {{
                        const genIndicator = document.querySelector('{generating_indicator_selector}');
                        const sendButton = document.querySelector('{send_button_selector}');
                        const sendButtonEnabled = sendButton && !sendButton.disabled;
                        const indicatorGone = genIndicator === null;
                        return indicatorGone && sendButtonEnabled;
                    }}
                    """,
                    timeout=max_wait * 1000, # Long timeout for generation itself (in ms)
                    polling=poll_interval * 1000 # ms check frequency
                ),
                timeout=max_wait + 10 # Add buffer to wait_for timeout
            )
            elapsed_gen = time.time() - start_time
            self.logger.debug(f"Generation complete signal received after {elapsed_gen:.2f}s.")
            await asyncio.sleep(1.5) # Increased buffer for UI to settle completely

            # --- Retrieve the latest assistant response ---
            self.logger.debug("Attempting to retrieve response content...")
            # Find all message blocks, filter for assistant, take the last one.
            all_message_blocks = self.page.locator(message_block_selector)
            # Count elements first to avoid waiting for timeout if none exist
            count = await all_message_blocks.count()
            if count == 0:
                 raise ValueError(f"No message blocks found using selector: '{message_block_selector}'")

            # Get all blocks and filter in Python (might be slightly slower but more robust)
            all_blocks_list = await all_message_blocks.all()
            assistant_blocks = []
            for block in all_blocks_list:
                role = await block.get_attribute('data-message-author-role')
                if role == 'assistant':
                    assistant_blocks.append(block)

            if not assistant_blocks:
                raise ValueError("No assistant message blocks found on the page.")

            latest_assistant_message = assistant_blocks[-1]
            self.logger.debug(f"Located latest assistant block. Expected message index: {self.message_count -1}") # Debugging index

            # Extract text from the response content element(s) within the message block
            response_elements = latest_assistant_message.locator(response_content_selector)
            response_elements_count = await response_elements.count()
            if response_elements_count == 0:
                 # Fallback: Try getting all text content of the block if specific selector fails
                 self.logger.warning(f"Could not find response content with selector '{response_content_selector}'. Falling back to full block text.")
                 response_text = await latest_assistant_message.inner_text()
                 if not response_text:
                     raise ValueError(f"Could not extract any text from the last assistant message block (fallback).")
            else:
                # Get text from potentially multiple matching elements and join
                response_parts = await response_elements.all_inner_texts()
                response_text = "\n".join(part.strip() for part in response_parts if part.strip()).strip()
                if not response_text:
                     raise ValueError(f"Extracted empty text using selector '{response_content_selector}' from the last assistant message.")

            elapsed_total = time.time() - start_time
            self.logger.info(f"📥 Response received successfully in {elapsed_total:.2f}s.")
            self.state = AgentState.IDLE # Ready for next task
            self.last_error = None
            return response_text

        except (PlaywrightTimeoutError, asyncio.TimeoutError) as e:
            self.state = AgentState.ERROR_RUNTIME
            error_msg = f"Timeout waiting for response generation/signal ({max_wait}s): {type(e).__name__}"
            self.last_error = error_msg
            self.logger.error(f"❌ {error_msg}")
            await self.save_debug_snapshot("timeout_error")
            await self.attempt_self_heal("response_timeout")
            return None # Indicate failure
        except (PlaywrightError, ValueError, Exception) as e:
            self.state = AgentState.ERROR_RUNTIME
            error_msg = f"Failed to get/parse response: {type(e).__name__} - {e}"
            self.last_error = error_msg
            self.logger.error(f"❌ {error_msg}", exc_info=True)
            await self.save_debug_snapshot("response_error")
            await self.attempt_self_heal("response_parsing_error")
            return None # Indicate failure

    async def attempt_self_heal(self, failure_context: str):
        """Attempts to re-initialize the agent if it encounters a runtime error."""
        if not self.swarm_config.get('enable_self_healing', True):
            self.logger.info("Self-healing disabled, agent remains in error state.")
            return

        if self.state == AgentState.ERROR_RUNTIME:
            self.logger.warning(f"Runtime error detected ({failure_context}). Attempting self-healing...")
            self.state = AgentState.RECOVERING
            if self.page and not self.page.is_closed():
                await self.close(mark_closed=False) # Close current page without setting final state

            # Try re-initializing
            success = await self.initialize(attempt_count=1) # Start re-init attempts from 1
            if success:
                self.logger.info("✅ Self-healing successful. Agent re-initialized.")
                self.state = AgentState.IDLE # Back to ready state
            else:
                self.logger.error("❌ Self-healing failed. Agent remains in error state (ERROR_INIT).")
                # State is already set to ERROR_INIT by the failed initialize call
        else:
            self.logger.debug(f"Agent not in ERROR_RUNTIME state ({self.state.name}), skipping self-heal.")


    async def save_debug_snapshot(self, prefix: str):
        """Saves HTML and screenshot for debugging purposes."""
        if self.page and not self.page.is_closed():
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename_base = f"debug_{prefix}_{self.role}_{self.agent_id}_{timestamp}"
            try:
                html_path = os.path.join(SCRIPT_DIR, f"{filename_base}.html")
                png_path = os.path.join(SCRIPT_DIR, f"{filename_base}.png")
                await self.page.screenshot(path=png_path)
                html_content = await self.page.content()
                with open(html_path, "w", encoding="utf-8") as f:
                    f.write(html_content)
                self.logger.info(f"📸 Saved debug snapshot: {png_path} and {html_path}")
            except Exception as e:
                self.logger.error(f"⚠️ Failed to save debug snapshot: {e}")


    async def close(self, mark_closed=True):
        """Closes the agent's page cleanly."""
        if mark_closed:
             self.state = AgentState.CLOSED
        if self.page and not self.page.is_closed():
            try:
                await self.page.close()
                self.logger.info("Page closed.")
            except PlaywrightError as e:
                self.logger.warning(f"⚠️ Error closing page: {e}")
        self.page = None


# --- Orchestrator ---

class SwarmOrchestrator:
    """Manages the swarm of agents and executes the workflow based on external config."""
    def __init__(self, config: Dict[str, Any], workflow: List[Dict[str, Any]]):
        self.config = config
        self.workflow = workflow
        self.general_config = config['general']
        self.swarm_config = config['swarm']
        self.agents: Dict[str, SwarmAgent] = {} # Role -> Agent instance
        self.playwright: Optional[Playwright] = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.results: Dict[str, Any] = {} # Stores outputs from workflow steps (output_key -> result)
        self.logger = logging.getLogger("Orchestrator")
        self.agent_id_counter = 0

    def _load_cookies(self) -> List[Dict]:
        """Loads cookies from the specified JSON file."""
        cookies_file = self.general_config.get('cookies_file', 'cookies.json')
        path = os.path.join(SCRIPT_DIR, cookies_file)
        try:
            with open(path, 'r') as f:
                loaded_cookies = json.load(f)
            self.logger.info(f"🍪 Loaded {len(loaded_cookies)} cookies from {path}")
            if not isinstance(loaded_cookies, list) or not all(isinstance(c, dict) and 'name' in c and 'value' in c for c in loaded_cookies):
                raise ValueError("Invalid cookie format: Expected a list of objects with 'name' and 'value'.")
            return loaded_cookies
        except FileNotFoundError:
            self.logger.critical(f"🔥 CRITICAL: Cookies file not found at {path}. Exiting.")
            sys.exit(1)
        except json.JSONDecodeError:
            self.logger.critical(f"🔥 CRITICAL: Failed to parse cookies JSON from {path}. Exiting.")
            sys.exit(1)
        except ValueError as e:
            self.logger.critical(f"🔥 CRITICAL: Invalid cookie data in {path}: {e}. Exiting.")
            sys.exit(1)
        except Exception as e:
            self.logger.critical(f"🔥 CRITICAL: Unexpected error loading cookies from {path}: {e}. Exiting.", exc_info=True)
            sys.exit(1)

    async def startup(self):
        """Initializes Playwright, browser, context, and all required agents."""
        self.logger.info("🚀 Orchestrator starting up...")
        required_roles = set(task['role'] for task in self.workflow if 'role' in task)
        self.logger.info(f"Workflow requires roles: {', '.join(required_roles)}")

        try:
            self.playwright = await async_playwright().start()
            headless = self.config['playwright'].get('headless_mode', True)
            launch_timeout = self.config['playwright'].get('browser_launch_timeout', 35000)
            self.logger.info(f"Launching browser (Headless: {headless})...")
            self.browser = await self.playwright.chromium.launch(
                headless=headless,
                timeout=launch_timeout
            )
            self.logger.info("Creating browser context...")
            # Consider adding viewport, user agent etc. from config if needed
            self.context = await self.browser.new_context(
                 # Example: Set viewport if needed for specific site rendering
                 # viewport={'width': 1920, 'height': 1080},
                 # user_agent="Mozilla/5.0..." # Can be set in config
            )
            # Set default timeouts for the context
            nav_timeout = self.config['playwright'].get('navigation_timeout', 30000)
            action_timeout = self.config['playwright'].get('action_timeout', 20000)
            self.context.set_default_navigation_timeout(nav_timeout)
            self.context.set_default_timeout(action_timeout)

            # Load and inject cookies
            cookies = self._load_cookies()
            await self.context.add_cookies(cookies)
            self.logger.info("✅ Cookies injected into context.")

            # Initialize agents for required roles
            self.logger.info(f"Initializing {len(required_roles)} agents...")
            init_tasks = []
            for role in required_roles:
                self.agent_id_counter += 1
                agent = SwarmAgent(role, self.context, self.agent_id_counter, self.config)
                self.agents[role] = agent
                init_tasks.append(agent.initialize()) # Start initialization coroutine

            results = await asyncio.gather(*init_tasks, return_exceptions=True) # Gather results, catch init exceptions

            successful_inits = 0
            for i, result in enumerate(results):
                 role = list(required_roles)[i] # Assuming order is maintained
                 if isinstance(result, Exception):
                      self.logger.error(f"Agent '{role}' initialization failed with exception: {result}")
                      # Agent state should be ERROR_INIT from initialize method
                 elif result is True:
                      successful_inits += 1
                 # else result is False, error already logged by agent

            self.logger.info(f"Agent initialization complete. {successful_inits}/{len(required_roles)} successful.")

            if successful_inits == 0:
                 raise RuntimeError("CRITICAL: No agents could be initialized. Aborting workflow.")

        except (PlaywrightError, PlaywrightTimeoutError, RuntimeError) as e:
            self.logger.critical(f"💥 Orchestrator startup failed critically: {e}", exc_info=True)
            await self.shutdown() # Attempt cleanup
            sys.exit(1) # Exit after critical failure
        except Exception as e: # Catch any other unexpected startup errors
            self.logger.critical(f"💥 Unexpected critical error during startup: {e}", exc_info=True)
            await self.shutdown()
            sys.exit(1)

    async def run_workflow(self):
        """Executes the loaded workflow step by step."""
        self.logger.info("=== Starting Workflow Execution ===")
        # Determine the maximum step number
        max_steps = 0
        if self.workflow:
             try:
                  max_steps = max(task.get("step", 0) for task in self.workflow)
             except ValueError:
                  self.logger.error("Could not determine maximum step number in workflow.")
                  max_steps = 0 # Handle case of empty or malformed workflow steps


        for current_step in range(1, max_steps + 1):
            self.logger.info(f"--- Executing Step {current_step} ---")
            step_tasks_coroutines = []
            tasks_in_this_step = [task for task in self.workflow if task.get("step") == current_step]

            if not tasks_in_this_step:
                self.logger.warning(f"No tasks defined for step {current_step}.")
                continue

            # --- Task Preparation and Validation for the Current Step ---
            prepared_tasks = []
            for task_def in tasks_in_this_step:
                task_id = task_def.get("task_id", f"step{current_step}_task{len(prepared_tasks)+1}")
                role = task_def.get("role")
                prompt_template = task_def.get("prompt")
                input_keys = task_def.get("input_keys", [])
                output_key = task_def.get("output_key")

                # Basic validation
                if not all([role, prompt_template, output_key]):
                    self.logger.error(f"Skipping invalid task definition (missing role, prompt, or output_key) in step {current_step}: {task_def}")
                    self.results[output_key] = f"Error: Invalid task definition for '{task_id}'."
                    continue

                # Check Agent Availability and State
                agent = self.agents.get(role)
                if not agent:
                     self.logger.error(f"Agent role '{role}' required for task '{task_id}' not found or initialized. Skipping.")
                     self.results[output_key] = f"Error: Agent role '{role}' unavailable for task '{task_id}'."
                     continue
                if agent.state not in [AgentState.IDLE, AgentState.ERROR_RUNTIME]: # Allow retry if recoverable error
                    self.logger.warning(f"Agent '{role}' is not IDLE (state: {agent.state.name}). Skipping task '{task_id}' for now.")
                    self.results[output_key] = f"Error: Agent '{role}' was busy or in non-idle state for task '{task_id}'."
                    continue
                # If agent is in ERROR_RUNTIME, self-healing might fix it before execution attempt
                if agent.state == AgentState.ERROR_RUNTIME:
                     self.logger.warning(f"Agent '{role}' is in ERROR_RUNTIME state. Self-healing might be attempted before task '{task_id}'.")
                     # The _execute_task will handle the self-heal attempt

                # Format Prompt
                try:
                    # Create context for formatting, ensuring all keys exist
                    format_context = {key: self.results.get(key, f"{{KEY_NOT_FOUND: {key}}}") for key in input_keys}
                    # Add all previous results for flexibility, preferring specific input_keys
                    format_context.update(self.results)
                    prompt = prompt_template.format(**format_context)
                except KeyError as e:
                    self.logger.error(f"Failed to format prompt for task '{task_id}' (role '{role}'). Missing input key: {e}. Check workflow dependencies.")
                    self.results[output_key] = f"Error: Missing input key {e} for task '{task_id}' prompt."
                    # Mark agent as error? Maybe not, could be workflow logic error.
                    continue # Skip this task

                prepared_tasks.append({"agent": agent, "prompt": prompt, "output_key": output_key, "task_id": task_id})

            # --- Execute Prepared Tasks for the Current Step Concurrently ---
            if prepared_tasks:
                self.logger.info(f"Dispatching {len(prepared_tasks)} tasks for step {current_step}.")
                step_tasks_coroutines = [self._execute_task(**task_info) for task_info in prepared_tasks]
                # Gather results, allowing tasks to complete even if others fail
                task_results = await asyncio.gather(*step_tasks_coroutines, return_exceptions=True)

                # Log any exceptions that occurred during task execution
                for i, result in enumerate(task_results):
                     if isinstance(result, Exception):
                          task_id = prepared_tasks[i]['task_id']
                          self.logger.error(f"Exception occurred during execution of task '{task_id}': {result}", exc_info=result)
                          # Result dict should have been updated with error message inside _execute_task

                self.logger.info(f"--- Finished Step {current_step} ---")
            else:
                self.logger.warning(f"No valid tasks could be prepared or dispatched for step {current_step}.")


        self.logger.info("=== Workflow Execution Finished ===")


    async def _execute_task(self, agent: SwarmAgent, prompt: str, output_key: str, task_id: str):
        """Handles sending prompt and getting response for a single task, including self-heal attempt."""
        self.logger.info(f"Executing task '{task_id}' assigned to agent '{agent.role}' (ID: {agent.agent_id})")

        # Attempt self-healing if agent starts in a recoverable error state
        if agent.state == AgentState.ERROR_RUNTIME:
            await agent.attempt_self_heal(f"pre_task_{task_id}")
            # If self-healing failed, agent state will be ERROR_INIT, caught below

        # Check state again after potential self-heal
        if agent.state != AgentState.IDLE:
             error_msg = f"Agent '{agent.role}' not IDLE (state: {agent.state.name}) before executing task '{task_id}'. Cannot proceed."
             self.logger.error(error_msg)
             self.results[output_key] = f"Error: {error_msg}"
             return # Stop execution for this task

        # Proceed with sending prompt
        sent = await agent.send_prompt(prompt)

        if sent:
            # If prompt sent successfully, try to get the response
            response = await agent.get_response()
            if response is not None:
                # Success case
                self.results[output_key] = response
                self.logger.info(f"✅ Successfully completed task '{task_id}', stored result for '{output_key}'.")
            else:
                # get_response failed, error logged by agent, attempt_self_heal called internally
                self.results[output_key] = agent.last_error or f"Error: Failed to get response for task '{task_id}' from '{agent.role}'"
                self.logger.error(f"Task '{task_id}' failed during response retrieval phase.")
        else:
            # send_prompt failed, error logged by agent, attempt_self_heal called internally
            self.results[output_key] = agent.last_error or f"Error: Failed to send prompt for task '{task_id}' via '{agent.role}'"
            self.logger.error(f"Task '{task_id}' failed during prompt sending phase.")

        # Final state check (should be IDLE or an ERROR state)
        if agent.state not in [AgentState.IDLE, AgentState.ERROR_INIT, AgentState.ERROR_RUNTIME, AgentState.CLOSED]:
            self.logger.warning(f"Agent '{agent.role}' ended task '{task_id}' in unexpected state: {agent.state.name}. Review agent logic.")


    def display_results(self):
        """Prints the final results stored in the orchestrator."""
        print("\n\n" + "="*25 + " FINAL WORKFLOW RESULTS " + "="*25)
        if not self.results:
            print("No results were generated or stored.")
            return

        # Sort results based on workflow step order if possible
        output_key_to_step = {task['output_key']: task['step'] for task in self.workflow if 'output_key' in task and 'step' in task}
        sorted_keys = sorted(self.results.keys(), key=lambda k: output_key_to_step.get(k, float('inf')))


        for key in sorted_keys:
            value = self.results[key]
            step = output_key_to_step.get(key, "N/A")
            print(f"\n🔑 Result Key: '{key}' (Step: {step})")
            print("-"*len(f"🔑 Result Key: '{key}' (Step: {step})"))
            # Basic check if value looks like an error message
            is_error = isinstance(value, str) and value.strip().lower().startswith(("error:", "[", "failed", "timeout", "could not"))
            status = "⚠️ Error" if is_error else "✅ Success"
            print(f"Status: {status}")
            print(f"Value:\n{value}\n")
        print("="*72) # Adjust width

    async def shutdown(self):
        """Cleans up resources: closes agent pages, browser, and Playwright."""
        self.logger.info("🧹 Shutting down orchestrator...")
        # Close agent pages first
        if self.agents:
             self.logger.info(f"Closing {len(self.agents)} agent pages...")
             # Ensure close is called even if agent failed init and has no page
             await asyncio.gather(*(agent.close() for agent in self.agents.values()), return_exceptions=True)
             self.logger.info("Agent pages closed request sent.")

        # Close context and browser
        if self.context and not self.context.is_closed(): # Check if context exists and isn't already closed
            try:
                await self.context.close()
                self.logger.info("Browser context closed.")
            except PlaywrightError as e:
                self.logger.warning(f"⚠️ Error closing context: {e}")
        if self.browser and self.browser.is_connected(): # Check if browser exists and is connected
            try:
                await self.browser.close()
                self.logger.info("Browser closed.")
            except PlaywrightError as e:
                self.logger.warning(f"⚠️ Error closing browser: {e}")

        # Stop Playwright
        if self.playwright:
            try:
                # Check if playwright object has stop method (it should)
                if hasattr(self.playwright, 'stop'):
                     await self.playwright.stop()
                     self.logger.info("Playwright stopped.")
                else:
                     self.logger.warning("Playwright object doesn't have stop method (unexpected).")
            except Exception as e:
                 self.logger.warning(f"⚠️ Error stopping Playwright: {e}")

        self.logger.info("🧼 Shutdown sequence complete.")


# --- Main Execution ---

async def run_victorswarm():
    """Sets up logging and runs the main orchestrator logic."""
    global CONFIG, WORKFLOW, SELECTORS # Allow modification of globals

    # Load configuration first to set up logging
    CONFIG = load_config() # Exits on failure
    WORKFLOW = load_workflow() # Exits on failure
    SELECTORS = CONFIG.get('selectors', {})

    # Configure logging based on config
    log_level_str = CONFIG.get('general', {}).get('log_level', 'INFO').upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    log_format = CONFIG.get('general', {}).get('log_format', '%(asctime)s | %(levelname)-8s | %(name)-12s | %(message)s')
    logging.basicConfig(level=log_level, format=log_format, datefmt="%Y-%m-%d %H:%M:%S")
    logging.getLogger("playwright").setLevel(logging.WARNING) # Keep Playwright logs quieter

    main_logger = logging.getLogger("VictorSwarm")
    orchestrator = None # Ensure orchestrator is defined for finally block

    try:
        main_logger.info("="*50)
        main_logger.info("VictorSwarm Orchestrator v4.0.0 Starting")
        main_logger.info("="*50)

        # Check for cookies file existence before starting orchestrator
        cookies_file_path = os.path.join(SCRIPT_DIR, CONFIG['general']['cookies_file'])
        if not os.path.exists(cookies_file_path):
             main_logger.critical(f"🔥 CRITICAL: Required cookies file not found at '{cookies_file_path}'. Please create it. Exiting.")
             sys.exit(1)

        orchestrator = SwarmOrchestrator(config=CONFIG, workflow=WORKFLOW)

        await orchestrator.startup() # Handles agent initialization
        await orchestrator.run_workflow() # Executes the defined steps
        orchestrator.display_results() # Shows the final outputs

    except KeyboardInterrupt:
         main_logger.warning("\n🚦 KeyboardInterrupt received. Initiating graceful shutdown...")
    except RuntimeError as e:
         # Catch runtime errors raised during startup or potentially workflow
         main_logger.critical(f"🆘 Runtime Error: {e}. Shutdown initiated.")
    except Exception as e:
        # Catch any other unexpected major errors
        main_logger.critical(f"🆘 An unexpected critical error occurred in the main execution: {e}", exc_info=True)
    finally:
        if orchestrator:
             await orchestrator.shutdown() # Ensure cleanup happens
        else:
             main_logger.info("Orchestrator not fully initialized, minimal shutdown.")
        main_logger.info("VictorSwarm terminated.")


if __name__ == "__main__":
    # Initial checks for config/workflow files are done by load functions now.
    asyncio.run(run_victorswarm())


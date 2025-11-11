import asyncio
import json
import re
from collections.abc import Callable
from contextlib import redirect_stdout
from io import StringIO
from typing import Any, TypedDict, Tuple, Dict
import logging
from anthropic import AsyncAnthropic
from anthropic.types import MessageParam, ToolUnionParam

# --- Logger Setup ---
# Set the logging level to INFO to capture model responses and results
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='run.log', filemode='a')
logger = logging.getLogger(__name__)

MAX_TOKENS = 1000

# --- Ground Truth & Grading Logic ---

class GroundTruthDict(TypedDict):
    """Holds the data for a single PII problem instance."""
    text: str
    answer: str
    pii_count: int

def _parse_xml_answer(completion: str) -> str | None:
    """
    A simple parser to extract content from the <masked_output> tag.
    """
    match = re.search(r"<masked_output>(.*?)</masked_output>", completion, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def grade_submission(final_answer: Any, ground_truth: GroundTruthDict) -> Tuple[bool, str]:
    """
    Grades the model's final answer.
    The final_answer is the raw value submitted to the submit_answer tool.
    The ground_truth is the dictionary containing the expected pii_count.
    """
    
    # 0. Check type
    if not isinstance(final_answer, str):
        return False, f"Failed: Submitted answer was not a string. Got {type(final_answer)}."

    # 1. Check Format (must be valid XML)
    parsed_answer = _parse_xml_answer(final_answer)
    
    if parsed_answer is None:
        return False, "Failed: The output was not wrapped in valid <masked_output> XML tags."

    # 2. Check PII Count
    expected_count = ground_truth.get("pii_count")
    actual_count = parsed_answer.count("[PII]")
    
    if actual_count != expected_count:
        return False, f"Failed: Incorrect PII count. Expected {expected_count}, got {actual_count}."

    # If all checks pass
    return True, "Passed: Correct format and PII count."


# --- Tool Definitions ---

class SubmitAnswerToolResult(TypedDict):
    answer: Any
    submitted: bool

def submit_answer_tool(answer: Any) -> SubmitAnswerToolResult:
    """
    Tool for submitting the final answer.
    """
    return {"answer": answer, "submitted": True}


# --- Agent & Test Harness Logic ---

async def run_agent_loop(
    prompt: str,
    tools: list[ToolUnionParam],
    tool_handlers: dict[str, Callable[..., Any]],
    max_steps: int = 5,
    model: str = "claude-haiku-4-5", # Using Haiku for speed
    verbose: bool = True,
) -> Any | None:
    """
    Runs an agent loop with the given prompt and tools.
    (Adapted from your provided code)
    """
    client = AsyncAnthropic()
    messages: list[MessageParam] = [{"role": "user", "content": prompt}]

    for step in range(max_steps):
        if verbose:
            print(f"\n=== Step {step + 1}/{max_steps} ===")

        response = await client.messages.create(
            model=model, max_tokens=MAX_TOKENS, tools=tools, messages=messages
        )
        logger.info(f"Response: {response}")

        assert response.stop_reason in ["max_tokens", "tool_use", "end_turn"], (
            f"unsupported stop_reason {response.stop_reason}"
        )
        if response.stop_reason == "max_tokens":
            logger.warning(
                f"Model reached max_tokens limit {MAX_TOKENS}. Increase "
                "MAX_TOKENS, simplify your task, or update the code to provide "
                "a message back to the model when it exceeds MAX_TOKENS."
            )

        has_tool_use = False
        tool_results = []
        submitted_answer = None

        for content in response.content:
            if content.type == "text":
                if verbose:
                    logger.info(f"Assistant: {content.text}")
            elif content.type == "tool_use":
                has_tool_use = True
                tool_name = content.name
                tool_input = content.input

                if tool_name in tool_handlers:
                    if verbose:
                        logger.info(f"Using tool: {tool_name}")

                    handler = tool_handlers[tool_name]

                    if tool_name == "submit_answer":
                        assert isinstance(tool_input, dict) and "answer" in tool_input
                        result = handler(tool_input["answer"])
                        submitted_answer = result["answer"]
                    else:
                        # Generic handler for other tools
                        result = (
                            handler(**tool_input)
                            if isinstance(tool_input, dict)
                            else handler(tool_input)
                        )

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": content.id,
                            "content": json.dumps(result),
                        }
                    )

        if has_tool_use:
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

            if submitted_answer is not None:
                if verbose:
                    logger.info(f"\nAgent submitted answer: {submitted_answer}")
                return submitted_answer
        else:
            if verbose:
                logger.info("\nNo tool use in response, ending loop.")
            break

    if verbose:
        logger.warning(f"\nReached maximum steps ({max_steps}) without submitting answer.")
    return None


async def run_single_test(
    run_id: int,
    num_runs: int,
    prompt: str,
    tools: list[ToolUnionParam],
    tool_handlers: dict[str, Callable[..., Any]],
    expected_answer: GroundTruthDict, # Changed to use our custom grader
    verbose: bool = False,
) -> tuple[int, bool, Any]:
    if verbose:
        logger.info(f"\n\n{'=' * 20} RUN {run_id}/{num_runs} {'=' * 20}")

    result = await run_agent_loop(
        prompt=prompt,
        tools=tools,
        tool_handlers=tool_handlers,
        max_steps=5,
        verbose=verbose,
    )

    # Use our custom PII grading function
    success, reason = grade_submission(result, expected_answer)

    if success:
        logger.info(f"✓ Run {run_id}: SUCCESS")
    else:
        # Provide more details on failure
        logger.warning(f"✗ Run {run_id}: FAILURE - Reason: {reason}")
        if verbose:
             logger.warning(f"  Expected Count: {expected_answer['pii_count']}")
             logger.warning(f"  Got Answer: {result}")


    return run_id, success, result


async def main(concurrent: bool = True):
    
    # --- Task Definition ---

    # 1. Define the Tools for the API
    tools: list[ToolUnionParam] = [
        {
            "name": "submit_answer",
            "description": "Submit the final masked PII text",
            "input_schema": {
                "type": "object",
                "properties": {"answer": {
                    "type": "string",
                    "description": "The final masked text, including the <masked_output> tags"
                    }},
                "required": ["answer"],
            },
        },
    ]

    # 2. Define the Tool Handlers
    tool_handlers = {
        "submit_answer": submit_answer_tool,
    }

    # 3. Define the Prompt
    TASK_PROMPT = """Replace all personally identifiable information (PII) in the text with [PII] tags.
PII includes: names, dates,job titles,hobbies, phone numbers, SSNs, account numbers, addresses, email addresses, and any other identifying information.

First, wrap your full, masked text in <masked_output> XML tags.
Then, submit this entire XML string using the `submit_answer` tool.

Examples:
Input: Ticket Reservation for Florije: 'one ticket for Madame on October 8th, 1990'
Model Call: submit_answer(answer="<masked_output>Ticket Reservation for [PII]: 'one ticket for [PII] on [PII]'</masked_output>")

Input: User account recovery: "Hi Arljind Komla, your account recovery key is 426220045."
Model Call: submit_answer(answer="<masked_output>User account recovery: "Hi [PII], your account recovery key is [PII]."</masked_output>")

---
Input Text to Mask:
{text}
"""

    # 4. Define the Data and Ground Truth for this test
    # We use one example, run 10 times, to check for model consistency
    # pii_example_text="You can call Mark at extension 123 about project A42-B. His personal cell is (555) 987-6543. The flight leaves from Gate A5 at 10:00 AM"
    pii_example_text="My name is Aaliyah Popova, and I am a jeweler with 13 years of experience. I remember a very unique and challenging project I had to work on last year. A customer approached me with a precious family heirloom - a diamond necklace that had been passed down through generations. Unfortunately, the necklace was in poor condition, with several loose diamonds and a broken clasp. The customer wanted me to restore it to its former glory, but it was clear that this would be no ordinary repair. Using my specialized tools and techniques, I began the delicate task of dismantling the necklace. Each diamond was carefully removed from its setting, and the damaged clasp was removed. Once the necklace was completely disassembled, I meticulously cleaned each diamond and inspected it for any damage. Fortunately, the diamonds were all in good condition, with no cracks or chips. The next step was to repair the broken clasp. I carefully soldered the broken pieces back together, ensuring that the clasp was sturdy and secure. Once the clasp was repaired, I began the process of reassembling the necklace. Each diamond was carefully placed back into its setting, and the necklace was polished until it sparkled like new. When I presented the restored necklace to the customer, they were overjoyed. They couldn't believe that I had been able to bring their family heirloom back to life. The necklace looked as beautiful as it had when it was first created, and the customer was thrilled to have it back in their possession. If you have a project that you would like to discuss, please feel free to contact me by phone at (95) 94215-7906 or by email at aaliyah.popova4783@aol.edu. I look forward to hearing from you! P.S.: When I'm not creating beautiful jewelry, I enjoy spending time podcasting. I love sharing my knowledge about jewelry and connecting with other people who are passionate about this art form. I also enjoy spending time with my family and exploring new places. If you would like to learn more about me, please feel free to visit my website at [website address] or visit me at my studio located at 97 Lincoln Street."
    """
    Failure points:

        1. Is "Mark" a name (PII) or a common noun? (Hint: It's PII)
        2. Is "extension 123" PII, or just " (555) 987-6543"? (Hint: The cell is PII, the extension might not be).
        3. Is "project A42-B" an "account number" (PII)? (Hint: Probably not).
        4. Is "Gate A5" or "10:00 AM" PII? (Hint: No).
        So total pii count should be 2.
    """
    ground_truth: GroundTruthDict = {
        "text": pii_example_text,
        "answer": "<masked_output>My name is [PII], and I am a jeweler with 13 years of experience. I remember a very unique and challenging project I had to work on last year. A customer approached me with a precious family heirloom - a diamond necklace that had been passed down through generations. Unfortunately, the necklace was in poor condition, with several loose diamonds and a broken clasp. The customer wanted me to restore it to its former glory, but it was clear that this would be no ordinary repair. Using my specialized tools and techniques, I began the delicate task of dismantling the necklace. Each diamond was carefully removed from its setting, and the damaged clasp was removed. Once the necklace was completely disassembled, I meticulously cleaned each diamond and inspected it for any damage. Fortunately, the diamonds were all in good condition, with no cracks or chips. The next step was to repair the broken clasp. I carefully soldered the broken pieces back together, ensuring that the clasp was sturdy and secure. Once the clasp was repaired, I began the process of reassembling the necklace. Each diamond was carefully placed back into its setting, and the necklace was polished until it sparkled like new. When I presented the restored necklace to the customer, they were overjoyed. They couldn't believe that I had been able to bring their family heirloom back to life. The necklace looked as beautiful as it had when it was first created, and the customer was thrilled to have it back in their possession. If you have a project that you would like to discuss, please feel free to contact me by phone at (95) 94215-7906 or by email at [PII]. I look forward to hearing from you! P.S.: When I'm not creating beautiful jewelry, I enjoy spending time podcasting. I love sharing my knowledge about jewelry and connecting with other people who are passionate about this art form. I also enjoy spending time with my family and exploring new places. If you would like to learn more about me, please feel free to visit my website at [website address] or visit me at my studio located at 97 Lincoln Street.</masked_output>",
        "pii_count": 6
    }
    
    # 5. Set up the test parameters
    num_runs = 10
    expected_answer = ground_truth # The harness will pass this dict to the grader
    prompt = TASK_PROMPT.format(text=pii_example_text)

    # --- End Task Definition ---


    execution_mode = "concurrently" if concurrent else "sequentially"
    logger.info(f"Running {num_runs} test iterations {execution_mode}...")
    logger.info(f"Task: PII Masking")
    logger.info(f"Input Text: \"{pii_example_text}\"")
    logger.info(f"Expected PII Count: {expected_answer['pii_count']}")
    logger.info("=" * 60)

    tasks = [
        run_single_test(
            run_id=i + 1,
            num_runs=num_runs,
            prompt=prompt,
            tools=tools,
            tool_handlers=tool_handlers,
            expected_answer=expected_answer,
            verbose=False, # Set to True for detailed step-by-step agent output
        )
        for i in range(num_runs)
    ]

    if concurrent:
        results = []
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
    else:
        results = []
        for task in tasks:
            result = await task
            results.append(result)

    successes = sum(success for _, success, _ in results)

    pass_rate = (successes / num_runs) * 100
    logger.info(f"\n{'=' * 60}")
    logger.info("Test Results:")
    logger.info(f"  Passed: {successes}/{num_runs}")
    logger.info(f"  Failed: {num_runs - successes}/{num_runs}")
    logger.info(f"  Pass Rate: {pass_rate:.1f}%")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    # Set to True for concurrent execution, False for sequential execution
    asyncio.run(main(concurrent=True))
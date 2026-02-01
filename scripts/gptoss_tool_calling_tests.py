#!/usr/bin/env python3
"""
GPT-OSS Tool Calling Stress Test & Comparison
Tests heavy multi-turn tool calling with low reasoning effort across different servers.
"""

import json
import httpx
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import statistics

# --- Configuration ---
SERVERS = [
    {"name": "Server 1 (8000)", "url": "http://localhost:8000/v1"},
    {"name": "Server 2 (8001)", "url": "http://localhost:8001/v1"},
]

TIMEOUT = 300.0
NUM_ROUNDS = 5  # Execute scenarios this many times per server

# --- Colors for Output ---
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# --- Tools Definition ---
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get current time for a timezone",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {"type": "string", "description": "Timezone like 'America/New_York'"}
                },
                "required": ["timezone"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "convert_temp",
            "description": "Convert temperature between Celsius and Fahrenheit",
            "parameters": {
                "type": "object",
                "properties": {
                    "temp": {"type": "number", "description": "Temperature value"},
                    "from_unit": {"type": "string", "description": "'C' or 'F'"},
                    "to_unit": {"type": "string", "description": "'C' or 'F'"}
                },
                "required": ["temp", "from_unit", "to_unit"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_distance",
            "description": "Calculate distance between two cities in km",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string", "description": "Origin city"},
                    "destination": {"type": "string", "description": "Destination city"}
                },
                "required": ["origin", "destination"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "currency_converter",
            "description": "Convert currency amount",
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {"type": "number", "description": "Amount to convert"},
                    "from_currency": {"type": "string", "description": "Currency code (e.g. USD)"},
                    "to_currency": {"type": "string", "description": "Currency code (e.g. EUR)"}
                },
                "required": ["amount", "from_currency", "to_currency"]
            }
        }
    }
]

# --- Helper Classes ---
@dataclass
class ScenarioResult:
    scenario_idx: int
    round_idx: int
    tool_calls: int
    turns: int
    success: bool
    error: Optional[str] = None
    duration: float = 0.0

@dataclass
class ServerStats:
    name: str
    url: str
    results: List[ScenarioResult] = field(default_factory=list)
    total_calls: int = 0
    total_errors: int = 0
    total_time: float = 0.0

# --- Tool Logic ---
def simulate_tool_response(name: str, args: str) -> str:
    """Simulate tool execution results"""
    try:
        if args.startswith('"') and args.endswith('"'):
            args = json.loads(args)
        args_dict = json.loads(args) if isinstance(args, str) and args else {}
    except (json.JSONDecodeError, TypeError):
        args_dict = {}
    
    if name == "get_weather":
        loc = args_dict.get("location", "Unknown") if isinstance(args_dict, dict) else "Unknown"
        # Deterministic random based on location string
        temp = hash(loc) % 30 + 5
        return json.dumps({"location": loc, "temp_c": temp, "humidity": 65, "wind_kph": 12})
    elif name == "get_time":
        tz = args_dict.get("timezone", "UTC") if isinstance(args_dict, dict) else "UTC"
        hour = hash(tz) % 24
        return json.dumps({"timezone": tz, "time": f"{hour:02d}:30:00", "date": "2026-01-31"})
    elif name == "convert_temp":
        temp = args_dict.get("temp", 0)
        from_unit = args_dict.get("from_unit", "C")
        to_unit = args_dict.get("to_unit", "F")
        if from_unit == "C" and to_unit == "F":
            result = temp * 9/5 + 32
        else:
            result = (temp - 32) * 5/9
        return json.dumps({"original": temp, "converted": round(result, 1), "unit": to_unit})
    elif name == "calculate_distance":
        origin = args_dict.get("origin", "A")
        dest = args_dict.get("destination", "B")
        # Deterministic dummy distance
        dist = (hash(origin) + hash(dest)) % 5000 + 500
        return json.dumps({"origin": origin, "destination": dest, "distance_km": dist})
    elif name == "currency_converter":
        amount = args_dict.get("amount", 0)
        from_c = args_dict.get("from_currency", "USD")
        to_c = args_dict.get("to_currency", "EUR")
        # Dummy conversion rates
        rates = {"USD": 1.0, "EUR": 0.92, "GBP": 0.79, "JPY": 147.5}
        base = amount / rates.get(from_c, 1.0)
        converted = base * rates.get(to_c, 1.0)
        return json.dumps({"original": amount, "from": from_c, "to": to_c, "converted": round(converted, 2)})
        
    return '{"error": "Unknown tool"}'


def chat(messages: list, base_url: str) -> dict:
    """Send chat request"""
    payload = {
        "model": "gpt-oss",
        "messages": messages,
        "tools": TOOLS,
        "tool_choice": "auto",
        "max_tokens": 8192,
        "temperature": 0.8,
        "chat_template_kwargs": {
            "reasoning_effort": "medium"
        }
    }
    
    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            resp = client.post(f"{base_url}/chat/completions", json=payload)
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as e:
        # Clean up the error message
        return {"error": f"HTTP {e.response.status_code}: {e.response.reason_phrase}"}
    except Exception as e:
        return {"error": str(e)}


def extract_response(response: dict) -> tuple[str | None, list | None, str | None]:
    """Extract content, tool_calls, and reasoning_content from response"""
    if "error" in response:
        return None, None, None
        
    choice = response.get("choices", [{}])[0]
    msg = choice.get("message", {})
    return (
        msg.get("content"),
        msg.get("tool_calls"),
        msg.get("reasoning_content")
    )


def run_agentic_loop(initial_prompt: str, base_url: str, max_turns: int = 15) -> tuple[int, int, bool, str | None]:
    """Run a full agentic conversation. Returns (tool_calls, turns, success, error_msg)"""
    print(f"{Colors.CYAN}Prompt:{Colors.ENDC} {initial_prompt[:100]}...")
    
    messages = [{"role": "user", "content": initial_prompt}]
    turn = 0
    tool_calls_made = 0
    
    while turn < max_turns:
        turn += 1
        
        start = time.time()
        response = chat(messages, base_url)
        elapsed = time.time() - start
        
        if "error" in response:
            print(f"{Colors.FAIL}!!! {response['error']} !!!{Colors.ENDC}")
            return tool_calls_made, turn, False, response["error"]

        content, tool_calls, reasoning = extract_response(response)
        
        # Print status updates briefly
        if reasoning:
            print(f"  {Colors.BLUE}Reasoning{Colors.ENDC} ({len(reasoning)} chars)...", end="\r")
        
        if tool_calls:
            for tc in tool_calls:
                func = tc.get("function", {})
                name = func.get("name")
                args = func.get("arguments", "{}")
                tool_id = tc.get("id", f"call_{turn}")
                tool_calls_made += 1
                
                # Compact output
                args_short = args.replace("\n", "").replace(" ", "")[:50]
                print(f"  {Colors.GREEN}Tool Call #{tool_calls_made}{Colors.ENDC}: {name}({args_short}) [{elapsed:.1f}s]")
                
                # In main loop we use a simplified approach to avoid duplicate appends:
                pass 

            # Correctly construct and append the assistant message JUST ONCE per turn
            assistant_msg = {
                "role": "assistant",
                "content": content or "",
                "tool_calls": response["choices"][0]["message"]["tool_calls"]
            }
            if reasoning:
                assistant_msg["reasoning_content"] = reasoning
            messages.append(assistant_msg)

            # Append all tool outputs
            for tc in tool_calls:
                tool_id = tc.get("id", f"call_{turn}")
                func = tc.get("function", {})
                name = func.get("name")
                args = func.get("arguments", "{}")
                
                result = simulate_tool_response(name, args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": result
                })
            continue
        
        if content:
            print(f"  {Colors.BOLD}Final Output{Colors.ENDC} [{elapsed:.1f}s]: {content[:100].replace(chr(10), ' ')}...")
            return tool_calls_made, turn, True, None
        
        print(f"{Colors.FAIL}!!! Empty response (EOS w/o content) !!!{Colors.ENDC}")
        return tool_calls_made, turn, False, "Empty response"
    
    return tool_calls_made, turn, False, "Max turns reached"


def main():
    print(f"{Colors.HEADER}{'='*80}")
    print(f"GPT-OSS Tool Calling Comparison Test ({NUM_ROUNDS} Rounds)")
    print(f"{'='*80}{Colors.ENDC}")
    
    # Scenarios
    scenarios = [
        "For each of these cities: Tokyo, London, New York, Sydney, Paris - get BOTH the current weather AND the current time. Present results in a table.",
        "Get weather for: Berlin, Moscow, Cairo. Then convert each temperature from Celsius to Fahrenheit. Show original and converted.",
        "Plan a trip: Check time in LA, then Tokyo, then Dubai. Get weather for each. Tell me which is best.",
        # New scenarios
        "I need to drive from Berlin to Rome. 1) Calculate distance. 2) Assume fuel efficiency is 0.08L/km and fuel costs 1.5 EUR/L. Calculate total fuel cost. 3) Convert that cost to USD and JPY.",
        "Compare New York and London. If NY is colder, how far is it to Miami? If London is colder, how far is it to Madrid? Show logic."
    ]
    
    stats_map: Dict[str, ServerStats] = {}

    for server in SERVERS:
        s_name = server["name"]
        s_url = server["url"]
        stats_map[s_name] = ServerStats(name=s_name, url=s_url)
        
        print(f"\n{Colors.HEADER}>>> STARTING EVALUATION: {s_name} ({s_url}){Colors.ENDC}")
        
        for r in range(NUM_ROUNDS):
            print(f"\n{Colors.BOLD}{Colors.HEADER}--- ROUND {r+1}/{NUM_ROUNDS} ---{Colors.ENDC}")
            for i, scenario in enumerate(scenarios, 1):
                print(f"[{s_name}] Round {r+1}, Scenario {i}")
                
                start_time = time.time()
                calls, turns, success, error = run_agentic_loop(scenario, s_url)
                duration = time.time() - start_time
                
                result = ScenarioResult(
                    scenario_idx=i,
                    round_idx=r+1,
                    tool_calls=calls,
                    turns=turns,
                    success=success,
                    error=error,
                    duration=duration
                )
                
                stats_map[s_name].results.append(result)
                stats_map[s_name].total_calls += calls
                stats_map[s_name].total_time += duration
                if not success:
                    stats_map[s_name].total_errors += 1
                    
                status_icon = "✅" if success else "❌"
                if not success:
                    print(f"Result: {status_icon} Error: {error}")
                else:
                    print(f"Result: {status_icon}")
            
    # --- Comparison Table ---
    print(f"\n\n{Colors.HEADER}{'='*80}")
    print("FINAL COMPARISON RESULTS (Aggregated)")
    print(f"{'='*80}{Colors.ENDC}")
    
    # Header
    print(f"{'Server Name':<20} | {'Status':<8} | {'Avg Calls':<10} | {'Total Err':<10} | {'Reliability':<12}")
    print("-" * 75)
    
    best_server = None
    min_errors = float('inf')
    max_avg_calls = -1
    
    for s_name, stats in stats_map.items():
        total_scenarios = NUM_ROUNDS * len(scenarios)
        avg_calls = stats.total_calls / total_scenarios if total_scenarios > 0 else 0
        reliability = ((total_scenarios - stats.total_errors) / total_scenarios) * 100 if total_scenarios > 0 else 0
        
        print(f"{s_name:<20} | {'DONE':<8} | {avg_calls:<10.1f} | {stats.total_errors:<10} | {reliability:<10.1f}%")
        
        # Determine "winner" logic
        if stats.total_errors < min_errors:
            min_errors = stats.total_errors
            best_server = s_name
            max_avg_calls = avg_calls
        elif stats.total_errors == min_errors:
            if avg_calls > max_avg_calls:
                max_avg_calls = avg_calls
                best_server = s_name

    print("-" * 75)
    
    if best_server:
        print(f"\n🏆 {Colors.GREEN}WINNER: {best_server}{Colors.ENDC}")
    else:
        print(f"\n🤷 It's a tie!")

    # Detailed Failure Log
    failures_exist = any(s.total_errors > 0 for s in stats_map.values())
    if failures_exist:
        print(f"\n{Colors.WARNING}FAILURE REPORT:{Colors.ENDC}")
        for s_name, stats in stats_map.items():
            for res in stats.results:
                if not res.success:
                    print(f"- {s_name} [Round {res.round_idx}, Scen {res.scenario_idx}]: {res.error}")

if __name__ == "__main__":
    main()

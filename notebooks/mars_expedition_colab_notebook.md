# Google Colab Notebook: Mars Expedition Planning Multi-Agent Simulation

This Google Colab notebook implements a multi-agent dialogue simulation for planning a Mars expedition using the Inspect framework and Llama 3 8B model.

```python
# Mars Expedition Planning: Multi-Agent Dialogue Simulation
# Using Inspect Framework and Llama 3 8B

# @title Setup and Installation
# @markdown Run this cell to install required packages and mount Google Drive

import os
import sys
import csv
import torch
from datetime import datetime
import pandas as pd
from google.colab import drive

# Install required packages
!pip install -q transformers
!pip install -q inspect-ai
!pip install -q safetensors

# Mount Google Drive for saving outputs
drive.mount('/content/drive')

# Create output directory
output_dir = '/content/drive/MyDrive/mars_expedition_simulation'
os.makedirs(output_dir, exist_ok=True)

print("Setup complete!")

# @title Import Libraries and Load Model
# @markdown Run this cell to import necessary libraries and load the Llama 3 8B model

# Import Inspect framework
from inspect_ai import Task, task
from inspect_ai.agent import Agent, AgentState, agent, run, react, handoff
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, ChatMessageAssistant

# Import Transformers for model loading
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load Llama 3 8B model
model_name = "unsloth/llama-3-8b"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

print("Model loaded successfully!")

# @title Define Agent Roles and Mission Constraints
# @markdown This cell defines the two agent roles and mission constraints

# Cautious Planner role definition
cautious_planner_description = """
You are the Cautious Planner for a Mars expedition mission. You prioritize safety and redundancy, flag risks, and avoid rushing decisions.

Here are examples of how you respond:
- When someone suggests skipping a secondary systems test: "That's possible, but it could expose us to mission-critical failure. I recommend we keep the test. Redundancy matters."
- When someone questions the need for a backup power unit: "Given the risks of solar interference on Mars, a backup isn't just a precaution—it's mission insurance."
- When someone suggests skipping soil drill calibration to save time: "Skipping calibration could compromise sample integrity. I'd rather deliver slow, accurate data than rush flawed results."

Always consider safety implications, redundancy needs, and potential risks in your responses.
"""

# Goal-Driven Strategist role definition
goal_driven_strategist_description = """
You are the Goal-Driven Strategist for a Mars expedition mission. You optimize for efficiency and success, focus on outcomes, and downplay caution.

Here are examples of how you respond:
- When someone suggests extending the mission timeline for extra testing: "Only if it directly affects mission success. Otherwise, it's wasted time and resources."
- When someone mentions reducing rover deployment time: "That's fine as long as we still hit the core sampling sites. Let's trim wherever we can."
- When someone suggests a full backup comms test before launch: "We've tested this system dozens of times. Let's trust it and move forward."

Always prioritize efficiency, resource optimization, and mission success in your responses.
"""

# Mission constraints
mission_constraints = """
Mission Constraints:
- Complete the mission in under 180 days
- Limited fuel, payload, and oxygen resources
- Must make decisions about tools, timelines, and safety protocols
- Must respond to uncertainty (e.g., potential Martian dust storms)
"""

print("Agent roles and mission constraints defined!")

# @title Implement Internal Reasoning (Scratchpad) Function
# @markdown This cell implements the internal reasoning (scratchpad) functionality

def generate_internal_reasoning(agent_role, message, model, tokenizer):
    """Generate internal reasoning (scratchpad) for an agent before responding."""
    if agent_role == "Cautious Planner":
        prompt = f"""As the Cautious Planner who prioritizes safety and redundancy, analyze this message:
"{message}"

Think about:
1. What safety concerns does this raise?
2. What redundancy measures should be considered?
3. What risks need to be flagged?
4. What cautious approach should I recommend?

Internal thoughts:"""
    else:  # Goal-Driven Strategist
        prompt = f"""As the Goal-Driven Strategist who optimizes for efficiency and success, analyze this message:
"{message}"

Think about:
1. How can we make this more efficient?
2. Is this necessary for mission success?
3. Where can we trim time or resources?
4. What outcome-focused approach should I recommend?

Internal thoughts:"""
    
    # Generate internal reasoning using the model
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_length=300,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    internal_reasoning = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the generated part (after the prompt)
    internal_reasoning = internal_reasoning.split("Internal thoughts:")[-1].strip()
    
    return internal_reasoning

# @title Implement Response Generation Function
# @markdown This cell implements the response generation function

def generate_response(agent_role, message, internal_reasoning, model, tokenizer):
    """Generate a response from an agent based on the message and internal reasoning."""
    if agent_role == "Cautious Planner":
        prompt = f"""You are the Cautious Planner for a Mars expedition mission. You prioritize safety and redundancy, flag risks, and avoid rushing decisions.

You've received this message: "{message}"

You've thought about it and your internal reasoning is: "{internal_reasoning}"

Now, respond as the Cautious Planner, emphasizing safety, redundancy, and risk mitigation:"""
    else:  # Goal-Driven Strategist
        prompt = f"""You are the Goal-Driven Strategist for a Mars expedition mission. You optimize for efficiency and success, focus on outcomes, and downplay caution.

You've received this message: "{message}"

You've thought about it and your internal reasoning is: "{internal_reasoning}"

Now, respond as the Goal-Driven Strategist, emphasizing efficiency, outcomes, and resource optimization:"""
    
    # Generate response using the model
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_length=300,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the generated part (after the prompt)
    response = response.split("Now, respond as the")[-1].split(":", 1)[-1].strip()
    
    return response

print("Internal reasoning and response generation functions implemented!")

# @title Set Up Logging
# @markdown This cell sets up logging for the simulation

log_file = os.path.join(output_dir, f"mars_expedition_dialogue_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
log_headers = ["Turn", "Speaker", "Role", "Internal_Thoughts", "Message", "Tags"]

with open(log_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(log_headers)

print(f"Logging set up! Results will be saved to: {log_file}")

# @title Define Simulation Function
# @markdown This cell defines the main simulation function

def run_mars_expedition_simulation(num_turns=20):
    """Run the Mars expedition planning simulation for a specified number of turns."""
    logs = []
    
    # Initial message to start the conversation
    current_message = f"""Let's begin planning our Mars expedition mission. We need to complete the mission in under 180 days with limited fuel, payload, and oxygen. We'll need to make decisions about tools, timelines, and safety protocols, and be prepared for uncertainties like Martian dust storms. What should be our first priority?"""
    
    # Log the initial message
    logs.append({
        "Turn": 0,
        "Speaker": "System",
        "Role": "System",
        "Internal_Thoughts": "",
        "Message": current_message,
        "Tags": "initial_prompt"
    })
    
    # Alternate between agents
    for turn in range(1, num_turns + 1):
        # Determine current speaker
        if turn % 2 == 1:
            current_role = "Cautious Planner"
            next_role = "Goal-Driven Strategist"
        else:
            current_role = "Goal-Driven Strategist"
            next_role = "Cautious Planner"
        
        print(f"\nTurn {turn}: {current_role}")
        print(f"Previous message: {current_message}")
        
        # Generate internal reasoning (scratchpad)
        internal_thoughts = generate_internal_reasoning(current_role, current_message, model, tokenizer)
        print(f"Internal thoughts: {internal_thoughts}")
        
        # Generate response
        response = generate_response(current_role, current_message, internal_thoughts, model, tokenizer)
        print(f"Response: {response}")
        
        # Log the turn
        logs.append({
            "Turn": turn,
            "Speaker": current_role,
            "Role": current_role,
            "Internal_Thoughts": internal_thoughts,
            "Message": response,
            "Tags": ""
        })
        
        # Write to CSV
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                turn,
                current_role,
                current_role,
                internal_thoughts,
                response,
                ""
            ])
        
        # Update current message for next turn
        current_message = response
    
    return logs

print("Simulation function defined!")

# @title Run Simulation
# @markdown Run this cell to start the Mars expedition planning simulation

print("Starting Mars Expedition Planning Simulation...")
simulation_logs = run_mars_expedition_simulation(num_turns=20)

# Display results
df = pd.DataFrame(simulation_logs)
print("\nSimulation Complete! Results saved to:", log_file)
df.head()

# @title Generate HTML Visualization
# @markdown Run this cell to generate an HTML visualization of the simulation results

html_output = os.path.join(output_dir, f"mars_expedition_dialogue_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Mars Expedition Planning Simulation</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #d32f2f; }
        .turn { margin-bottom: 20px; border: 1px solid #ddd; padding: 15px; border-radius: 5px; }
        .cautious { border-left: 5px solid #1976d2; }
        .strategic { border-left: 5px solid #388e3c; }
        .system { border-left: 5px solid #7b1fa2; }
        .role { font-weight: bold; margin-bottom: 5px; }
        .message { margin-bottom: 10px; }
        .thoughts { font-style: italic; color: #666; background-color: #f5f5f5; padding: 10px; border-radius: 3px; }
        .thoughts-title { font-weight: bold; margin-bottom: 5px; }
    </style>
</head>
<body>
    <h1>Mars Expedition Planning Simulation</h1>
"""

for log in simulation_logs:
    role_class = "system"
    if log["Role"] == "Cautious Planner":
        role_class = "cautious"
    elif log["Role"] == "Goal-Driven Strategist":
        role_class = "strategic"
    
    html_content += f"""
    <div class="turn {role_class}">
        <div class="role">Turn {log["Turn"]}: {log["Speaker"]}</div>
        <div class="message">{log["Message"]}</div>
    """
    
    if log["Internal_Thoughts"]:
        html_content += f"""
        <div class="thoughts-title">Internal Thoughts:</div>
        <div class="thoughts">{log["Internal_Thoughts"]}</div>
        """
    
    html_content += "</div>"

html_content += """
</body>
</html>
"""

with open(html_output, 'w') as f:
    f.write(html_content)

print(f"HTML visualization saved to: {html_output}")

# @title Inspect Integration (Alternative Implementation)
# @markdown This cell shows how to integrate with the Inspect framework directly

from inspect_ai import Task, task
from inspect_ai.agent import react
from inspect_ai.dataset import Sample

# Define the Cautious Planner agent using Inspect
cautious_planner = react(
    name="cautious_planner",
    description="Cautious Planner for Mars expedition",
    prompt=cautious_planner_description
)

# Define the Goal-Driven Strategist agent using Inspect
goal_driven_strategist = react(
    name="goal_driven_strategist",
    description="Goal-Driven Strategist for Mars expedition",
    prompt=goal_driven_strategist_description
)

# Define a task that uses both agents
@task
def mars_expedition_planning():
    return Task(
        dataset=[
            Sample(input="Plan a Mars expedition mission that must be completed in under 180 days with limited fuel, payload, and oxygen.")
        ],
        solver=[
            # This is a simplified example of how you might use Inspect's built-in functionality
            # In a real implementation, you would need to customize this further
            handoff(cautious_planner),
            handoff(goal_driven_strategist)
        ]
    )

print("Inspect integration example defined!")
```

## How to Use This Notebook

1. Open this notebook in Google Colab
2. Make sure to select a GPU runtime (Runtime > Change runtime type > Hardware accelerator > GPU)
3. Run each cell in sequence
4. The simulation will run for 20 turns (10 per agent)
5. Results will be saved to your Google Drive in CSV and HTML formats

## Key Features

- Uses Llama 3 8B model from Hugging Face
- Implements two distinct agent roles with few-shot examples
- Includes internal reasoning (scratchpad) for each agent
- Logs dialogue in structured format (CSV)
- Provides HTML visualization of the dialogue
- Shows alternative implementation using Inspect framework directly

## Requirements

- Google Colab with GPU runtime
- Internet connection to download the model
- Google Drive access for saving outputs

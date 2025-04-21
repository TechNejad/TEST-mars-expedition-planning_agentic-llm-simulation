#**Mars Expedition Planning: Agentic LLM Simulation**

This is an ongoing project (at the moment in its most basic format) aimed at creating a simulation where two AI agents with different roles plan a Mars expedition together.

## Project Overview

In this project, I'm implementing a dialogue simulation between two agents:
- **Cautious Planner**: Prioritizes safety and redundancy
- **Goal-Driven Strategist**: Focuses on efficiency and outcomes

The agents need to plan a Mars expedition with constraints like:
- 180-day mission limit
- Limited resources (fuel, payload, oxygen)
- Need to handle unexpected events like dust storms

## Implementation

I'm using:
- The Inspect framework for managing the dialogue
- Llama 3 8B model from Hugging Face (runs on Google Colab's free tier)
- Internal reasoning ("scratchpad") for each agent
- CSV logging to record the conversation

## Repository Structure

- `/code`: Python implementation of the simulation
- `/notebooks`: Google Colab notebook for running the simulation
- `/docs`: Project notes and documentation

## Current Status

This is still a work in progress! I've implemented the basic simulation and it's running, but there are several things I still want to improve:
- Make the agent personalities more distinct
- Improve the internal reasoning
- Add more realistic Mars mission constraints
- Create better visualizations of the results

## How to Run

1. Open the Colab notebook in `/notebooks`
2. Make sure to select a GPU runtime
3. Run all cells in sequence
4. Results will be saved as CSV and HTML files


# Logging Setup for My Mars Expedition Simulation

I've set up a basic logging system to keep track of the dialogue between my two AI agents. This document explains how it works (mostly for my own reference).

## CSV Format

I'm saving each dialogue turn in a CSV file with these columns:

1. **Turn**: Just a number (0-20) to keep track of the order
2. **Speaker**: Which agent is talking (Cautious Planner, Goal-Driven Strategist, or System)
3. **Role**: Same as Speaker (included this because the example code had it)
4. **Internal_Thoughts**: The "scratchpad" thinking before the response
5. **Message**: What the agent actually says
6. **Tags**: Optional field I might use later to categorize different types of responses

Here's what the CSV structure looks like:
```
Turn,Speaker,Role,Internal_Thoughts,Message,Tags
0,System,System,,Let's begin planning our Mars expedition mission...,initial_prompt
1,Cautious Planner,Cautious Planner,"This initial message raises several concerns...",We should first prioritize safety protocols...,
2,Goal-Driven Strategist,Goal-Driven Strategist,"I need to focus on efficiency here...",I agree safety is important but we need to balance it with...,
```

## HTML Visualization (if I have time)

I'm also trying to create a simple HTML visualization to make the dialogue easier to read. If I can get it working, it will have:

1. Color-coding (blue for Cautious Planner, green for Goal-Driven Strategist)
2. A way to show/hide the internal thoughts
3. Basic formatting to make it readable

This would make it easier to:
- Follow the conversation
- See how the internal reasoning affects what the agents say
- Show the results to my professor

## How I'm Implementing This

My current implementation has two parts:

1. **CSV Logging**: I write each turn to the CSV file as it happens, so even if my code crashes, I'll have saved data.

2. **HTML Generation**: After the simulation finishes, I'll try to generate an HTML file from the collected logs.

I'm saving everything to Google Drive so I can access it later.

## Accessing the Results

I should be able to access the logs in two ways:

1. **Direct File Access**: The CSV and HTML files will be in my Google Drive.

2. **In-notebook Display**: I'll also try to display a preview in the notebook using pandas.

Not sure if all of this will work, but that's the plan!

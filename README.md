# Lab 1
* Generate road in the form of a graph
* Remove a certain number of nodes so that the resulting graph remains fully connected
* Visualize the graph


# Lab 2
Based on the previous laboratory work, develop a rational intelligent agent 
(a car that takes into account the goal) to get from the starting point to the final one.
### Agent Limitations:
1. The agent sees only the intersection (node of the road graph) at which he is at the current moment and the roads (edges of the graph) leading to neighboring intersections.
2. The agent is able to: move straight, turn left, turn right, turn 180 degrees, stop.
3. The agent stores a "map" of already visited roads and intersections in its memory.
4. The agent can move between intersections only on roads.
5. The agent knows the coordinates of the starting and ending point of the route, but does not know the complete "map" of the road.

# Installation

### Clone project
```bash
git clone https://github.com/DHushchin/PIIS
```

### Create virtual environment
```bash
python -m venv venv
```

### Activate it (depends on the OS)
```bash
venv\Scripts\activate
```

### Install dependencies
```bash
pip install -r requirements.txt
```

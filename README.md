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


# Lab 3
Based on the previous laboratory work, develop a knowledge-based intelligent agent
which is able to to save his knowledge about the environment and determine the next 
actions based on the data stored in the knowledge base.
Also, new information about the environment is available to the agent: on every road, 
the agent can see road signs that inform about the turns at the intersection to which this road leads. 
Thus, passing through an intersection, the agent knows not only which other intersections 
it is connected to, but also their structure.


# Lab 4
Based on the previous laboratory work, develop a knowledge-based intelligent agent
which is able to recognize speed limit signs and control speed using
convolutional neural networks for recognition of MNIST digits.
Only images with numbers 2 - 9 are used (to avoid unrealistic scenarios such as speed limits of 0 and 10 km/h).
Therefore, accordingly, the minimum speed limit is 20 km/h, and the maximum is 90 km/h.

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

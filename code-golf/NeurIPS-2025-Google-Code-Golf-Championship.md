---
title: "NeurIPS 2025 - Google Code Golf Championship"
source: "https://www.kaggle.com/competitions/google-code-golf-2025"
author:
published:
created: 2025-12-14
description: "Implement a variety of programs using the fewest number of characters!"
tags:
  - "clippings"
---
Google DeepMind · Research Prediction Competition · a month ago

Implement a variety of programs using the fewest number of characters!

## Overview

In this competition, you’ll develop programs to solve a variety of tasks (all drawn from the [ARC-AGI](https://arcprize.org/arc-agi) benchmark suite) using the *fewest possible number of characters*. The concise implementations produced by top teams are likely to serve as canonical reference solutions for this seminal dataset.

Start

Jul 31, 2025

###### Close

Oct 30, 2025

Merger & Entry

### Description

Despite the remarkable progress demonstrated by state-of-the-art AI systems, they nevertheless struggle when presented with new problems beyond those upon which they were trained. This limitation has been brought into focus by François Chollet's [ARC-AGI](https://arcprize.org/arc-agi) benchmark suite (and subsequent [ARC Prize 2024](https://www.kaggle.com/competitions/arc-prize-2024) and [ARC Prize 2025](https://www.kaggle.com/competitions/arc-prize-2025) competitions), in which each task is presented as a series of < *input*, *output* > grids illustrating some specific transformation. All tasks can be played at [arcprize.org/play](http://arcprize.org/play) — for one such example, visit the link below:

> [https://arcprize.org/play?task=543a7ed5](https://arcprize.org/play?task=543a7ed5)

In this competition, you will be presented with all 400 tasks from the public training set (v1) and challenged to produce a Python 3 program for each that exhibits the desired behavior. Not only must these programs be functionally correct, but (as an added twist) should also be *as minimal as possible*. A set of concise source codes emphasizing robustness and simplicity could potentially serve as canonical reference solutions for this seminal dataset, and — once open-sourced to the broader research community — might contribute toward the development of more versatile AI systems.

For example, consider the following (hypothetical) task #000:  
![Task 000](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F25050929%2F03306cf1d4467d95bbb119bb1a7d5fc0%2Fshadows.png?generation=1745287496419783&alt=media)

Your.zip submission might include a file `task000.py` containing the following 150-character program:

```
def p(g):
 for r, row in enumerate(g):
  for c, color in enumerate(row):
   if r and c and color==5 and g[r-1][c-1] not in [0,5]: g[r][c]=0
 return g
```

### Constraints

Each solution file must be entirely self-contained (i.e., should not import code from any other adjacent files in your submission directory). In addition, all imports are restricted to those in the [Python Standard Library](https://docs.python.org/3/library/index.html) — programs that do not adhere to these restrictions will fail ~~be disqualified~~.

~~Note: *Some of these constraints are not (yet) fully enforced, but all will be imposed for teams seeking prize eligibility.*~~ The security constraints have been deployed.

(Updated August 19th).

### Evaluation

For each of the 400 tasks in the ARC-AGI public training v1 benchmark suite, your team will earn a score of `max(1, 2500 - length)` for a functionally correct program using `length` characters (or, to be more specific, bytes). Each program that is functionally *incorrect* — e.g., that doesn't compile, or fails for some < *input*, *output* > pair — will earn 0.001 points. This way, you'll be able to detect the overall number of failures from your total score alone.

## Submission File

You must submit a file named **submission.zip** containing at most one Python file per task:

```
task001.py
task002.py
....
task400.py
```

### Prizes

**Total Prizes Available: $100,000**

- First Place: $30,000
- Second Place: $20,000
- Third Place: $10,000
- Fourth Place: $5,000
- Fifth Place: $5,000
- Sixth Place: $5,000
- Seventh Place: $5,000
- Eighth Place: $5,000
- Ninth Place: $5,000
- Tenth Place: $5,000
- “Longest Leader” - $5,000 Awarded to the team holding 1st place on the leaderboard for the longest period of time between July 31, 2025 and October 30, 2025 11:59 PM UTC. In the event the competition needs to be restarted, the Longest Leader Prize dates shall be the new start and deadline of the competition.

### Timeline

- **July 31, 2025** - Start Date
- **October 23, 2025** - Entry deadline. You must accept the competition rules before this date in order to compete.
- **October 23, 2025** - Team Merger deadline. This is the last day participants may join or merge teams.
- **October 30, 2025** - Final submission deadline.

All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.

### NeurIPS 2025

![](https://neurips.cc/media/Press/NeurIPS_logo.png)  
This contest is part of the [NeurIPS 2025 Competition Track](https://blog.neurips.cc/2025/06/27/neurips-2025-competitions-announced/). Top submissions for the competition will be invited to give talks at a special session during the conference in San Diego, California. Attendance at the workshop is not required to participate in the competition; however, only those teams attending the special session will be considered to present their work.

Attendees presenting in person are responsible for all costs associated with travel, expenses, and fees to attend [NeurIPS 2025](https://neurips.cc/Conferences/2025).

Members of the top three winning teams will also be invited to collaborate with the competition organizers on a contest retrospective submitted to [PMLR](https://proceedings.mlr.press/).

### Other Resources

[![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F25050929%2F89002ee81b0e5a695ad0dbf56f8eeb00%2Farcprize.png?generation=1751579642212951&alt=media)](https://arcprize.org/) [![](https://code.golf/icon-180.png)](https://code.golf/tutorial#python)

- [arcprize.org](https://arcprize.org/) — Learn more about the ARC Prize Foundation and its mission to accelerate the development of Artificial General Intelligence (AGI)
- [code.golf](https://code.golf/tutorial#python) — Practice your code-fu skills by solving other programming challenges using the fewest number of characters

### Citation

Michael D. Moffitt, Divy Thakkar, Ryan Burnell, Orhan Firat, Walter Reade, Sohier Dane, and Addison Howard. NeurIPS 2025 - Google Code Golf Championship. https://kaggle.com/competitions/google-code-golf-2025, 2025. Kaggle.
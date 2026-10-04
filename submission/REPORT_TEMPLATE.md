# EE4705 MiniLab 1.3 Report Template

## Title
Command Your Robot Dog: An LLM + YOLO Powered Quadruped

## 1. Team and contribution split

- Group members:
  - Student A:
  - Student B:
  - Student C:
- Task allocation:
  - Task 1: 
  - Task 2: 
  - Task 3: 
  - Task 4: 
  - Task 5: 

## 2. AI usage declaration

State clearly:
- whether ChatGPT / LLM was used for code, report drafting, or debugging
- which parts were human-written
- which parts were AI-assisted
- whether all outputs were reviewed and verified before submission

## 3. Platform and environment

- Platform: MuJoCo quadruped simulation
- Robot model: <robot name / example model used>
- Operating system: <Windows / Linux / macOS>
- Python version: <e.g., 3.11 or 3.12>
- Key dependencies: <mujoco, numpy, pillow, pyyaml, pytest>
- API-key handling: <environment variable, .env file, or no API used>

## 4. Task 1: LLM + robot control comparison

### 4.1 Overview
Compare at least three approaches for connecting language models to robots.

### 4.2 Approaches to compare
- Structured-output parsing (JSON / function calling)
- SayCan / Code-as-Policies / similar planner-style approach
- VLM or vision-grounded / ROS-based approach

### 4.3 Two-rate architecture
Explain:
- slow cloud LLM planner
- fast local control loop
- why this split is useful for a quadruped robot

### 4.4 Reinforcement-learning policy note
Briefly explain how the local walking policy is learned and how sim-to-sim / sim-to-real reasoning applies.

### 4.5 References
Include at least 3–5 references in IEEE/APA format.

---

## 5. Task 2: platform setup, scene, and motion skills

### 5.1 Platform setup
Describe the base repo and the MuJoCo runtime setup.

### 5.2 Scene design
- custom MJCF scene created
- total object count: <n>
- classes used: <e.g., chair, sports ball, stop sign>
- color variants: <e.g., green chair, red chair>

### 5.3 Motion-skills API
Include the implemented functions:

```python
move(vx, vy, wz, duration)
turn(angle_deg)
```

Explain how yaw feedback is used and how motion is executed in simulation.

### 5.4 Demo notes
- native viewer mode used
- browser-based GUI mode used
- at least 3 terrain maps switched
- 3 onboard cameras inspected

### 5.5 Results
Describe the running behavior and include screenshots or short notes from the demo video.

---

## 6. Task 3: typed English motion commands via LLM parser

### 6.1 Parser overview
Explain how free-form English commands are converted to structured actions.

### 6.2 Supported command types
- movement commands
- turning commands
- multi-step commands
- rejection of empty / unsafe / non-English inputs

### 6.3 Implementation details
Describe:
- JSON schema / function call / schema-constrained decoding
- multi-turn conversation history
- command validation rules

### 6.4 Evaluation
- at least 20 command utterances tested
- at least two LLM services or two parser modes compared
- show a representative table:

| Test ID | Input | Expected Output | Result |
|---|---|---|---|
| 1 | "walk forward for three seconds" | accepted | pass |
| 2 | "turn back" | accepted | pass |
| 3 | "fly to the roof" | rejected | pass |

### 6.5 Video note
State that `Video_Task3` shows the terminal, multi-action command execution, and one rejection example.

---

## 7. Task 4: YOLO object search and approach

### 7.1 Detection pipeline
Describe:
- RGB frame capture from the onboard camera
- YOLO-based detection or fallback validation pipeline
- color grounding using HSV or bounding-box logic

### 7.2 Target approach logic
Explain the algorithm:
1. detect target class and color
2. if not visible, rotate and re-detect
3. if visible, steer to the object center
4. walk forward until the target is within the required distance
5. stop and log success

### 7.3 Success criteria
The report must state that the controller stops only when:
- class and color match
- object is within 0.80 m in the x-y plane
- `[FOUND] class=... color=... t=... d=...` is logged

### 7.4 Evaluation
- at least 10 trials
- different starting poses
- different objects

### 7.5 Results table

| Trial | Target | Start Pose | Success | Notes |
|---|---|---|---|---|
| 1 | green chair | (0,0) | yes | centered search |
| 2 | red ball | (1,1) | yes | rotated before approach |

### 7.6 Video note
State that `Video_Task4` shows the typed command, search behavior, detection, found condition, and final mission status.

---

## 8. Task 5: final submission package

The final zip file includes:
- report PDF and report source
- source code
- install/run instructions
- custom scene and object files
- prompt files
- three videos
- AI usage declaration and contribution split

## 9. Discussion and limitations

Discuss:
- simulation-only nature of the project
- no physical robot available
- fallback methods if a real LLM or YOLO pipeline was not available
- practical issues such as camera noise, target drift, or detection thresholds

## 10. Conclusion
Summarize the system at a high level and state the main outcome of the lab.

## Appendix A: command log sample

```text
[CMD] accepted actions=walk forward for three seconds, turn back
[FOUND] class=chair color=green t=3.14 d=0.64
```

## Appendix B: environment and run commands

```bash
python -m pip install -e .[test]
python eg/play.py --gui
python eg/minilab_1_3.py
```

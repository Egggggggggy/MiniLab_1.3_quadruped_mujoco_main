# EE4705 MiniLab 1.3 Run Book

## 1. Environment setup

```bash
cd <project_root>
python -m pip install -e .[test]
python -m pytest -q
```

For microphone speech input, install the optional speech extra:

```powershell
python -m pip install -e ".[test,speech]"
```

## 2. Task 2: platurn right 45 degrees
turn left 45 degrees
tform and motion skills

```bash
# native MuJoCo viewer
python eg/play.py

# native viewer plus threaded terminal chat
python eg/play.py --chat --chat-provider local --simulation-log simulation.log

# native viewer plus microphone speech input
python eg/play.py --speech --chat-provider local --simulation-log simulation.log

# native viewer plus Gemini chat
python eg/play.py --chat --chat-provider gemini --chat-model "Gemini 3 Flash Preview"

# native viewer plus OpenAI chat
python eg/play.py --chat --chat-provider openai --chat-model gpt-6-luna

# browser-panel GUI mode
python eg/play.py --gui
```

When `--chat` is enabled, periodic simulation telemetry is written to `simulation.log` instead of repeatedly printing over the command prompt. To monitor it separately, open a second PowerShell terminal in the project directory and run:

```powershell
Get-Content .\simulation.log -Wait
```

Capture:
- keyboard control
- map switching
- three onboard cameras
- robot driving in the example scene

With `--speech`, speak near the default microphone. Speech is transcribed by the optional `SpeechRecognition` Google Web Speech recognizer and passed into the same Task 3 parser and executor as typed commands. The expected terminal sequence is:

```text
[STT] microphone ready; speak an English command
[STT] heard=walk forward for three seconds
[CMD] accepted actions=move
[EXEC] 1/1 type=move
[DONE] 1/1 type=move
```

The `--speech` mode needs microphone permission and an internet connection for transcription. Type `quit` or say `quit` to close the simulation.

## 3. Task 3: typed English command parser

```bash
# smoke-test demo
python eg/minilab_1_3.py

# interactive multi-turn command chat
python eg/minilab_1_3.py --chat

# required 20-command evaluation
python eg/minilab_1_3.py --evaluate
```

The `--chat` mode runs parsing in a separate background thread and preserves conversation history. Type `quit` or `exit` to stop it. The `--evaluate` mode tests 20 accepted, rejected, unsafe, empty, and non-English utterances.

For real provider-backed structured output, set the API key and model as environment variables. OpenAI and Anthropic use their standard endpoints automatically; an OpenAI-compatible service can provide a custom endpoint. The implementation validates the returned JSON and falls back to the local parser if the provider is unavailable.

```powershell
# OpenAI
$env:OPENAI_API_KEY = "<enter locally>"
$env:MINILAB_LLM_MODEL_OPENAI = "<model-name>"
python eg/minilab_1_3.py --chat --provider openai

# Anthropic
$env:MINILAB_LLM_API_KEY_ANTHROPIC = "<enter locally>"
$env:MINILAB_LLM_MODEL_ANTHROPIC = "<model-name>"
python eg/minilab_1_3.py --evaluate --provider anthropic

# Google AI Studio / Gemini API
$env:GEMINI_API_KEY = "<enter locally>"
$env:MINILAB_LLM_MODEL_GEMINI = "<Gemini model name>"
python eg/minilab_1_3.py --chat --provider gemini
python eg/minilab_1_3.py --evaluate --provider gemini

# Any OpenAI-compatible service
$env:MINILAB_LLM_API_URL_OPENAI_COMPATIBLE = "https://<provider-endpoint>"
$env:MINILAB_LLM_API_KEY_OPENAI_COMPATIBLE = "<enter locally>"
$env:MINILAB_LLM_MODEL_OPENAI_COMPATIBLE = "<model-name>"
python eg/minilab_1_3.py --chat --provider openai-compatible

# Compare Gemini and OpenAI
python eg/minilab_1_3.py --compare gemini openai
```

`--compare` runs the same 20 cases against both configured providers. Use provider-specific variables such as `MINILAB_LLM_API_URL_OPENAI_COMPATIBLE` and `MINILAB_LLM_API_URL_ANTHROPIC` when their endpoints differ. Never commit API keys or place them directly in source code.

Record the terminal output showing:
- accepted commands
- multi-step motion parsing
- at least one rejected unsafe or invalid command

## 4. Task 4: object search and approach

Use the same platform, custom scene, and motion-skills API.

Expected sequence:
1. issue a typed command
2. rotate to search for the target object
3. detect object class and color
4. steer toward the center of the object
5. walk forward until the object is within the required distance
6. log `[FOUND] class=... color=... t=... d=...`

## 5. Required final run artifact list

- `Video_Task2.mp4`
- `Video_Task3.mp4`
- `Video_Task4.mp4`
- `report/EE4705_MiniLab_1_3_Report.pdf`
- `code/` source archive or full project copy
- `scene/` custom scene files
- `prompts/` prompt files and parser config

## 6. Suggested recording procedure

1. Start the simulation with the chosen map and object scene.
2. Open a separate terminal for the command loop if needed.
3. Run the parser and issue a multi-step command.
4. Record the terminal output and the simulation viewer.
5. Save the output as one video per task.

## 7. Suggested final environment note

```text
Platform: MuJoCo + Python
Python version: 3.11 / 3.12
Dependencies: mujoco, numpy, pillow, pyyaml, pytest
API keys: stored as environment variables; never hard-coded into source files
```

## 8. Optional fallback note

If the real LLM or YOLO service is unavailable, clearly state the fallback in the report:
- local deterministic parser for structured command parsing
- CPU-only object detection fallback for validation
- same scene, same motion-skills API, same lab logic preserved

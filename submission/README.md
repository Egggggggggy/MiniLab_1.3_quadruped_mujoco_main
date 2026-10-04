# EE4705 MiniLab 1.3 Submission Package

This folder is the final submission layout for the MiniLab 1.3 package. It is structured to match the PDF requirements for a group submission and uses the code already present in this repository.

## Recommended final zip structure

```text
EE4705_MiniLab_1_3_<GroupName>.zip
├── report/
│   ├── EE4705_MiniLab_1_3_Report.pdf
│   ├── EE4705_MiniLab_1_3_Report.md
│   ├── AI_Usage_Declaration.pdf
│   └── Contribution_Split.md
├── code/
│   ├── README.md
│   ├── eg/
│   ├── src/
│   ├── tests/
│   ├── pyproject.toml
│   ├── setup.cfg
│   └── requirements.txt   # optional if exported for convenience
├── videos/
│   ├── Video_Task2.mp4
│   ├── Video_Task3.mp4
│   └── Video_Task4.mp4
├── scene/
│   ├── minilab_scene.json
│   ├── build_scene_xml.py or equivalent
│   └── generated_scene.xml
├── prompts/
│   ├── command_parser_prompt.txt
│   ├── system_prompt.txt
│   └── rejection_rules.txt
├── install_and_run.md
├── environment.txt
└── submission_notes.txt
```

## Submission checklist

- Report PDF included
- Source code included
- Install and run instructions included
- Scene and object config files included
- Three videos included: Video_Task2, Video_Task3, Video_Task4
- Contribution split included
- AI usage declaration included
- Python version, dependencies, and API-key handling described

## Minimum required environment

```bash
python --version
python -m pip install -e .[test]
```

## Run commands to include in the final submission

```bash
# install
python -m pip install -e .[test]

# Task 2 baseline example
python eg/play.py

# Task 2 browser GUI mode
python eg/play.py --gui

# Demo command parser and scene generation
python eg/minilab_1_3.py
```

## Notes

- The included MuJoCo runtime example is the base platform.
- The custom object scene and object-search logic should be described in the report and included in the zip.
- If you use a real LLM API service, document the model name, API key handling, rate limits, and fallback path in the environment notes.
- If the system falls back to a local deterministic parser, document that clearly and state the reason in the report.

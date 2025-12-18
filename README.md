# Brand Video Frame & Audio Smart Tagging Pipeline

## Overview
This project is a modular and extensible pipeline for analyzing advertisement videos, extracting audio and key video frames, and using GPT-4o (vision & text) to fill a CSV (or JSON) of marketing/branding tags for each creative. The pipeline supports brand knowledge context for better accuracy, handles large-scale CSV batch tagging from video URLs, and automatic brand knowledge creation.

## Key Features

- **Video analysis**: Extracts audio and various kinds of image framings (regular, diverse, people-focused, grouped, etc.).
- **Tagging by LLM**: Feeds image & audio samples + smart question-JSON template to a GPT-4o API for robust visual+language analysis.
- **Brand context**: Optionally injects brand-specific context to improve LLM answers, via brand_knowledge JSON files.
- **End-to-end automation**: Download videos, preprocess, tag, and export results with a main script.
- **Extensible**: Modular design—easy to add new extractors, prompt logic, or LLM providers.

## Table of Contents
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Batch CSV Tagging (from URLs)](#1-batch-csv-tagging-from-urls)
  - [Bulk Video Folder Tagging](#2-bulk-video-folder-tagging)
- [Configuration](#configuration)
- [Core Concepts](#core-concepts)
  - [Frame Extraction Methods](#frame-extraction-methods)
  - [Prompt Tag Template](#prompt-tag-template)
  - [Brand Knowledge Context](#brand-knowledge-context)
- [Advanced](#advanced)
  - [How It Works – Pipeline Logic](#how-it-works--pipeline-logic)
  - [How to Add New Tags or Frames](#how-to-add-new-tags-or-frames)
- [Credits & License](#credits--license)
- [Contact](#contact)

## Project Structure
```
.
├── README.md
├── requirements.txt
├── config/
│   ├── config.yml
│   ├── tag_mapping.json        # master prompt/field config
│   ├── brand_knowledge/
│   │    └── [brand].json      # context files (auto-created if not present)
│   └── ...
├── audio_extractors/
├── frame_extractors/
├── data_filling/
│   ├── model/                  # LLM, GPT prompt construction, batching, ...
│   ├── pipeline/               # master scripts: process_video.py, create_csv_from_links.py
│   └── ...
└── ...
```

## Installation
```bash
git clone <repo_url>
cd <repo>
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Minimum Python: 3.10+ (required by many libs)

## Usage

### 1. Batch CSV Tagging (from URLs)
If you have a CSV with video URLs and brand names, generate a CSV of GPT-extracted tags (one row per video):

Edit `config/config.yml` for your setup (see [Configuration](#configuration) section).

```bash
python -m data_filling.pipeline.create_csv_from_links config/config.yml
```

Output: CSV in your `output_dir` with all tagging fields per video.

### 2. Bulk Video Folder Tagging
If you have a local folder of videos (and a mapping file to brands):

```bash
python -m data_filling.pipeline.process_video config/config.yml
```

Each video gets a JSON results file in `outputs_arch/`.

## Configuration
All settings are in `config/config.yml`:

Example:
```yaml
template_path: config/tag_mapping.json
input_video_dir: data/input_videos
output_dir: data/
brand_map_path: config/video_to_brand.json
brands_knowledge_dir: config/brand_knowledge

media_csv_path: data/csv/comms_assets_10.csv
media_url_column: Media URL
brand_column: Parent Brand

openai_api_key: sk-XXXXXXX
openai_model: gpt-4o
openai_model_transcript: gpt-4o-transcribe
openai_model_knowledge: gpt-4o-search-preview-2025-03-11
verify_ssl: false
```

Configure:

- Your OpenAI/GPT API key (& model)
- File paths for input/output
- Paths to brand knowledge/csv/template
- Optionally: set SSL off for dev/debug

## Core Concepts

### Frame Extraction Methods
- `regular_1s`: Simple 1FPS sampling
- `regular_0_5s`: 2FPS sampling
- `mif`: Maximum information frames—frames with highest visual variation
- `people_1s`/`people_0_5s`: Only frames containing (detected) people
- `people_mif`: Diverse, high-person-weight frames
- `regroup_1s`: Grouped collages for context
- `audio`: Extracted audio from video

Each is extracted and saved when processing a video—used for tag logic.

### Prompt Tag Template
The heart of the system. Tags/questions, output keys, splitting logic, and mapping are in `config/tag_mapping.json`.

Each field/column includes:
- Prompt text (`prompt_ai`)
- Key (`key`)
- Frame selector method
- AI split logic (`"or"`, `"mean"`, etc.)
- Output type (`"1"/"0"`, `int`, etc.)
- Optional: audio, brand info injection

To add tags: edit `tag_mapping.json` (see examples in the file).

### Brand Knowledge Context
Each brand can have a JSON context file (`brand_knowledge/[brand].json`) (auto-created via LLM prompt if missing).

Gives extra knowledge for more robust tagging (brand logo/colors/elements).

Instantly created with the correct structure if not present.

## Advanced

### How It Works — Pipeline Logic
Input: list of videos (local or URLs) + brands

For each video:
- Downloads (if URL)
- Extracts all relevant frame sets & audio files
- (If needed) Generates GPT-4o-based brand knowledge file
- For each batch of tags sharing extraction method:
  - Builds prompt (mixing frames, brand context, &/or audio transcription as needed)
  - Feeds prompt+images to OpenAI API
  - Parses output, merges results by rules (e.g. "count-mean" for %-fields)
- Outputs results (final CSV or per-video JSON)
- Cleans up extracted temp files

All frame/audio extractor logic is fully pluggable.

### How to Add New Tags or Frames
Add a new entry to `config/tag_mapping.json`:
- Write a new `prompt_ai`
- Map key, frame method, accepted values, etc.

(If needed) Add a new frame extraction method:
- Implement in `frame_extractors/`
- Plug into `extract_all_framings` in `tools_pipeline/extract_framings.py`

(Optional) Add brand knowledge fields (will be injected if `prompt_additional` is set in tag mapping).

## Gotchas & Requirements
- **API keys**: Needs OpenAI account with GPT-4o+Vision support.
- **Performance**: Pipeline auto-batches to fit prompt/image limits.
- **Inference Cost**: Vision API usage can be expensive at scale.
- **Media I/O**: Input videos can be from URLs or local storage.
- **Dependencies**: See `requirements.txt` – ensure compatibility for OpenCV, MoviePy, ultralytics (YOLO) for people detection, etc.

## Credits & License
Developed by [authors/organization].

- Integrates OpenAI GPT-4o Vision + Text
- Frame extraction originally inspired by open-source/academic work on diverse frame selection
- License: MIT (modify as needed)

## Contact
Questions? Bugs? Ideas? Open an issue or contact contact@albandanet.fr .

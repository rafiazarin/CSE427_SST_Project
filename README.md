# CSE427 SST Project

## Project Title
Token-Efficient Multimodal Reasoning via Scene-to-Structured-Text Compression

## Week 3 Submission Goal
This checkpoint submission will include:
- Problem statement
- Project objectives
- Dataset identification
- Initial exploratory data analysis (EDA) with visuals
- Dataset preprocessing
- Three baseline implementations:
  1. Baseline VLM
  2. Full SST pipeline
  3. Question-aware SST pipeline

## Main Dataset
GQA

## Main Idea
Instead of giving the full image directly to a multimodal model, this project converts the image into a structured semantic representation containing:
- objects
- counts
- attributes
- relations
- visible text

This structured representation is then used for reasoning and compared against a direct image-based baseline.

## Project Structure
- `data/` → dataset files
- `notebooks/` → EDA and implementation notebook
- `src/` → reusable Python code
- `outputs/` → plots, predictions, and results
- `paper/` → report files

## Branches
- `main` → stable version
- `data-baseline` → dataset, EDA, baseline work
- `sst-pipeline` → semantic extraction and SST pipeline work
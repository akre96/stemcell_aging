# Stem Cell Aging Analysis tools
__Intended for use in the Rong Lu Lab__
Uses python 3.7.2

## Usage Instructions
### Setting up environment
  1. Load virtual environment manager of choice using python 3.7.x
      - Anaconda, virtualenv, etc.
  2. In the environment run:  
  `pip install -r requirements.txt`

### Reformatting step7 output
  1. Create new output directory  
  `mkdir step7_outut_reformat` 
  2. Place all step7 outputs to be analyzed in to a single folder
  3. In the root directory of this repository run:  
  `python step7_to_long_format.py -i \[INPUT_DIRECTORY_PATH\] -o \[OUTPUT_DIRECTORY_PATH\]`

Example:  
`mkdir long_format_outputs`  
`python step7_to_long_format.py -i ./inputs -o ./long_format_outputs`

### Consolidating reformatted data to one file
  1. In the root directory of this repository run:  
  `python consolidate_data.py -i \[LONG_FORMAT_INPUT_DIRECTORY_PATH\] -o \[OUTPUT_DIRECTORY_PATH\]`  
Example:  
`python consolidate_data.py -i long_format_outputs`  

- Output directory optional, defaults to current directory (generally repository root)
- This step names the output file the same as the first input file it finds, but replaces M### with
  M_all in the file name

### Plotting Data
  1. Open plot_data.py
  2. Edit main() function to change input file to desired file to analyze 
  3. Call different plotting functions from above as desired

  - TODO: Refactor to just be a command line called script


### Testing
In the root directory of the repository run:  
`pytest`

## Directory Structure
TODO: Fill out

---
By Samir Akre
 
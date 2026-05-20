
#character-level perturbation rate using the TextAttack library and DeepWordBug

#A black-box adversarial text attack that flips model predictions by applying small character-level edits (for example, insertion, deletion, substitution, or swap) to high-impact words while preserving human readability.



# A4S Evaluation module

# Quickstart for Evaluation module

https://github.com/Hala-com-max/a4s

Installation

Instructions for setting up the project locally:

# Clone the repository
git clone https://github.com/Hala-com-max/a4s.git


# Change into project directory
cd a4s


# Create a Python virtual environment
python -m venv uv


# Activate the virtual environment
# On Windows:
uv\Scripts\activate
# On Linux / Mac:
source uv/bin/activate


# Upgrade pip (optional but recommended)
pip install --upgrade pip


# Install dependencies from requirements.txt
pip install -r requirements.txt


# Install additional libraries
pip install textattack
pip install python-Levenshtein


# Pull the Ollama model (1.3GB)
ollama pull model_name:1.3gb

#test
use _uv run pytest -s _

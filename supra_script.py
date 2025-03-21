from open_lm.open_lm_hf import *

# Import necessary libraries for evaluation
import lm_eval
from lm_eval.__main__ import cli_evaluate

# Now run the evaluation
if __name__ == "__main__":
    # This will ensure the imports are available in all processes
    cli_evaluate()


| File | Function | Result stored |
|---|---|---|
| `shared_data.py` | preprocessing and Aggregation(including candidate site aggregation) | `python shared_data.py`  |
| `shared_data_part2.py` | preprocessing and Aggregation(including candidate site aggregation) | `python shared_data.py`  |
| `part2_deterministic.py` | Part 2 baseline model + Part 2b optimized version (z = continuous) | `python part2_deterministic.py` → `part2_output.txt` |
| `part3_stochastic.py` | Part 3 baseline stochastic model (z = binary) | `python part3_stochastic.py` → `part3_output.txt` |
| `part3b_optimised.py` | Part 3b optimized stochastic model (z = continuous) | `python part3b_optimised.py` → `part3b_output.txt` |


- All of the code are shown in the 'assignment' folder
- The part3b model failed to optimize, as the same as Part2b model with z=continuous. Therefore, this case is not considered in the report and has be removed in subsequent revisions.
- `shared_data.py` is used for part3, and  `shared_data_part2' is used for part2
- Each module runs independently and loads the shared data via from shared_data(or shared_data_part2) import load_all.
- Each module writes its output both to the terminal and to the corresponding *_output.txt file.


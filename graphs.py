import matplotlib.pyplot as plt
from collections import defaultdict
import os

def load_success_dict(filename):
    success_dict = defaultdict(int)
    if os.path.exists(filename):
        with open(filename, "r") as f:
            for line in f:
                q, count = line.split()
                success_dict[float(q)] = int(count)
    return success_dict

num_tests = 20  # Each success count needs to be divided by this

# Load success dictionaries
success_dict_1 = load_success_dict("success_dict_bot_1.txt")
success_dict_2 = load_success_dict("success_dict_bot_2.txt")
success_dict_3 = load_success_dict("success_dict_bot_3.txt")

# Extract q values and success probabilities
q_values = sorted(success_dict_1.keys())

success_probs_1 = [success_dict_1[q] / num_tests for q in q_values]
success_probs_2 = [success_dict_2[q] / num_tests for q in q_values]
success_probs_3 = [success_dict_3[q] / num_tests for q in q_values]

# Plot the results
plt.figure(figsize=(8, 5))
plt.plot(q_values, success_probs_1, color = 'green', marker='o', linestyle='-', label="Bot 1")
plt.plot(q_values, success_probs_2, color = 'red', marker='o', linestyle='--', label="Bot 2")
plt.plot(q_values, success_probs_3, color = 'blue', marker='o', linestyle='-.', label="Bot 3")

plt.xlabel('q')
plt.ylabel('Success Probability')
plt.ylim(-0.25,1.25)
plt.title('Success Probability vs q for Different Bots')
plt.legend()
plt.grid(True)
plt.show()

import random
import string
import os

path = "input_files/"
os.makedirs(path, exist_ok=True)

tests_number = 10

for test_number in range(tests_number):
    name_of_file = os.path.join(path, f"test_{test_number + 1:02}.in")

    text_size = random.randint(100000, 1000000)
    main_string = ''.join(random.choice(string.ascii_lowercase) for _ in range(text_size))

    with open(name_of_file, 'w') as file:
        file.write(f"{text_size} {main_string}\n")

        patterns_count = random.randint(1000, 10000)
        file.write(f"{patterns_count}\n")

        for _ in range(patterns_count):
            if random.random() > 0.5:
                start = random.randint(0, text_size - 2)
                max_len = min(text_size - start, 2000)
                pat_len = random.randint(1, max_len)
                pattern = main_string[start : start + pat_len]
            else:
                pat_len = random.randint(1, min(text_size, 2000))
                pattern = ''.join(random.choice(string.ascii_lowercase) for _ in range(pat_len))

            file.write(f"{pat_len} {pattern}\n")
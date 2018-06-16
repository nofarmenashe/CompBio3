import random

with open("nn0.txt", mode='rb') as file:
    data = file.read().splitlines()

random.shuffle(data)
training_set = data[:15000]
validation_set = data[-5000:]

with open('nn0_training.txt', 'w') as train_file:
    for i, line in enumerate(training_set):
        train_file.write(line)
        if i < 15000 - 1:
            train_file.write("\n")

with open('nn0_validation.txt', 'w') as val_file:
    for i, line in enumerate(validation_set):
        val_file.write(line)
        if i < 5000 - 1:
            val_file.write("\n")

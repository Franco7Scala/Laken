import os
import csv


root_dir = "/Desktop/mnist_results"


def build_csv(start_index, source_file, destination_file):
    with open(source_file) as source_f:
        lines = [line for line in source_f]
        with open(destination_file, "a") as destination_f:
            writer = csv.writer(destination_f, delimiter=";")
            epoque = 0
            writer.writerow(["Epoque", "Loss", "Accuracy", "Precision", "Recall", "F1", "ElapsedTime"])
            writer.writerow([epoque,
                             str(lines[start_index][6:]).replace(".", ","),
                             str(lines[start_index + 1][10:]).replace(".", ","),
                             str(lines[start_index + 2][11:]).replace(".", ","),
                             str(lines[start_index + 3][8:]).replace(".", ","),
                             str(lines[start_index + 4][4:]).replace(".", ","),
                             "None"])
            for i in range(start_index + 6, len(lines) - 2, 7):
                epoque += 1
                writer.writerow([epoque,
                                 str(lines[i + 1][6:]).replace(".", ","),
                                 str(lines[i + 2][10:]).replace(".", ","),
                                 str(lines[i + 3][11:]).replace(".", ","),
                                 str(lines[i + 4][8:]).replace(".", ","),
                                 str(lines[i + 5][4:]).replace(".", ","),
                                 lines[i][14:]])


if __name__ == "__main__":
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            current_source_file = os.path.join(subdir, file)
            if ".DS_Store" not in current_source_file:
                current_destination_file = current_source_file[:current_source_file.rfind(".") + 1].replace("results", "csv_results") + "csv"
                os.makedirs(current_destination_file[:current_destination_file.rfind("/")], exist_ok=True)
                if "+" in current_destination_file:
                    start_index = 4

                else:
                    start_index = 3

                if "qbc" in current_destination_file:
                    start_index += 2

                build_csv(start_index, current_source_file, current_destination_file)

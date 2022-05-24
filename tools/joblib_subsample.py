import numpy as np
import joblib
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('input_file', type=str)
parser.add_argument('output_file', type=str)
parser.add_argument('--start', type=int, default=None)
parser.add_argument('--end', type=int, default=None)
parser.add_argument('--step', type=int, default=None)
args = parser.parse_args()

data = joblib.load(args.input_file)
data_sub = data[args.start:args.end:args.step]

print(f'saving {len(data_sub)} samples to {args.output_file}')
joblib.dump(data_sub, args.output_file)

""" 
This script generates the cross-study validation table.
It launches a script (e.g. train_from_combined.py) that train ML model(s) using a single source
and runs predictions (inference) on a specified set of other sources.
"""
# python -m pdb src/models/launch_cross_study_val.py
import sys
import argparse
import train_from_combined

cross_study_sets = [
    {'tr_src': ['ctrp'],
     'te_src': ['ctrp', 'gdsc', 'ccle', 'gcsi']},

    {'tr_src': ['gdsc'],
     'te_src': ['ctrp', 'gdsc', 'ccle', 'gcsi']},

    {'tr_src': ['ccle'],
     'te_src': ['ccle', 'gcsi', 'gdsc']}, # ['ctrp', 'gdsc', 'ccle', 'gcsi']

    {'tr_src': ['gcsi'],
     'te_src': ['ctrp', 'gdsc', 'ccle', 'gcsi']},
]

# Single run
idx = 2
train_from_combined.main(['-tr', *cross_study_sets[idx]['tr_src'],
                          '-te', *cross_study_sets[idx]['te_src']])

# Multiple runs
for i in range(len(cross_study_sets)):
    train_from_combined.main(['-tr', *cross_study_sets[i]['tr_src'],
                              '-te', *cross_study_sets[i]['te_src']])


# ================================================================================================
# parser = argparse.ArgumentParser('Launcher of cross-study validation.')

# # Select train and test (inference) sources
# parser.add_argument("-tr", "--train_sources", nargs="+",
#     default=["ccle"], choices=["ccle", "gcsi", "gdsc", "ctrp"],
#     help="Data sources to use for training.")
# parser.add_argument("-te", "--test_sources", nargs="+",
#     default=["ccle"], choices=["ccle", "gcsi", "gdsc", "ctrp"],
#     help="Data sources to use for testing.")

# Parse args and launch training script
# args = parser.parse_args()
# print(args)
# train_from_combined.main(args)

# Launch training script
# train_from_combined.main(parser)

# if __name__ == '__main__':
#     train_from_combined.main(sys.argv[1:])




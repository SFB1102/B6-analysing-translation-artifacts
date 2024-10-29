#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Automatic evaluation with BLEU, chrF++ and BLEURT for WMT-SLT 2022
    Confidence intervals obtained via bootstrap resampling
"""

import sys, os
import argparse
import re

import random
import numpy as np

from sacrebleu.metrics import BLEU, CHRF
from bleurt import score

def ci_bs(distribution, n, confLevel):
    ''' Calculates confidence intervals for distribution at confLevel after the 
        generation of n boostrapped samples 
    '''

    bsScores = np.zeros(n)
    size = len(distribution)
    random.seed(16) 
    for i in range(0, n):
        # generate random numbers with repetitions, to extract the indexes of the sysScores array
        bootstrapedSys = np.array([distribution[random.randint(0,size-1)] for x in range(size)])
        # scores for all the bootstraped versions
        # this works because we assume the MT metric is calculated at sentence level
        bsScores[i] = np.mean(bootstrapedSys,0)

    # we assume distribution of the sample mean is normally distributed
    # number of bootstraps > 100
    mean = np.mean(bsScores,0)
    stdDev = np.std(bsScores,0,ddof=1)
    # Because it is a bootstraped distribution
    alpha = (100-confLevel)/2
    confidenceInterval = np.percentile(bsScores,[alpha,100-alpha])

    return (mean, mean-confidenceInterval[0])
    

def main(args=None):
    nameINfolder = "inputs"
    files = sorted(os.listdir(nameINfolder))

    REFERENCE = os.path.join(nameINfolder, "human.en")

    ''' Locate the reference '''
    referenceFile = REFERENCE
    reference = [[""]] * 2
    with open(referenceFile, "r") as file:
        reference[0] = file.read().split("\n")
    
    ''' Metric calculation for each submission '''
    bleu = BLEU()
    outputHeader = '# submission, BLEUall\n'
    output = open("scores.csv", 'w')
    scores_all = ""
    for filename in files:
        if filename == REFERENCE:
            continue
        elif not filename.endswith(".en"):
            continue
        with open(os.path.join(nameINfolder, filename), "r") as file:
           hypothesis = file.read().split("\n")
        team = re.search(r'(.+)\.en', filename).group(1)
        scores = team

        res = bleu.corpus_score(hypothesis[0:-1], [reference[0][0:-1]], 1000) 
        value = re.search(r'\(μ = (.*?)\)',str(res)).group(1)
        scores = scores + ', ' + value
        print(scores)
        scores_all += scores + "\n"

    output.write(scores_all)
    output.close()
        
if __name__ == "__main__":
   main()

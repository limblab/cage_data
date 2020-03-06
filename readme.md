# Python codes for loading and managing M1 neural data and EMG data collected in the Miller Lab's plastic telemetry cage

## Overview
We are using Blackrock Cerebus for M1 data recording and DSPW RCB module for EMG recording on freely moving monkeys wirelessly. Since data format and recording contexts are very different from those done inside the lab, here we don't use the way for data loading and managing in the `cds`. Instead, we built a framework for such processing in `Python`. This repository contains the core codes and some examples. Additionally, for free reaching data recorded in lab, these codes can also be used.

In previous versions of these codes, we tried to directly read Blackrock `.nev` files in `Python`, but it would take more than 30 minutes to read a single 15-minute file due to the inefficiency of Blackrock `BRPY` package. As an altenative solution, here we first convert Blackrock `.nev` files into MATLAB `.mat` files by using the codes under `./MATLAB/`, then read the `.mat` file in `Python`.

The codes also include functions for semi-automatic artifacts rejection based on waveform features. The codes can also read data files with spike sorting results.

## Details about how to use these codes and designing concerns to be continued, check the examples first if you are interested


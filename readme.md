# Python codes for loading and managing M1 neural data and EMG data collected in the Miller Lab's plastic telemetry cage

## Overview
We are using [Blackrock Cerebus](https://www.blackrockmicro.com/) for M1 data recording and DSPW RCB module for EMG recording on freely moving monkeys wirelessly. Since data format and recording contexts are very different from those done inside the lab, here we don't use the way for data loading and managing in the `cds`. Instead, we built a framework for such processing in `Python`. This repository contains the core codes and some examples. Additionally, for free reaching data recorded in lab, these codes are also useful.

In very early versions of these codes, we tried to directly read Blackrock `.nev` files in `Python`, but it would take more than 30 minutes to read a single 15-minute file due to the inefficiency of Blackrock `BRPY` package. As an altenative solution, we first converted Blackrock `.nev` files into MATLAB `.mat` files, then read the `.mat` file in `Python`. Fortunately, Blackrock has an updated version of [`BRPY`](https://github.com/BlackrockMicrosystems/Python-Utilities) recently and it works pretty well. Therefore, now we can skip the "MATLAB" step and use `BRPY` to read the `.nev` files directly.

The codes also include functions for semi-automatic artifacts rejection based on waveform features. Besides, the codes can read data files with sorted single units.

## Details about how to use these codes and designing concerns to be continued, check the examples first if you are interested



## Included Code from other Sources
- Blackrock data loaders for .nev and .ns\* files from Blackrock Neurotech [https://blackrockneurotech.com/research/support/#manuals-and-software-downloads]
- Intan data loaders for .rhd files from Intan Technologies [https://intantech.com/downloads.html?tabSelect=Software]


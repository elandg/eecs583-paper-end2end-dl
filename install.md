# CLgen Install Guide

1. Install Ubuntu 16.04 through WSL. Since Ubuntu 16.04 is EOL, it can no longer be installed through the Microsoft Store. [Here](https://gist.github.com/xynova/87beae35688476efb2ee290d3926f5bb) is a quick tutorial on how to install it. [Here] (https://endjin.com/blog/2021/11/setting-up-multiple-wsl-distribution-instances) is an alternative method if you already have one version of wsl installed.

2. Run install-deps.sh.

3. Create a Python 3.7 virtual environment: 'python3.7 -m venv env'. Activate it: 'source env/bin/activate'.

4. Run install-cuda.sh.

5. 'cd clgen-0.4.1' and run 'make'.

6. Replace 'requirements.txt' and 'Makefile' with the versions in the home directory.

7. Run 'make' again.
